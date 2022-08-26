"""Wrapper for the SSVAE"""
import os
import json
import time
import torch
import argparse
import shutil
import numpy as np
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict
from sklearn.metrics import accuracy_score

from Generator.utils import to_var, idx2word, expierment_name
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from Generator.CVAE.model import CVAE_model
from Encoders.encoder import encoder
from Decoders.decoder import decoder
from metrics import calc_au, calc_mi


class CVAE(pl.LightningModule):

    def __init__(self, argdict, train, dev, test):
        super().__init__()
        self.argdict=argdict
        self.splits=['train', 'dev', 'test']
        self.datasets={'train':train, 'dev':dev, 'test':test}
        self.model, self.params=self.init_model_dataset()
        # optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)  # self.argdict.learning_rate)
        self.loss_function_basic=train.loss_function
        self.loss_function_ppl = torch.nn.CrossEntropyLoss(ignore_index=train.pad_idx, reduction='mean')




    def configure_optimizers(self):
        optimizer=torch.optim.Adam(self.model.parameters(), lr=0.001)
        return optimizer

    def init_model_dataset(self):
        enco=encoder(self.argdict)#vocab_size=self.datasets['train'].vocab_size, embedding_size=300, hidden_size=self.argdict['hidden_size'], latent_size=self.argdict['latent_size'])
        deco=decoder(self.argdict)

        params = dict(
            argdict=self.argdict,
            encoder=enco,
            decoder=deco
        )

        model = CVAE_model(**params)
        if torch.cuda.is_available():
            model = model.cuda()

        return model, params

    def kl_anneal_function(self, anneal_function, step, k, x0):
        if anneal_function == 'logistic':
            return float(1 / (1 + np.exp(-k * (step - x0))))
        elif anneal_function == 'linear':
            return min(1, step / x0)

    def loss_fn(self, logp, target, mean, logv, anneal_function, step, k):
        NLL = torch.nn.NLLLoss(ignore_index=self.datasetsLabelled['train'].pad_idx, reduction='sum')
        # cut-off unnecessary padding from target, and flatten
        target = target.contiguous().view(-1)
        logp = logp.view(-1, logp.size(2))

        # Negative Log Likelihood
        NLL_loss = NLL(logp, target)

        # KL Divergence
        KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        KL_weight = self.kl_anneal_function(anneal_function, step, k, self.dataset_length*self.argdict['x0'])
        return NLL_loss, KL_loss, KL_weight

    def training_step(self, batch, batch_idx):
        batch_size = batch['input'].size(0)

        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = to_var(v)

        # Forward pass
        logp, mean, logv, z = self.model(batch['input'], batch['label'])

        # loss calculation
        NLL_loss, KL_loss, KL_weight = self.loss_fn(logp, batch['target'], mean, logv, 'logistic', self.step,
                                                    0.0025)
        self.step+=1

        loss = (NLL_loss + KL_weight * KL_loss) / batch_size
        self.log("Loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("KL Div", KL_loss/batch_size, on_epoch=True, on_step=False, prog_bar=True)
        self.log("NLL", NLL_loss/batch_size, on_epoch=True, on_step=False, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        batch_size = batch['input'].size(0)

        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = to_var(v)

        # Forward pass
        logp, mean, logv, z = self.model(batch['input'], batch['label'])

        # loss calculation
        NLL_loss, KL_loss, KL_weight = self.loss_fn(logp, batch['target'], mean, logv, 'logistic', self.step,
                                                    0.0025)

        loss = (NLL_loss + KL_weight * KL_loss) / batch_size
        self.log("Loss Dev", loss, on_epoch=True, prog_bar=True)
        self.log("KL Div Dev", KL_loss/batch_size, on_epoch=True, prog_bar=True)
        self.log("NLL Dev", NLL_loss/batch_size, on_epoch=True, prog_bar=True)
        return loss

    def train_test(self):
        ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())

        print(self.model)

        self.trainer = pl.Trainer(gpus=self.argdict['gpus'], max_epochs=self.argdict['nb_epoch_algo'], accelerator="dp")

        train_loader = DataLoader(
            dataset=self.datasets["train"],
            batch_size=64,  # self.argdict.batch_size,
            shuffle=True,
            num_workers=cpu_count(),
            pin_memory=torch.cuda.is_available()
        )
        self.dataset_length = len(train_loader)
        dev_loader = DataLoader(
            dataset=self.datasets["dev"],
            batch_size=64,  # self.argdict.batch_size,
            shuffle=False,
            num_workers=cpu_count(),
            pin_memory=torch.cuda.is_available()
        )

        self.trainer.fit(self, train_loader, dev_loader)

        self.interpolate()
        self.generate_from_train()
        self.evaluate_accuracy()

    def test(self):
        data_loader = DataLoader(
            dataset=self.datasets['test'],
            batch_size=64,  # self.argdict.batch_size,
            shuffle=False,
            num_workers=cpu_count(),
            pin_memory=torch.cuda.is_available()
        )

        self.model.eval()


        Average_loss=[]
        Average_NLL=[]
        Average_KL_Div=[]
        MIs=[]
        mus=[]
        NLL_mean_for_ppl=[]
        for iteration, batch in enumerate(data_loader):

            # Forward pass
            logp, mean, logv, z = self.model(batch)
            
            #Keeping track of the means for AU
            mus.append(mean.detach().squeeze(0))
            batch_size = logp.shape[0]
            logp, target=self.datasets['train'].shape_for_loss_function(logp, batch['target'])
            NLL_loss, KL_loss= self.loss_fn(logp, target.to('cuda'),  mean, logv, 'logistic', self.step, 0.0025)

            NLL_mean=self.loss_function_ppl(logp, target.to('cuda'))

            loss = (NLL_loss +  KL_loss) / batch_size
            Average_loss.append(loss.item())
            Average_KL_Div.append(KL_loss.cpu().detach()/batch_size)
            Average_NLL.append(NLL_loss.cpu().detach())
            NLL_mean_for_ppl.append(NLL_mean.cpu().detach())
            # aggr=self.get_aggregate()
            MIs.append(calc_mi(z, mean, logv))
            # print(MIs)
            # fds

        # print(MIs)
        AU=calc_au(mus)
        encoded = self.encode()
        X = encoded['encoded_test']
        Y = encoded['labels_test']

        svc = LinearSVC()
        svc.fit(X, Y)
        sep=svc.score(X, Y)
        # print(AU)
        return {'Mean ELBO': np.mean(Average_loss), 'Mean LF' :np.mean(Average_NLL), 'Mean KL div' :np.mean(Average_KL_Div), 'PPL': {torch.exp(torch.mean(torch.Tensor(NLL_mean_for_ppl)))},
                'Separability': sep, 'MI': {np.mean(MIs)}, 'Active Units': AU[0]}

    def evaluate_accuracy(self):
        self.classifier.requires_grad=False
        # self.classifier.eval()
        self.model.eval()
        pos=torch.randn(500, self.argdict['latent_size']).cuda()
        pos=self.generate(pos, 1)
        neg = torch.randn(500, self.argdict['latent_size']).cuda()
        neg = self.generate(neg, 0)
        # print(pos[:5])
        # print(neg[:5])
        pos_label, conf=self.classifier.label(pos)
        neg_label, conf=self.classifier.label(neg)
        # print(pos_label)
        # print([1]*500)
        pos_acc=accuracy_score([1]*500, pos_label.cpu())
        neg_acc=accuracy_score([0]*500, neg_label.cpu())
        print(f"pos Accuracy : {pos_acc}")
        print(f"neg Accuracy : {neg_acc}")
        print(f"Accuracy average: {(pos_acc+neg_acc)/2}")
        # self.classifier.train()
        self.model.train()
        self.classifier.requires_grad=True
        # fds
    def encode(self):
        with torch.no_grad():
            dico={}
            for split in self.splits:
                data_loader = DataLoader(
                    dataset=self.datasetsLabelled[split],
                    batch_size=64,#self.argdict.batch_size,
                    shuffle=False,
                    num_workers=cpu_count(),
                    pin_memory=torch.cuda.is_available()
                )
                # Enable/Disable Dropout

                self.model.eval()
                # print(f"The dataset length is {len(data_loader.dataset)}")
                dataset = torch.zeros(len(data_loader.dataset), self.params['latent_size'])
                labels = torch.zeros(len(data_loader.dataset))
                counter = 0
                for iteration, batch in enumerate(data_loader):
                    # print("Oh la la banana")
                    batch_size = batch['input'].size(0)
                    # print(batch['input'].shape)
                    for k, v in batch.items():
                        if torch.is_tensor(v):
                            batch[k] = to_var(v)
                    #
                    # print(batch['input'])
                    # print(batch['input'].shape)
                    z = self.model.encode(batch['input'], batch['label'])
                    # print(batch_size)
                    # print(z.shape)
                    dataset[counter:counter + batch_size] = z
                    labels[counter:counter + batch_size] = batch['label']
                    counter += batch_size
                # print(dataset)
                dico[f"labels_{split}"]=labels
                dico[f"encoded_{split}"]=dataset
                # torch.save(labels, f"bin/labels_{split}.pt")
                # torch.save(dataset, f"bin/encoded_{split}.pt")
            return dico

    def generate(self, datapoints, cat):
        #Generates from fixed datapoints. If cat is not mentionned, generate from class unknown
        self.model.eval()
        self.model.to(self.device_for_gen)
        datapoints=datapoints.to(self.device_for_gen)
        samples, z = self.model.inference(z=datapoints, cat=cat)
        # print(samples)
        # print('----------SAMPLES----------')
        return idx2word(samples, i2w=self.datasetsLabelled['train'].get_i2w(), pad_idx=self.datasetsLabelled['train'].get_w2i()['<pad>'], eos_idx=self.datasetsLabelled['train'].get_w2i()['<eos>'])



    def generate_from_train(self):
        data_loader = DataLoader(
            dataset=self.datasets['train'],
            batch_size=2,  # self.argdict.batch_size,
            shuffle=False,
            num_workers=cpu_count(),
            pin_memory=torch.cuda.is_available()
        )

        self.model.eval()

        for iteration, batch in enumerate(data_loader):

            batch_size = batch['input'].size(0)
            batch_label= batch['label']
            # print(batch)
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = to_var(v)

            # Forward pass
            z = self.model.encode(batch['input'], batch['label'])
            samples, z = self.model.inference(z=z, cat=batch_label)
            gend=idx2word(samples, i2w=self.datasets['train'].get_i2w(), pad_idx=self.datasets['train'].get_w2i()['<pad>'],
                     eos_idx=self.datasets['train'].get_w2i()['<eos>'])
            # print(gend)
            for sent, gen in zip(batch['sentence'], gend):
                print(f"Original sentence: {sent}, generated: {gen}")
            break



    def interpolate(self, n=5):
        p0=to_var(torch.randn([1, self.argdict['latent_size']]))
        p1=to_var(torch.randn([1, self.argdict['latent_size']]))
        points=torch.zeros(n, self.argdict['latent_size'])
        points[0]=p0
        points[n-1]=p1
        for i in range(n):
            ratio=i/n
            px=(1-ratio)*p0+ratio*p1
            points[i]=px
        points=points.to(self.device_for_gen)
        self.model.to(self.device_for_gen)
        samples, z = self.model.inference(n=n, z=points.to(self.device_for_gen))
        generated = idx2word(samples, i2w=self.datasets['train'].get_i2w(), pad_idx=self.datasets['train'].get_w2i()['<pad>'], eos_idx=self.datasetsLabelled['train'].get_w2i()['<eos>'])
        print("Interpolation:")
        for sent in generated:
            print("------------------")
            print(sent)