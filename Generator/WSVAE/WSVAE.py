"""Wrapper for the WSVAE, as defined in https://arxiv.org/pdf/1703.00955.pdf, inspired by the github https://github.com/GBLin5566/toward-controlled-generation-of-text-pytorch"""
import os
import json
import time
import torch
import torch.nn.functional as F
import argparse
import shutil
import numpy as np
import pandas as pd
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict
from sklearn.metrics import accuracy_score
import copy

# from Generators.VAE.ptb import PTB
from Generator.utils import to_var, idx2word, expierment_name
from Generator.WSVAE.model import WSVAE_model
from Encoders.encoder import encoder
from Decoders.decoder import decoder
from Discriminators.discriminator import discriminator
from sklearn.svm import LinearSVC
from metrics import calc_mi, calc_au

class WSVAE():

    def __init__(self, argdict, train, dev, test):
        self.argdict=argdict
        self.splits=['train', 'dev', 'test']
        self.datasets={'train':train, 'dev':dev, 'test':test}
        self.model, self.params=self.init_model_dataset()
        # optimizers
        self.optimizer=torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.optimizer_encoder = torch.optim.Adam(self.model.encoder.parameters(), lr=0.001)  # self.argdict.learning_rate)
        self.optimizer_decoder = torch.optim.Adam(self.model.decoder.parameters(), lr=0.001)  # self.argdict.learning_rate)
        self.optimizer_discriminator = torch.optim.Adam(self.model.discriminator.parameters(), lr=0.001)  # self.argdict.learning_rate)
        self.loss_function_basic=train.loss_function
        # self.loss_function_discriminator=torch.nn.BCEWithLogitsLoss()
        self.loss_function_discriminator=torch.nn.CrossEntropyLoss()
        self.loss_function_encoder=torch.nn.L1Loss()
        if argdict['dataset'] in ['SST2']:
            self.loss_function_ppl=torch.nn.CrossEntropyLoss(ignore_index=train.pad_idx, reduction='mean')
        else:
            self.loss_function_ppl=self.loss_function_basic

    def init_model_dataset(self):
        self.step = 0
        self.epoch = 0

        enco=encoder(self.argdict)#vocab_size=self.datasets['train'].vocab_size, embedding_size=300, hidden_size=self.argdict['hidden_size'], latent_size=self.argdict['latent_size'])

        argdict_decoder=copy.deepcopy(self.argdict)
        #Need to modify here because the decoder takes more than just the latent space
        argdict_decoder['latent_size']=argdict_decoder['latent_size']
        deco=decoder(argdict_decoder)
        discri=discriminator(self.argdict)

        params = dict(
            argdict=self.argdict,
            encoder=enco,
            decoder=deco,
            discriminator=discri
        )
        model = WSVAE_model(**params)
        if torch.cuda.is_available():
            model = model.cuda()

        return model, params


    def loss_fn(self, logp, target,  mean, logv):
        # NLL = torch.nn.NLLLoss(ignore_index=self.datasets['train'].pad_idx, reduction='sum')
        # cut-off unnecessary padding from target, and flatten
        # target = target[:, :torch.max(length).item()].contiguous().view(-1)
        # target = target.contiguous().view(-1)
        # logp = logp.view(-1, logp.size(2))

        # Negative Log Likelihood
        NLL_loss = self.loss_function_basic(logp, target)
        # BCE = torch.nn.functional.binary_cross_entropy(logp, target.view(-1, 784), reduction='sum')
        # KL Divergence
        KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        # KL_weight = self.kl_anneal_function(anneal_function, step, k, self.dataset_length*self.argdict['x0'])

        return NLL_loss, KL_loss
        # return BCE, KL_loss

    def run_epoch(self, pretraining=False):
        for split in self.splits:

            data_loader = DataLoader(
                dataset=self.datasets[split],
                batch_size=64,  # self.argdict.batch_size,
                shuffle=split == 'train',
                num_workers=cpu_count(),
                pin_memory=torch.cuda.is_available()
            )

            # Enable/Disable Dropout
            if split == 'train':
                self.model.train()
                self.dataset_length=len(data_loader)
            else:
                self.model.eval()


            Average_loss=[]
            Average_NLL=[]
            Average_KL_Div=[]
            for iteration, batch in enumerate(data_loader):

                # Forward pass
                logp, mean, logv, z = self.model(batch, pretraining)
                batch_size = logp.shape[0]
                # print(batch_size)

                logp, target=self.datasets['train'].shape_for_loss_function(logp, batch['target'])
                NLL_loss, KL_loss= self.loss_fn(logp, target.to('cuda'),  mean, logv)

                loss = (NLL_loss +  KL_loss) / batch_size

                # backward + optimization
                if split == 'train':
                    # self.optimizer.zero_grad()
                    self.optimizer_encoder.zero_grad()
                    self.optimizer_decoder.zero_grad()
                    loss.backward()
                    # self.optimizer.step()
                    self.optimizer_encoder.step()
                    self.optimizer_decoder.step()
                    self.step += 1

                Average_loss.append(loss.item())
                Average_KL_Div.append(KL_loss.cpu().detach()/batch_size)
                Average_NLL.append(NLL_loss.cpu().detach()/batch_size)

            print(f"{split.upper()} Epoch {self.epoch}/{self.argdict['nb_epoch']}, Mean ELBO {np.mean(Average_loss)}, Mean LF {np.mean(Average_NLL)}, Mean KL div {np.mean(Average_KL_Div)}")

    def train_discriminator(self):
        #Equation 11: Sum of labelled and unlabelled examples. For now, only labelled
        print("Warning, not adapadted for semi supervised yet")
        for epoch in range(self.argdict['nb_epoch_discriminator']):
            for split in self.splits:
                data_loader = DataLoader(
                    dataset=self.datasets[split],
                    batch_size=64,  # self.argdict.batch_size,
                    shuffle=split == 'train',
                    num_workers=cpu_count(),
                    pin_memory=torch.cuda.is_available()
                )

                # Enable/Disable Dropout
                if split == 'train':
                    self.model.discriminator.train()
                else:
                    self.model.discriminator.eval()

                preds=[]
                ground_truth=[]
                losses=[]
                for iteration, batch in enumerate(data_loader):
                    output=self.model.discriminator.forward(batch['input'])
                    # print(output)
                    # print(output.shape)
                    # print(batch['label'])
                    loss=self.loss_function_discriminator(output, batch['label'].long().cuda())
                    # print(torch.argmax(torch.softmax(output, dim=-1).cpu().detach(), dim=-1))
                    preds.extend(torch.argmax(torch.softmax(output, dim=-1).cpu().detach(), dim=-1).tolist())
                    ground_truth.extend(batch['label'])
                    losses.append(loss.item())
                    if split == 'train':
                        # self.optimizer.zero_grad()
                        self.optimizer_discriminator.zero_grad()
                        loss.backward()
                        # self.optimizer.step()
                        self.optimizer_discriminator.step()
                # print(preds, ground_truth)
                # fds
                print(f"Epoch {epoch} split {split}, accuracy {accuracy_score(ground_truth, preds)}, loss {np.mean(losses)}")

    def train_generator(self):
        #Line 4 of the algo 1
        for split in self.splits:
            data_loader = DataLoader(
                dataset=self.datasets[split],
                batch_size=64,  # self.argdict.batch_size,
                shuffle=split == 'train',
                num_workers=cpu_count(),
                pin_memory=torch.cuda.is_available()
            )

            # Enable/Disable Dropout
            if split == 'train':
                self.model.train()
                self.dataset_length=len(data_loader)
            else:
                self.model.eval()


            Average_loss=[]
            Average_NLL=[]
            Average_KL_Div=[]
            preds = []
            ground_truth = []
            loss_reconstruction=[]
            for iteration, batch in enumerate(data_loader):

                # Forward pass
                logp, mean, logv, z = self.model(batch, pretraining=False)
                batch_size = logp.shape[0]
                # print(batch_size)
                if len(z.shape)==3:
                    z_normal, c = z[:, :, :-1], z[:, :, -1:]
                elif len(z.shape)==2:
                    z_normal, c = z[:,  :-1], z[:, -1:]
                else:
                    raise ValueError

                #Getting discriminator loss
                softmaxed_gumbeled = F.gumbel_softmax(logp, tau=1, hard=True, dim=-1)
                output_discriminator = self.model.discriminate(softmaxed_gumbeled)
                loss_discriminator = self.loss_function_discriminator(output_discriminator, batch['label'].long().cuda())
                preds.extend(torch.argmax(torch.softmax(output_discriminator, dim=-1), dim=-1).view(-1).tolist())
                ground_truth.extend(batch['label'].cpu().tolist())
                #Getting reconstruction loss
                #Why not optimize on the difference of mu and logv directly
                # print("WARNING THIS SHOULD BE DONE AFTER GETTING ENCODER LOSS")
                encoded_generated=self.model.encode(softmaxed_gumbeled)
                if len(z.shape)==3:
                    encoded_generated[:, :, :-1]
                elif len(z.shape)==2:
                    encoded_generated[:, :-1]
                else:
                    raise ValueError
                # print(z_normal_encoded.shape)
                # print(z_normal.shape)


                #Regular loss
                logp, target=self.datasets['train'].shape_for_loss_function(logp, batch['target'])
                NLL_loss, KL_loss= self.loss_fn(logp, target.to('cuda'),  mean, logv)

                #Equation 8: Lg= LVAE (eq 4 + Loss discriminator + loss attribute generator
                # loss_vae=(NLL_loss +  KL_loss) / batch_size

                loss_generator = self.loss_function_encoder(z_normal_encoded, z_normal)/batch_size
                loss_generator+=loss_discriminator/batch_size

                # backward + optimization
                #Optimization is difficult
                if split=='train':
                    self.optimizer_decoder.zero_grad()
                    loss_generator.backward()
                    self.optimizer_decoder.step()
                    self.step+=1

                Average_loss.append(loss_generator.item())
                Average_KL_Div.append(KL_loss.cpu().detach()/batch_size)
                Average_NLL.append(NLL_loss.cpu().detach()/batch_size)
            print(f"{split.upper()} Epoch {self.epoch}/{self.argdict['nb_epoch']}, Mean ELBO {np.mean(Average_loss)}, Mean LF {np.mean(Average_NLL)}, Mean KL div {np.mean(Average_KL_Div)}"
                  f"Acc recon {accuracy_score(ground_truth, preds)}")

    def train(self):
        print(self.model)
        save_model_path = os.path.join(self.argdict['path'], 'bin')
        # shutil.
        os.makedirs(save_model_path, exist_ok=True)

        #Pretraining the VAE
        #Instruction 1
        for epoch in range(self.argdict['nb_epoch']):
            self.epoch=epoch
            self.run_epoch(pretraining=True)

        #Until convergence
        print("Change for until convergence")
        for i in range(self.argdict['nb_epoch_fine_tuning']):
            print(f'---{i}----')
            #Train the discriminator by Eq 11 - Only labelled part for now
            #Instruction 3
            self.train_discriminator()
            #Train the generator with equation 8, which is sum of the VAE loss, the attribute c loss which is the expectation over p(z)p(c) that the discriminator can recover the correct c,
            #abd the z loss where we check whether the encoder can recover the correct z code
            #Because of the difficulty of path gradient calculation it is better to split it in two steps:
            self.run_epoch(pretraining=False)
            self.train_generator()

        self.interpolate()
        # self.generate_from_train()
        # self.create_graph()

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
            # print(batch)
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = to_var(v)

            # Forward pass
            logp, mean, logv, z = self.model(batch['input'], batch['length'])
            samples, z = self.model.inference(z=z)
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
        points=points.cuda()
        samples, z = self.model.inference(n=n, z=points)
        self.datasets['train'].process_generated(samples)
        # generated = idx2word(samples, i2w=self.datasets['train'].get_i2w(), pad_idx=self.datasets['train'].get_w2i()['<pad>'], eos_idx=self.datasets['train'].get_w2i()['<eos>'])
        # print("Interpolation:")
        # for sent in generated:
        #     print("------------------")
        #     print(sent)

    def encode(self):
        with torch.no_grad():
            dico={}
            for split in self.splits:
                data_loader = DataLoader(
                    dataset=self.datasets[split],
                    batch_size=64,#self.argdict.batch_size,
                    shuffle=False,
                    num_workers=cpu_count(),
                    pin_memory=torch.cuda.is_available()
                )
                # Enable/Disable Dropout

                self.model.eval()
                # print(f"The dataset length is {len(data_loader.dataset)}")
                dataset = torch.zeros(len(data_loader.dataset), self.argdict['latent_size'])
                labels = torch.zeros(len(data_loader.dataset))
                sentences=[]
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
                    z = self.model.encode(batch['input'])
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

    def generate(self, datapoints, labels):
        #Generates from fixed datapoints
        self.model.eval()

        samples, z = self.model.inference(z=datapoints)
        # print(samples)
        # print('----------SAMPLES----------')
        return idx2word(samples, i2w=self.datasets['train'].get_i2w(), pad_idx=self.datasets['train'].get_w2i()['<pad>'], eos_idx=self.datasets['train'].get_w2i()['<eos>'])

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
            NLL_loss, KL_loss= self.loss_fn(logp, target.to('cuda'),  mean, logv)

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