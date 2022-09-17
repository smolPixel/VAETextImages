"""Wrapper for the SSVAE"""
import os
import json
import time
import torch
import argparse
import shutil
import numpy as np
import pandas as pd
from multiprocessing import cpu_count
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict
from sklearn.metrics import accuracy_score

# from Generators.VAE.ptb import PTB
from Generator.utils import to_var, idx2word, expierment_name
from Generator.VAE_Annealing.model import VAE_Annealing_model
from Encoders.encoder import encoder
from Decoders.decoder import decoder
from metrics import calc_au, calc_mi

from sklearn.svm import LinearSVC

class VAE_Annealing():

    def __init__(self, argdict, train, dev, test):
        self.argdict=argdict
        self.splits=['train', 'dev', 'test']
        self.datasets={'train':train, 'dev':dev, 'test':test}
        self.model, self.params=self.init_model_dataset()
        # optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)  # self.argdict.learning_rate)
        self.loss_function_basic=train.loss_function
        if argdict['dataset'] in ['SST2']:
            self.loss_function_ppl=torch.nn.CrossEntropyLoss(ignore_index=train.pad_idx, reduction='mean')
        else:
            self.loss_function_ppl=self.loss_function_basic


    def init_model_dataset(self):
        self.step = 0
        self.epoch = 0

        enco=encoder(self.argdict)#vocab_size=self.datasets['train'].vocab_size, embedding_size=300, hidden_size=self.argdict['hidden_size'], latent_size=self.argdict['latent_size'])
        deco=decoder(self.argdict)

        params = dict(
            argdict=self.argdict,
            encoder=enco,
            decoder=deco
        )
        model = VAE_Annealing_model(**params)
        if torch.cuda.is_available():
            model = model.cuda()

        return model, params

    def kl_anneal_function(self, anneal_function, step, k, x0):
        if anneal_function == 'logistic':
            return float(1 / (1 + np.exp(-k * (step - x0))))
        elif anneal_function == 'linear':
            return min(1, step / x0)

    def loss_fn(self, logp, target,  mean, logv, anneal_function, step, k):
        NLL_loss = self.loss_function_basic(logp, target)
        # KL Divergence
        KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        KL_weight = self.kl_anneal_function(anneal_function, step, k, self.dataset_length*self.argdict['x0'])

        return NLL_loss, KL_loss, KL_weight

    def run_epoch(self):
        for split in self.splits:

            data_loader = DataLoader(
                dataset=self.datasets[split],
                batch_size=self.argdict['batch_size'],
                shuffle=split == 'train',
                num_workers=cpu_count(),
                pin_memory=torch.cuda.is_available()
            )

            # tracker = defaultdict(tensor)

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
                logp, mean, logv, z = self.model(batch)
                batch_size=logp.shape[0]
                logp, target=self.datasets['train'].shape_for_loss_function(logp, batch['target'])
                # NLL_loss, KL_loss, KL_weight = loss_fn(logp, batch['target'],
                #                                        batch['length'], mean, logv, self.argdict.anneal_function, step,
                #                                        self.argdict.k, self.argdict.x0)
                NLL_loss, KL_loss, KL_weight = self.loss_fn(logp, target,  mean, logv, 'logistic', self.step,
                                                            0.0025)
                # print(NLL_loss)
                # batch_size = logp.shape[0]
                # print(batch_size)
                loss = (NLL_loss + KL_weight * KL_loss) / batch_size

                # backward + optimization
                if split == 'train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.step += 1

                Average_loss.append(loss.item())
                Average_KL_Div.append(KL_loss.cpu().detach()/batch_size)
                Average_NLL.append(NLL_loss.cpu().detach()/batch_size)
            print(f"{split.upper()} Epoch {self.epoch}/{self.argdict['nb_epoch']}, Mean ELBO {np.mean(Average_loss)}, Mean NLL {np.mean(Average_NLL)}, Mean KL div {np.mean(Average_KL_Div)} KL Weight {KL_weight}")

    def test(self):
        data_loader = DataLoader(
            dataset=self.datasets['test'],
            batch_size=64,  # self.argdict.batch_size,
            shuffle=False,
            num_workers=cpu_count(),
            pin_memory=torch.cuda.is_available()
        )

        self.model.eval()

        Average_loss = []
        Average_NLL = []
        Average_KL_Div = []
        MIs = []
        mus = []
        NLL_mean_for_ppl = []
        for iteration, batch in enumerate(data_loader):
            # Forward pass
            logp, mean, logv, z = self.model(batch)
            # Keeping track of the means for AU
            mus.append(mean.detach().squeeze(0))
            batch_size = logp.shape[0]
            logp, target = self.datasets['train'].shape_for_loss_function(logp, batch['target'])
            NLL_loss, KL_loss, KL_weight = self.loss_fn(logp, target.to('cuda'), mean, logv, 'logistic', self.step, 0.0025)

            NLL_mean = self.loss_function_ppl(logp, target.to('cuda'))

            loss = (NLL_loss + KL_weight * KL_loss) / batch_size
            Average_loss.append(loss.item())
            Average_KL_Div.append(KL_loss.cpu().detach() / batch_size)
            Average_NLL.append(NLL_loss.cpu().detach()/batch_size)
            NLL_mean_for_ppl.append(NLL_mean.cpu().detach())
            # aggr=self.get_aggregate()
            MIs.append(calc_mi(z, mean, logv))
            # print(MIs)
            # fds

        # print(MIs)
        AU = calc_au(mus)
        encoded = self.encode()
        X = encoded['encoded_test']
        Y = encoded['labels_test']

        svc = LinearSVC()
        svc.fit(X, Y)
        sep = svc.score(X, Y)
        # print(AU)
        return {'Mean ELBO': np.mean(Average_loss), 'Mean LF': np.mean(Average_NLL),
                'Mean KL div': np.mean(Average_KL_Div), 'PPL': {torch.exp(torch.mean(torch.Tensor(NLL_mean_for_ppl)))},
                'Separability': sep, 'MI': {np.mean(MIs)}, 'Active Units': AU[0]}

    def create_graph(self):
        """First encode all train into the latent space"""
        encoded = self.encode()
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2)

        features = tsne.fit_transform(encoded['encoded_train'])
        x = features[:, 0]
        y = features[:, 1]
        labs = encoded['labels_train']
        # sentences=encoded['sentences_train']
        dico = {}
        for i in range(len(labs)):
            dico[i] = {'x': x[i], 'y': y[i], 'labs': labs[i].item(), 'points': encoded['encoded_train'][i].tolist()}

        df = pd.DataFrame.from_dict(dico, orient='index')
        print(df)
        df.to_csv(f'graph_{self.argdict["dataset"]}.tsv', sep='\t')
        sdffd

    def train_model(self):
        ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())


        print(self.model)

        # if self.argdict.tensorboard_logging:
        #     writer = SummaryWriter(os.path.join(self.argdict.logdir, expierment_name(self.argdict, ts)))
        #     writer.add_text("model", str(model))
        #     writer.add_text("self.argdict", str(self.argdict))
        #     writer.add_text("ts", ts)

        save_model_path = os.path.join(self.argdict['path'], 'bin')
        # shutil.
        os.makedirs(save_model_path, exist_ok=True)

        # with open(os.path.join(save_model_path, 'model_params.json'), 'w') as f:
        #     json.dump(self.params, f, indent=4)



        # tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        # step = 0
        # for epoch in range(self.argdict.epochs):
        for epoch in range(self.argdict['nb_epoch']):
            self.epoch=epoch
            self.run_epoch()
        self.interpolate()
        # self.generate_from_train()
        # self.create_graph()
        # fds


                # if self.argdict.tensorboard_logging:
                #     writer.add_scalar("%s-Epoch/ELBO" % split.upper(), torch.mean(tracker['ELBO']), epoch)

                # save a dump of all sentences and the encoded latent space
                # if split == 'valid':
                #     dump = {'target_sents': tracker['target_sents'], 'z': tracker['z'].tolist()}
                #     if not os.path.exists(os.path.join('dumps', ts)):
                #         os.makedirs('dumps/' + ts)
                #     with open(os.path.join('dumps/' + ts + '/valid_E%i.json' % epoch), 'w') as dump_file:
                #         json.dump(dump, dump_file)

                # save checkpoint
                # if split == 'train':
                #     checkpoint_path = os.path.join(save_model_path, "E%i.pytorch" % epoch)
                #     torch.save(self.model.state_dict(), checkpoint_path)
                #     print("Model saved at %s" % checkpoint_path)


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
        p0 = to_var(torch.randn([1, self.argdict['latent_size']]))
        p1 = to_var(torch.randn([1, self.argdict['latent_size']]))
        points = torch.zeros(n, self.argdict['latent_size'])
        points[0] = p0
        points[n - 1] = p1
        for i in range(n):
            ratio = i / n
            px = (1 - ratio) * p0 + ratio * p1
            points[i] = px
        points = points.cuda()
        samples, z = self.model.inference(n=n, z=points)
        self.datasets['train'].process_generated(samples)

    def encode(self):
        with torch.no_grad():
            dico = {}
            for split in self.splits:
                data_loader = DataLoader(
                    dataset=self.datasets[split],
                    batch_size=64,  # self.argdict.batch_size,
                    shuffle=False,
                    num_workers=cpu_count(),
                    pin_memory=torch.cuda.is_available()
                )
                # Enable/Disable Dropout

                self.model.eval()
                # print(f"The dataset length is {len(data_loader.dataset)}")
                dataset = torch.zeros(len(data_loader.dataset), self.argdict['latent_size'])
                labels = torch.zeros(len(data_loader.dataset))
                sentences = []
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
                dico[f"labels_{split}"] = labels
                dico[f"encoded_{split}"] = dataset
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
