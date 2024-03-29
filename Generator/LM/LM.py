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
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict
from sklearn.metrics import accuracy_score

# from Generators.VAE.ptb import PTB
from Generator.utils import to_var, idx2word, expierment_name
from Generator.LM.model import LM_model
from Decoders.decoder import decoder



class LM():

    def __init__(self, argdict, train, dev, test):
        self.argdict=argdict
        self.splits=['train', 'dev']
        self.datasets={'train':train, 'dev':dev, 'test':test}
        self.model, self.params=self.init_model_dataset()
        # optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)  # self.argdict.learning_rate)
        self.loss_function_basic=train.loss_function

    def init_model_dataset(self):
        deco=decoder(self.argdict)

        params = dict(
            argdict=self.argdict,
            decoder=deco
        )
        model = LM_model(**params)
        if torch.cuda.is_available():
            model = model.cuda()

        return model, params


    def loss_fn(self, logp, target):
        NLL_loss = self.loss_function_basic(logp, target)
        return NLL_loss

    def run_epoch(self):
        for split in self.splits:

            data_loader = DataLoader(
                dataset=self.datasets[split],
                batch_size=64,  # self.argdict.batch_size,
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
            for iteration, batch in enumerate(data_loader):

                # Forward pass
                self.model.to('cuda')
                logp = self.model(batch)
                batch_size = logp.shape[0]
                logp, target=self.datasets['train'].shape_for_loss_function(logp, batch['target'])
                NLL_loss= self.loss_fn(logp, target.to('cuda'))
                loss = NLL_loss / batch_size

                # backward + optimization
                if split == 'train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                Average_loss.append(loss.item())
                Average_NLL.append(NLL_loss.cpu().detach()/batch_size)

            print(f"{split.upper()} Epoch {self.epoch}/{self.argdict['nb_epoch']}, Mean LF {np.mean(Average_NLL)}")


    def test(self):
        print("Testing system")
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
        Average_ppl=[]
        critppl=torch.nn.CrossEntropyLoss()
        for iteration, batch in enumerate(data_loader):

            # Forward pass
            self.model.to('cuda')
            logp = self.model(batch)
            batch_size = logp.shape[0]
            logp, target=self.datasets['train'].shape_for_loss_function(logp, batch['target'])
            NLL_loss= self.loss_fn(logp, target.to('cuda'))
            loss = NLL_loss / batch_size
            Average_ppl.append(critppl(logp, target.to('cuda')).item())
            Average_loss.append(loss.item())
            Average_NLL.append(NLL_loss.cpu().detach()/batch_size)


        print((torch.mean(torch.exp(Average_loss))))
        print(Average_NLL)
        print(f"Mean LF on Test {np.mean(Average_NLL)}")

    def create_graph(self):
        """First encode all train into the latent space"""
        encoded=self.encode()
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2)

        features = tsne.fit_transform(encoded['encoded_train'])
        x=features[:, 0]
        y=features[:, 1]
        labs=encoded['labels_train']
        # sentences=encoded['sentences_train']
        dico={}
        for i in range(len(labs)):
            dico[i]={'x':x[i], 'y':y[i], 'labs':labs[i].item(), 'points':encoded['encoded_train'][i].tolist()}


        df = pd.DataFrame.from_dict(dico, orient='index')
        print(df)
        df.to_csv(f'graph_{self.argdict["dataset"]}.tsv', sep='\t')
        sdffd

    def train(self):
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

    def encode_decode(self):
        #Encodes and decodes all the training dataset
        data_loader = DataLoader(
            dataset=self.datasets['dev'],
            batch_size=2,  # self.argdict.batch_size,
            shuffle=False,
            num_workers=cpu_count(),
            pin_memory=torch.cuda.is_available()
        )

        self.model.eval()
        sentences=[]
        ground_truth=[]

        for iteration, batch in enumerate(data_loader):

            batch_size = batch['input'].size(0)
            # print(batch)
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = to_var(v)

            # Forward pass
            logp, mean, logv, z = self.model(batch)
            samples, z = self.model.inference(z=z.squeeze(0))
            gend=idx2word(samples, i2w=self.datasets['train'].get_i2w(), pad_idx=self.datasets['train'].get_w2i()['<pad>'],
                     eos_idx=self.datasets['train'].get_w2i()['<eos>'])
            sentences.extend(gend)
            ground_truth.extend(batch['sentence'])
        return ground_truth, sentences

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
#
# class VAE():
#
#     def __init__(self, argdict, datasets, datasetLabelled):
#         self.argdict=argdict
#         self.splits=['train', 'dev']
#         self.datasets = datasets
#         self.datasetsLabelled = datasetLabelled
#         self.model, self.params=self.init_model_dataset()
#         # optimizers
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)  # self.argdict.learning_rate)
#         self.loss_function_discriminator = torch.nn.CrossEntropyLoss()
#         self.step=0
#         self.epoch = 0
#
#     def init_model_dataset(self):
#         splits = ['train', 'dev']  # + (['test'] if self.argdict.test else [])
#
#         # datasets = OrderedDict()
#         # for split in splits:
#         #     datasets[split] = PTB(
#         #         data_dir=self.argdict['pathFolder'] + '/Generators/VAE/data',
#         #         split=split,
#         #         create_data=False,  # self.argdict.create_data,
#         #         max_sequence_length=60,  # self.argdict.max_sequence_length,
#         #         min_occ=0  # self.argdict.min_occ
#         #     )
#
#         # print("BLIBLBILBi")
#         # print(datasetsLabelled['train'])
#
#         params = dict(
#             vocab_size=self.datasets['train'].vocab_size,
#             sos_idx=self.datasets['train'].sos_idx,
#             eos_idx=self.datasets['train'].eos_idx,
#             pad_idx=self.datasets['train'].pad_idx,
#             unk_idx=self.datasets['train'].unk_idx,
#             max_sequence_length=60,  # self.argdict.max_sequence_length,
#             embedding_size=300,  # self.argdict.embedding_size,
#             rnn_type='gru',  # self.argdict.rnn_type,
#             hidden_size=self.argdict['hidden_size_algo'],
#             word_dropout=self.argdict['word_dropout'],  # self.argdict.word_dropout,
#             embedding_dropout=self.argdict['dropout_algo'],  # self.argdict.embedding_dropout,
#             latent_size=self.argdict['latent_size'],
#             num_layers=2,  # self.argdict['num_layers_algo'],
#             bidirectional=False  # self.argdict.bidirectional
#         )
#         model = SentenceVAE(**params)
#         if torch.cuda.is_available():
#             model = model.cuda()
#
#         self.step=0
#         self.epoch=0
#
#         return model, params
#
#     def kl_anneal_function(self, anneal_function, step, k, x0):
#         if anneal_function == 'logistic':
#             return float(1 / (1 + np.exp(-k * (step - x0))))
#         elif anneal_function == 'linear':
#             return min(1, step / x0)
#
#     def loss_fn(self, logp, target, length, mean, logv, anneal_function, step, k, x0):
#         NLL = torch.nn.NLLLoss(ignore_index=self.datasets['train'].pad_idx, reduction='sum')
#         # cut-off unnecessary padding from target, and flatten
#         target = target.contiguous().view(-1)
#         logp = logp.view(-1, logp.size(2))
#
#         # Negative Log Likelihood
#         NLL_loss = NLL(logp, target)
#
#         # KL Divergence
#         KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
#         KL_weight = self.kl_anneal_function(anneal_function, step, k, x0)
#
#         return NLL_loss, KL_loss, KL_weight
#
#     def run_epoch(self):
#         for split in self.splits:
#
#             data_loader = DataLoader(
#                 dataset=self.datasets[split],
#                 batch_size=self.argdict['batch_size_generator'],  # self.argdict.batch_size,
#                 shuffle=split == 'train',
#                 num_workers=cpu_count(),
#                 pin_memory=False#torch.cuda.is_available()
#             )
#
#             # tracker = defaultdict(tensor)
#
#             # Enable/Disable Dropout
#             if split == 'train':
#                 self.model.train()
#             else:
#                 self.model.eval()
#
#             for iteration, batch in enumerate(data_loader):
#
#                 batch_size = batch['input'].size(0)
#
#                 for k, v in batch.items():
#                     if torch.is_tensor(v):
#                         batch[k] = to_var(v)
#                 # print(len(data_loader))
#                 # Forward pass
#                 logp, mean, logv, z = self.model(batch['input'], batch['length'])
#
#                 # loss calculation
#                 # NLL_loss, KL_loss, KL_weight = loss_fn(logp, batch['target'],
#                 #                                        batch['length'], mean, logv, self.argdict.anneal_function, step,
#                 #                                        self.argdict.k, self.argdict.x0)
#                 # print(self.argdict['x0'])
#                 # print(len(self.datasets['train']))
#                 # print(self.argdict['x0']*len(self.datasets['train']))
#                 NLL_loss, KL_loss, KL_weight = self.loss_fn(logp, batch['target'],
#                                                             batch['length'], mean, logv, 'logistic', self.step, 0.00025, self.argdict['x0'])
#                                                             # self.argdict['k'], self.argdict['x0'])#*len(self.datasets['train'])/self.argdict['batch_size_generator'])
#
#                 loss = (NLL_loss + KL_weight * KL_loss) / batch_size
#
#                 # backward + optimization
#                 if split == 'train':
#                     self.optimizer.zero_grad()
#                     loss.backward()
#                     self.optimizer.step()
#                     self.step += 1
#
#                 if iteration % 50 == 0 or iteration + 1 == len(data_loader):
#                     print("%s Batch %04d/%i, Loss %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f, KL-Weight %6.3f"
#                           % (
#                               split.upper(), iteration, len(data_loader) - 1, loss.item(), NLL_loss.item() / batch_size,
#                               KL_loss.item() / batch_size, KL_weight))
#
#                 # if split == 'valid':
#                 #     if 'target_sents' not in tracker:
#                 #         tracker['target_sents'] = list()
#                 #     tracker['target_sents'] += idx2word(batch['target'].data, i2w=self.datasets['train'].get_i2w(),
#                 #                                         pad_idx=self.datasets['train'].pad_idx)
#                 #     tracker['z'] = torch.cat((tracker['z'], z.data), dim=0)
#
#             # print("%s Epoch %02d/%i, Mean ELBO %9.4f" % (split.upper(), epoch, self.argdict['nb_epoch_algo'], tracker['ELBO'].mean()))
#             print("%s Epoch %02d/%i, Mean ELBO %9.4f" % (split.upper(), self.epoch, self.argdict['nb_epoch_algo'], 0))
#
#     def train(self):
#         ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())
#
#
#         print(self.model)
#
#         # if self.argdict.tensorboard_logging:
#         #     writer = SummaryWriter(os.path.join(self.argdict.logdir, expierment_name(self.argdict, ts)))
#         #     writer.add_text("model", str(model))
#         #     writer.add_text("self.argdict", str(self.argdict))
#         #     writer.add_text("ts", ts)
#
#         save_model_path = os.path.join(self.argdict['pathFolder'], 'bin')
#         # shutil.
#         os.makedirs(save_model_path, exist_ok=True)
#
#         # with open(os.path.join(save_model_path, 'model_params.json'), 'w') as f:
#         #     json.dump(self.params, f, indent=4)
#
#
#
#         # tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
#         # step = 0
#         # for epoch in range(self.argdict.epochs):
#         for epoch in range(self.argdict['nb_epoch_algo']):
#             self.epoch=epoch
#             self.run_epoch()
#
#
#                 # if self.argdict.tensorboard_logging:
#                 #     writer.add_scalar("%s-Epoch/ELBO" % split.upper(), torch.mean(tracker['ELBO']), epoch)
#
#                 # save a dump of all sentences and the encoded latent space
#                 # if split == 'valid':
#                 #     dump = {'target_sents': tracker['target_sents'], 'z': tracker['z'].tolist()}
#                 #     if not os.path.exists(os.path.join('dumps', ts)):
#                 #         os.makedirs('dumps/' + ts)
#                 #     with open(os.path.join('dumps/' + ts + '/valid_E%i.json' % epoch), 'w') as dump_file:
#                 #         json.dump(dump, dump_file)
#
#                 # save checkpoint
#                 # if split == 'train':
#                 #     checkpoint_path = os.path.join(save_model_path, "E%i.pytorch" % epoch)
#                 #     torch.save(self.model.state_dict(), checkpoint_path)
#                 #     print("Model saved at %s" % checkpoint_path)
#
#     def encode(self):
#         dico={}
#         for split in self.splits:
#             data_loader = DataLoader(
#                 dataset=self.datasets[split],
#                 batch_size=self.argdict['batch_size_generator'],#self.argdict.batch_size,
#                 shuffle=False,
#                 num_workers=cpu_count(),
#                 pin_memory=torch.cuda.is_available()
#             )
#             # Enable/Disable Dropout
#
#             self.model.eval()
#             # print(f"The dataset length is {len(data_loader.dataset)}")
#             dataset = torch.zeros(len(data_loader.dataset), self.params['latent_size'])
#             labels = torch.zeros(len(data_loader.dataset))
#             counter = 0
#             for iteration, batch in enumerate(data_loader):
#                 # print("Oh la la banana")
#                 batch_size = batch['input'].size(0)
#                 # print(batch['input'].shape)
#                 for k, v in batch.items():
#                     if torch.is_tensor(v):
#                         batch[k] = to_var(v)
#                 #
#                 # print(batch['input'])
#                 # print(batch['input'].shape)
#                 z = self.model.encode(batch['input'], batch['length'])
#                 # print(batch_size)
#                 # print(z.shape)
#                 dataset[counter:counter + batch_size] = z
#                 labels[counter:counter + batch_size] = batch['label']
#                 counter += batch_size
#             # print(dataset)
#             dico[f"labels_{split}"]=labels
#             dico[f"encoded_{split}"]=dataset
#             # torch.save(labels, f"bin/labels_{split}.pt")
#             # torch.save(dataset, f"bin/encoded_{split}.pt")
#         return dico
#
#     def generate(self, datapoints, label):
#         #Generates from fixed datapoints
#         self.model.eval()
#
#         samples, z = self.model.inference(z=datapoints)
#         # print(samples)
#         # print('----------SAMPLES----------')
#         return idx2word(samples, i2w=self.datasets['train'].get_i2w(), pad_idx=self.datasets['train'].pad_idx)