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

# from Generators.VAE.ptb import PTB
from Generator.utils import to_var, idx2word, expierment_name
from Generator.WSVAE.model import WSVAE_model
from Encoders.encoder import encoder
from Decoders.decoder import decoder
from Discriminators.discriminator import discriminator


class WSVAE():

    def __init__(self, argdict, train, dev, test):
        self.argdict=argdict
        self.splits=['train', 'dev']
        self.datasets={'train':train, 'dev':dev, 'test':test}
        self.model, self.params=self.init_model_dataset()
        # optimizers
        self.optimizer=torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.optimizer_encoder = torch.optim.Adam(self.model.encoder.parameters(), lr=0.001)  # self.argdict.learning_rate)
        self.optimizer_decoder = torch.optim.Adam(self.model.decoder.parameters(), lr=0.001)  # self.argdict.learning_rate)
        self.optimizer_discriminator = torch.optim.Adam(self.model.discriminator.parameters(), lr=0.001)  # self.argdict.learning_rate)
        self.loss_function_basic=train.loss_function
        self.loss_function_discriminator=torch.nn.BCEWithLogitsLoss()
        self.loss_function_encoder=torch.nn.L1Loss()

    def init_model_dataset(self):
        self.step = 0
        self.epoch = 0

        enco=encoder(self.argdict)#vocab_size=self.datasets['train'].vocab_size, embedding_size=300, hidden_size=self.argdict['hidden_size'], latent_size=self.argdict['latent_size'])
        deco=decoder(self.argdict)
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
                logp, mean, logv, z = self.model(batch)
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
                    loss=self.loss_function_discriminator(output, batch['label'].cuda().float())
                    preds.extend(torch.round(torch.sigmoid(output).cpu().detach()))
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

    def train_gen_enc(self):
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
                z_normal, c = z[:, :, :-1], z[:, :, -1]

                #Getting discriminator loss
                softmaxed_gumbeled = F.gumbel_softmax(logp, tau=1, hard=True, dim=-1)
                output_discriminator = self.model.discriminate(softmaxed_gumbeled)
                loss_discriminator = self.loss_function_discriminator(output_discriminator, batch['label'].cuda().float())
                preds.extend(torch.round(torch.sigmoid(output_discriminator).cpu().detach()))
                ground_truth.extend(batch['label'])

                #Getting reconstruction loss
                #Why not optimize on the difference of mu and logv directly
                # print("WARNING THIS SHOULD BE DONE AFTER GETTING ENCODER LOSS")
                encoded_generated=self.model.encode(softmaxed_gumbeled)
                z_normal_encoded= encoded_generated[:, :, :-1]
                loss_encoder=self.loss_function_encoder(z_normal_encoded, z_normal)


                #Regular loss
                logp, target=self.datasets['train'].shape_for_loss_function(logp, batch['target'])
                NLL_loss, KL_loss= self.loss_fn(logp, target.to('cuda'),  mean, logv)

                # loss = (NLL_loss +  KL_loss) / batch_size

                #Minimize Eq 8= Eq 4 (standard VAE) + Eq 6 (reconstruction of c) + Eq 7 (reconstruction of z)
                #Equation
                loss_encoder=(NLL_loss +  KL_loss) / batch_size
                # backward + optimization
                # if split == 'train':
                #     # self.optimizer.zero_grad()
                #     self.optimizer_encoder.zero_grad()
                #     loss_encoder.backward(retain_graph=True)
                #     # self.optimizer.step()
                #     self.optimizer_encoder.step()
                #     self.step += 1

                loss_generator=(NLL_loss +  KL_loss) / batch_size
                loss_generator+=loss_discriminator/batch_size
                loss_generator+=loss_encoder/batch_size

                # backward + optimization
                if split == 'train':
                    # self.optimizer.zero_grad()
                    self.optimizer_encoder.zero_grad()
                    self.optimizer_decoder.zero_grad()
                    loss_generator.backward()
                    # self.optimizer.step()
                    self.optimizer_encoder.step()
                    self.optimizer_decoder.step()
                    self.step += 1

                Average_loss.append(loss_generator.item())
                Average_KL_Div.append(KL_loss.cpu().detach()/batch_size)
                Average_NLL.append(NLL_loss.cpu().detach()/batch_size)

            print(f"{split.upper()} Epoch {self.epoch}/{self.argdict['nb_epoch']}, Mean ELBO {np.mean(Average_loss)}, Mean LF {np.mean(Average_NLL)}, Mean KL div {np.mean(Average_KL_Div)}"
                  f"Acc recon {accuracy_score(ground_truth, preds)}")


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
        print(self.model)
        save_model_path = os.path.join(self.argdict['path'], 'bin')
        # shutil.
        os.makedirs(save_model_path, exist_ok=True)

        #Pretraining the VAE
        for epoch in range(self.argdict['nb_epoch']):
            self.epoch=epoch
            self.run_epoch(pretraining=True)

        #Until convergence
        print("Change for until convergence")
        for i in range(10):
            #Train the discriminator by Eq 11 - Only labelled part for now
            self.train_discriminator()
            #Train the generator with equation 8, which is sum of the VAE loss, the attribute c loss which is the expectation over p(z)p(c) that the discriminator can recover the correct c,
            #abd the z loss where we check whether the encoder can recover the correct z code
            self.train_gen_enc()

        self.interpolate()
        # self.generate_from_train()
        self.create_graph()

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