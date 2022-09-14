"""Take in charge all generation algorithms"""
import os, shutil
from process_data import *
import subprocess
import torch
import copy
from sklearn.svm import SVC

def run_external_process(process):
    output, error = process.communicate()
    if process.returncode != 0:
        raise SystemError
    return output, error

class generator():

    def __init__(self, argdict, train, dev, test, classifier=None):
        self.argdict=argdict

        # self.argdict['c']


        algo = self.argdict['algo']
        if algo == "LM":
            from Generator.LM.LM import LM
            self.generator=LM(self.argdict, train, dev, test)
        elif algo == "GPT2":
            from Generator.GPT2.GPT2 import GPT2
            self.generator=GPT2(self.argdict, train, dev, test)
        elif algo == "AE":
            from Generator.AE.AE import AE
            self.generator=AE(self.argdict, train, dev, test)
        elif algo == "VAE":
            from Generator.VAE.VAE import VAE
            self.generator=VAE(self.argdict, train, dev, test)
        elif algo == "VAE_Annealing":
            from Generator.VAE_Annealing.VAE import VAE_Annealing
            self.generator = VAE_Annealing(self.argdict, train, dev, test)
        elif algo == "VAE_Annealing_Reg":
            from Generator.VAE_Annealing_Reg.VAE import VAE_Annealing_Reg
            self.generator = VAE_Annealing_Reg(self.argdict, train, dev, test)
        elif algo == "CVAE":
            from Generator.CVAE.CVAE import CVAE
            self.generator=CVAE(self.argdict, train, dev, test)
        elif algo == "CVAE_Annealing":
            from Generator.CVAE_Annealing.CVAE import CVAE
            self.generator=CVAE(self.argdict, train, dev, test)
        elif algo == "CVAE_Classic":
            from Generator.CVAE_Classic.CVAE_Classic import CVAE_Classic
            self.generator=CVAE_Classic(self.argdict, train, dev, test)
        elif algo == "CVAE_Classic_Annealing":
            from Generator.CVAE_Classic_Annealing.CVAE_Classic import CVAE_Classic
            self.generator=CVAE_Classic(self.argdict, train, dev, test)
        elif algo == "BVAE":
            from Generator.BVAE.BVAE import BVAE
            self.generator = BVAE(self.argdict, train, dev, test)
        elif algo == "WSVAE":
            from Generator.WSVAE.WSVAE import WSVAE
            self.generator = WSVAE(self.argdict, train, dev, test)
        else:
            raise ValueError(f"No generator named {algo}")



        self.classifier=classifier

    def train(self):
        self.generator.train()

        #Weight graph

    def test(self):
        return self.generator.test()

    def encode(self):
        return self.generator.encode()



    def generate(self, datapoints, cat):
        return self.generator.generate(datapoints, cat)

    def generate_from_label(self, label, number):
        """Generate from a given label and a number of sentences"""
        points_to_label = torch.randn(number, self.argdict['latent_size']).cuda()
        # points_to_label = torch.zeros(self.argdict['batch_size_MQS'],
        #                               self.argdict['latent_size'] + len(self.argdict['categories'])).cuda()
        #In some case you can't control the generation
        # if self.argdict['algo'] in ['SVAE', 'CVAE']:
        #     labels=torch.zeros(number, len(self.argdict['categories'])).cuda()
        #     labels[:, label]=1
        #     points_to_label=torch.cat([points_to_label, labels], dim=1)
        return self.generator.generate(points_to_label, label)

    def test_separability(self):
        separator=SVC()
        encoded=self.encode()
        X=encoded['encoded_train']
        Y=encoded['true_labels_train']
        separator.fit(X, Y)
        print(separator.score(X, Y))

        print(encoded.keys())


    def encode_decode(self):
        return self.generator.encode_decode()

    def run_epoch(self, datasets, datasetsLabelled):
        self.generator.datasets=datasets
        self.generator.datasetsLabelled=datasetsLabelled
        self.generator.run_epoch()
        # self.generator.run_epoch_classifier()
        # self.generator.run_epoch_VAE()