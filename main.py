"""Test for several things concerning """


import argparse
import subprocess, os
from process_data import *
import random
import numpy as np
import torch
from data.dataset import create_datasets
from Generator.Generator import generator
import yaml

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

#In the VAE paradigm, we do not use the label except for selecting the next point
#In the CVAE paradigm, we use the label




def main(argdict):
    # run_lstm(argdict)
    train, dev, test=create_datasets(argdict)
    Gen = generator(argdict, train, dev, test)
    Gen.train()
    Gen.test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiments for VAE')
    # #General arguments on training

    parser.add_argument('--config_file', type=str, default='Configs/SST-2/VAE.yaml', help="dataset you want to run the process on. Includes SST2, TREC6, FakeNews")
    args = parser.parse_args()
    args = args.__dict__


    print(args)
    stream=open(args['config_file'], "r")
    argsdict=yaml.safe_load(stream)
    # print(dico)
    # fds


    if argsdict['computer'] == 'home':
        argsdict['path'] = "/media/frederic/VAETI"
    elif argsdict['computer'] == 'labo':
        argsdict['path'] = "/u/piedboef/Documents/VAETI"

    if argsdict['dataset'] == "SST2":
        categories = ["neg", "pos"]
    elif argsdict['dataset'] == "TREC6":
        categories = ["ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"]
    elif argsdict['dataset'] == "FakeNews":
        categories = ["Fake", "Real"]
    elif argsdict['dataset'] == "QNLI":
        categories = ["entailment", "not_entailment"]
    elif argsdict['dataset'] == "Irony":
        categories = ["NotIro", "Iro"]
    elif argsdict['dataset'] == "IronyB":
        categories = ["Clash", "Situational", "Other", "NotIro"]
    elif argsdict['dataset'] == 'MNIST':
        categories = [0,1,2,3,4,5,6,7,8,9]
    else:
        raise ValueError("Dataset not found")
    argsdict['categories'] = categories

    main(argsdict)