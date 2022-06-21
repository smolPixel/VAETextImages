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
    Gen.test_separability()

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='VAE for data augmentation')
    # #General arguments on training
    # parser.add_argument('--dataset', type=str, default='SST-2', help="dataset you want to run the process on. Includes SST2, TREC6, FakeNews")
    # parser.add_argument('--computer', type=str, default='labo', help="Whether you run at home or at iro. Automatically changes the base path")
    # parser.add_argument('--dataset_size', type=int, default=0, help='number of example in the original dataset. If 0, use the entire dataset')
    # parser.add_argument('--random_seed', type=int, default=7, help='Random seed ')
    # parser.add_argument('--max_length', type=int, default=0, help='max length of the data, 0 for no max length')
    # parser.add_argument('--algo', type=str, default='VAE', help='data augmentation algorithm to use, includes, VAE, CVAE, CVAE_Classic, SSVAE')
    # # Algo parameters
    # parser.add_argument('--nb_epoch_algo', type=int, default=30, help='Number of epoch of the algo')
    # parser.add_argument('--accum_batch_size', type=int, default=64, help='Accum')
    # parser.add_argument('--latent_size', type=int, default=5, help='Latent Size')
    # parser.add_argument('--hidden_size_algo', type=int, default=1024, help='Hidden Size Algo')
    # parser.add_argument('--num_layers_algo', type=int, default=2, help='Hidden Size Algo')
    # parser.add_argument('--batch_size_algo', type=int, default=64, help='batch size of the algo')
    # parser.add_argument('--dropout_algo', type=float, default=0.3, help='dropout of the classifier')
    # parser.add_argument('--word_dropout', type=float, default=0.3, help='dropout of the classifier')
    # parser.add_argument('--x0', default=50, type=int, help='x0')
    # args = parser.parse_args()
    # argsdict = args.__dict__

    stream=open("Configs/SST-2/VAE.yaml", "r")
    argsdict=yaml.safe_load(stream)
    # print(dico)
    # fds


    if argsdict['computer'] == 'home':
        argsdict['path'] = "/media/frederic/VAETI"
    elif argsdict['computer'] == 'labo':
        argsdict['path'] = "/u/piedboef/Documents/VAETI"

    if argsdict['dataset'] == "SST-2":
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
    else:
        raise ValueError("Dataset not found")
    argsdict['categories'] = categories

    main(argsdict)