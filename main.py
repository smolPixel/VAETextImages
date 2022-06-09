"""Test for several things concerning """


import argparse
import subprocess, os
from process_data import *
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

#In the VAE paradigm, we do not use the label except for selecting the next point
#In the CVAE paradigm, we use the label




def main(argdict):
    # run_lstm(argdict)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE for data augmentation')
    #General arguments on training
    parser.add_argument('--dataset', type=str, default='SST-2', help="dataset you want to run the process on. Includes SST2, TREC6, FakeNews")

    args = parser.parse_args()
    argsdict = args.__dict__
    main(argsdict)