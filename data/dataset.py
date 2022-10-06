import os
import io
import json
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
import pandas as pd
from nltk.tokenize import TweetTokenizer
# from process_data import *
from torchtext.vocab import build_vocab_from_iterator
from torchvision import datasets, transforms
import copy
import math

def get_dataFrame(argdict, task, dataset_size):
    """Get the dataframe for the particular split. If it does not exist: create it"""
    create_train=False


    # try:
    #     dfTrain=pd.read_csv(f"{argdict['path']}/SelectedData/{argdict['dataset']}/{argdict['dataset_size']}/train_{argdict['random_seed']}.tsv", sep='\t', index_col=0)
    # except:
    #     create_train=True
    os.makedirs(f"{argdict['path']}/SelectedData/task/{argdict['dataset_size']}/",exist_ok=True)

    # if create_train:
    print(f"{argdict['path']}/data/{task}/train.tsv")
    dfTrain=pd.read_csv(f"{argdict['path']}/data/{task}/train.tsv", sep='\t')


        # print(len(dfTrain))
        # fds
    dfVal=pd.read_csv(f'{argdict["path"]}/data/{task}/dev.tsv', sep='\t')
    dfTest=pd.read_csv(f'{argdict["path"]}/data/{task}/test.tsv', sep='\t')
    #
    if argdict['dataset_size']!=0:
    #     #Sampling balanced data
    #     # print(len(dfTrain[dfTrain['label']==0]))
    #     # prop=len(dfTrain[dfTrain['label']==0])/len(dfTrain)
        dfTrain['true_label']=dfTrain['label']
        nb_points=math.ceil(argdict['dataset_size']/len(argdict['categories']))
        # print(prop)
        # print(int(argdict['dataset_size']/len(argdict['categories'])))
        # print(argdict['labelled_dataset_size'])
        NewdfTrain=dfTrain[dfTrain['label']==0].sample(n=nb_points)
        for i in range(1, len(argdict['categories'])):
            # print(int(argdict['dataset_size']/len(argdict['categories'])))
            # print(i)
            NewdfTrain=pd.concat([NewdfTrain ,dfTrain[dfTrain['label']==i].sample(n=nb_points)])
        dfTrain=NewdfTrain
        # dfTrain.drop(NewdfTrain.index)
        # dfTrain.to_csv(f"{argdict['path']}/SelectedData/{argdict['dataset']}/{argdict['dataset_size']}/train_{argdict['random_seed']}.tsv", sep='\t')
    return dfTrain, dfVal, dfTest

def create_datasets(argdict, dataset, size):
    if dataset in ['SST2', "SST100"]:
        from data.SST2.SST2Dataset import SST2_dataset
        #Textual dataset
        tokenizer=TweetTokenizer()
        train, dev, test=get_dataFrame(argdict, dataset, size)
        vocab = build_vocab_from_iterator((iter([tokenizer.tokenize(sentence) for sentence in list(train['sentence'])])),specials=["<unk>", "<pad>", "<bos>", "<eos>"])
        vocab.set_default_index(vocab["<unk>"])
        train=SST2_dataset(train, tokenizer, vocab, argdict)
        dev=SST2_dataset(dev, tokenizer, vocab, argdict)
        test=SST2_dataset(test, tokenizer, vocab, argdict)
        argdict['input_size']=train.vocab_size
        return train, dev, test
    elif dataset in ["Wikipedia"]:
        from data.Wikipedia.WikiDataset import Wiki_dataset
        #Textual dataset
        tokenizer=TweetTokenizer()
        train, dev, test=get_dataFrame(argdict, dataset, size)
        vocab = build_vocab_from_iterator((iter([tokenizer.tokenize(sentence) for sentence in list(train['sentence'])])),specials=["<unk>", "<pad>", "<bos>", "<eos>"])
        vocab.set_default_index(vocab["<unk>"])
        train=Wiki_dataset(train, tokenizer, vocab, argdict)
        dev=Wiki_dataset(dev, tokenizer, vocab, argdict)
        # test=SST2_dataset(test, tokenizer, vocab, argdict)
        argdict['input_size']=train.vocab_size
        return train, dev
    elif dataset in ['MNIST']:
        #Image dataset
        from data.MNIST.MNIST_dataset import MNIST_dataset
        train = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
        test = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(),
                                      download=False)
        train, dev=torch.utils.data.random_split(train, [55000, 5000])
        train=MNIST_dataset(train)
        dev=MNIST_dataset(dev)
        test=MNIST_dataset(test)
        argdict['input_size']=784
        return train, dev, test
    else:
        raise ValueError("dataset not found")

def create_datasets_from_dataframes(argdict, train):
    tokenizer=TweetTokenizer()
    _, dev = get_dataFrame(argdict)
    train=train.dropna()
    vocab = build_vocab_from_iterator((iter([tokenizer.tokenize(sentence) for sentence in list(train['sentence'])])),specials=["<unk>", "<pad>", "<bos>", "<eos>"])
    vocab.set_default_index(vocab["<unk>"])
    unlabelled={'train':dataset(train, tokenizer, vocab, argdict, False), 'dev':dataset(dev, tokenizer, vocab, argdict, False)}
    labelled={'train':dataset(train, tokenizer, vocab, argdict, True), 'dev':dataset(dev, tokenizer, vocab, argdict, True)}
    return unlabelled, labelled


class Transformerdataset(Dataset):
    """Dataset for fine tuning transformers. Constructed from a pandas dataframe. """

    def __init__(self, df, split, labelled_only, **kwargs):

        super().__init__()
        self.df=df.return_pandas()
        print(self.df)
        fds
        self.split = split
        self.max_sequence_length = kwargs.get('max_sequence_length', 50)
        self.min_occ = kwargs.get('min_occ', 1)
        # Whether to only return labelled examples (for fine tuning vs predicting)
        self.labelled_only = labelled_only

        self._create_data()
        # if self.split=='train':
        #     print(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = str(idx)
        # print(idx)
        # print(self.data[idx])
        return {
            'sentence': self.data[idx]['sentence'],
            'label' : self.data[idx]['label']
        }

    @property
    def vocab_size(self):
        return len(self.w2i)

    @property
    def pad_idx(self):
        return self.w2i['<pad>']

    @property
    def sos_idx(self):
        return self.w2i['<sos>']

    @property
    def eos_idx(self):
        return self.w2i['<eos>']

    @property
    def unk_idx(self):
        return self.w2i['<unk>']

    def get_w2i(self):
        return self.w2i

    def get_i2w(self):
        return self.i2w

    def _create_data(self):

        data = defaultdict(dict)
        #keep only labelled examples
        if self.labelled_only:
            self.df=self.df[self.df['label']!=2]


        print(self.df)
        for i, ex in self.df.iterrows():

            id = str(len(data))
            data[id]['sentence'] = ex['sentence']
            data[id]['label']=int(ex['label'])
        self.data=data


def fuse_datasets(d1, d2):
    """Fuse two datasets together, useful for training the labeller on train+dev. Both datasets need to have the same fields"""

    # print(d2.data)
    # print(len(d1))
    newd1=copy.deepcopy(d1)
    for key, value in d2.data.items():
        id = str(len(newd1.data))
        newd1.data[id] = value

    # print(len(d1))
    # fds
    # d1.reset_index()
    return newd1