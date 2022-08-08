import os
import io
import json
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
import pandas as pd
from nltk.tokenize import TweetTokenizer
from process_data import *
from torchtext.vocab import build_vocab_from_iterator
from torchvision import datasets, transforms
import copy

# #
# class dataset(Dataset):
#     def __init__(self, data, tokenizer, vocab, argdict):
#         super().__init__()
#         """data: tsv of the data
#            tokenizer: tokenizer trained
#            vocabInput+Output: vocab trained on train"""
#         self.data = {}
#         self.max_len = argdict['max_length']
#         find_max_len=False
#         if self.max_len==0:
#             find_max_len=True
#         self.vocab_object = vocab
#         self.tokenizer = tokenizer
#         self.pad_idx = self.vocab_object['<pad>']
#         self.sos_idx = self.vocab_object['<sos>']
#         self.eos_idx = self.vocab_object['<eos>']
#         self.unk_idx = self.vocab_object['<unk>']
#         index=0
#         mask=len(argdict['categories'])
#         for i, row in data.iterrows():
#             tokenized_text = self.tokenizer.tokenize("<bos> "+row['sentence']+" <eos>")
#             input =vocab(tokenized_text)
#             if find_max_len and len(input)>self.max_len:
#                 self.max_len=len(input)
#             self.data[index] = {'sentence':row['sentence'], 'input': input, 'label':row['label'], 'true_label':row['true_label']}
#             index+=1
#
#     # def tokenize_and_vectorize(self, sentences):
#     #     """Takes an array of sentences and return encoded data"""
#
#     def get_unlabelled(self):
#         dico={key:value for key, value in self.data.items() if value['label']==2}
#         return dico
#
#     def __call__(self, input_text, return_tensors, add_special_tokens):
#         fdadfsa
#
#     def reset_index(self):
#         new_dat = {}
#         for i, (j, dat) in enumerate(self.data.items()):
#             new_dat[i] = dat
#         self.data = new_dat
#
#     @property
#     def vocab_size(self):
#         return len(self.vocab_object)
#
#     @property
#     def vocab(self):
#         return self.vocab_object.get_stoi()
#         # fsd
#         # return self.vocab_object
#
#     def get_w2i(self):
#         return self.vocab_object.get_stoi()
#
#     def get_i2w(self):
#         return self.vocab_object.get_itos()
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, item):
#         input = self.data[item]['input'][:self.max_len]
#         length= len(input)
#         label = self.data[item]['label']
#         input.extend([self.pad_idx] * (self.max_len - len(input)))
#         target=input[1:]
#         input=input[:-1]
#         return {
#             'sentence': self.data[item]['sentence'],
#             'length': length,
#             'input': np.asarray(input, dtype=int),
#             'target': np.asarray(target, dtype=int),
#             'label': label,
#             'true_label' : self.data[item]['true_label']
#         }
#
#     def get_texts(self):
#         """returns a list of the textual sentences in the dataset"""
#         ll=[]
#         for _, dat in self.data.items():
#             ll.append(dat['sentence'])
#         return ll
#
#     def add_data(self, sentences, labels):
#         """Add data to the dataset"""
#         for sentence, label in zip(sentences, labels):
#             words=sentence.split(' ')
#             input = ['<bos>'] + words + ['<eos>']
#             # input = input[:self.max_sequence_length]
#
#             # target = words[:self.max_sequence_length - 1]
#             # target = target + ['<eos>']
#
#             # assert len(input) == len(target), "%i, %i" % (len(input), len(target))
#             # length = len(input)
#
#             # input.extend(['<pad>'] * (self.max_sequence_length - length))
#             # target.extend(['<pad>'] * (self.max_sequence_length - length))
#             input = self.vocab_object(input)
#             # target = [self.w2i.get(w, self.w2i['<unk>']) for w in target]
#
#             id = len(self.data)
#             self.data[id]={}
#
#             self.data[id]['sentence']=sentence
#             self.data[id]['input'] = input
#             self.data[id]['label'] = int(label)
#             self.data[id]['true_label'] = int(label)
#             # print(self.data[id])
#
#     def convert_tokens_to_string(self, tokens):
#         if tokens==[]:
#             return ""
#         else:
#             raise ValueError("Idk what this is supposed to return")
#
#     def arr_to_sentences(self, array):
#         sentences=[]
#         for arr in array:
#             arr=arr.int()
#             sent=self.vocab_object.lookup_tokens(arr.tolist())
#             ss=""
#             for token in sent:
#                 if token== "<bos>":
#                     continue
#                 if token =="<eos>":
#                     break
#                 ss+=f" {token}"
#             sentences.append(ss)
#         return sentences
#
#     def iterexamples(self):
#         for i, ex in self.data.items():
#             yield i, ex
#
#     def return_pandas(self):
#         """Return a pandas version of the dataset"""
#         dict={}
#         for i, ex in self.iterexamples():
#             dict[i]={'sentence':ex['sentence'], 'label':ex['label']}
#         return pd.DataFrame.from_dict(dict, orient='index')

def create_datasets(argdict):
    if argdict['dataset'] in ['SST2', "SST100"]:
        from data.SST2.SST2Dataset import SST2_dataset
        #Textual dataset
        tokenizer=TweetTokenizer()

        train, dev, test=get_dataFrame(argdict)
        vocab = build_vocab_from_iterator((iter([tokenizer.tokenize(sentence) for sentence in list(train['sentence'])])),specials=["<unk>", "<pad>", "<bos>", "<eos>"])
        vocab.set_default_index(vocab["<unk>"])
        train=SST2_dataset(train, tokenizer, vocab, argdict)
        dev=SST2_dataset(dev, tokenizer, vocab, argdict)
        test=SST2_dataset(test, tokenizer, vocab, argdict)
        argdict['input_size']=train.vocab_size
        return train, dev, test
    elif argdict['dataset'] in ['MNIST']:
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