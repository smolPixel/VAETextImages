import os
import io
import json
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
import pandas as pd
import torch
from nltk.tokenize import TweetTokenizer
from process_data import *
from torchtext.vocab import build_vocab_from_iterator
from torchvision import datasets, transforms
import copy
from Generator.utils import to_var, idx2word
#
class SST2_dataset(Dataset):
	def __init__(self, data, tokenizer, vocab, argdict):
		super().__init__()
		"""data: tsv of the data
		   tokenizer: tokenizer trained
		   vocabInput+Output: vocab trained on train"""
		self.data = {}
		self.max_len = argdict['max_length']
		find_max_len=False
		if self.max_len==0:
			find_max_len=True
		self.vocab_object = vocab
		self.tokenizer = tokenizer
		self.pad_idx = self.vocab_object['<pad>']
		self.sos_idx = self.vocab_object['<bos>']
		self.eos_idx = self.vocab_object['<eos>']
		self.unk_idx = self.vocab_object['<unk>']
		argdict['pad_idx']=self.pad_idx
		argdict['sos_idx']=self.sos_idx
		argdict['unk_idx']=self.unk_idx
		# self.loss_function=torch.nn.CrossEntropyLoss(ignore_index=self.pad_idx, reduction='mean')
		self.loss_function=torch.nn.CrossEntropyLoss(ignore_index=self.pad_idx, reduction='sum')
		index=0
		mask=len(argdict['categories'])
		for i, row in data.iterrows():
			tokenized_text = self.tokenizer.tokenize("<bos> "+row['sentence']+" <eos>")
			input =vocab(tokenized_text)
			if find_max_len and len(input)>self.max_len:
				self.max_len=len(input)
			self.data[index] = {'sentence':row['sentence'], 'input': input, 'label':row['label']}
			index+=1

	# def tokenize_and_vectorize(self, sentences):
	#     """Takes an array of sentences and return encoded data"""

	def batch_tokenize_and_pad(self, sentences):
		bs=len(sentences)
		array=torch.zeros((bs, self.max_len))
		for i, sent in enumerate(sentences):
			tokenized_text = self.tokenizer.tokenize(sent)
			input = self.vocab_object(tokenized_text)
			input = input[:self.max_len]
			input.extend([self.pad_idx] * (self.max_len - len(input)))
			input=torch.Tensor(input)
			array[i]=input
		return array.long()

	def test_reconstruction(self, model, tokenizer=None):
		sentences = ["<bos> This is an excellent movie <eos>".lower(),
					 "<bos> I hated this movie so much I couldn't finish it <eos>".lower()]
		if tokenizer is None:
			tokenized = self.batch_tokenize_and_pad(sentences)
			z = model.encode({'input': tokenized}).squeeze(0)
			samples, z = model.inference(z=z, bos_token=tokenizer.bos_token_id)
			self.process_generated(samples)

		else:
			tokenized=tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
			z, _, _ = model.encode(input_ids=tokenized['input_ids'], attention_mask=tokenized['attention_mask'])
			samples, z = model.inference(z=z, bos_token=tokenizer.bos_token_id)
			print(tokenizer.batch_decode(samples, skip_special_tokens=True))


	def get_unlabelled(self):
		dico={key:value for key, value in self.data.items() if value['label']==2}
		return dico

	def __call__(self, input_text, return_tensors, add_special_tokens):
		fdadfsa

	def reset_index(self):
		new_dat = {}
		for i, (j, dat) in enumerate(self.data.items()):
			new_dat[i] = dat
		self.data = new_dat

	@property
	def vocab_size(self):
		return len(self.vocab_object)

	@property
	def vocab(self):
		return self.vocab_object.get_stoi()
		# fsd
		# return self.vocab_object

	def get_w2i(self):
		return self.vocab_object.get_stoi()

	def process_generated(self, exo):
		generated = idx2word(exo, i2w=self.get_i2w(),
							 pad_idx=self.get_w2i()['<pad>'],
							 eos_idx=self.get_w2i()['<eos>'])
		for sent in generated:
			print("------------------")
			print(sent)


	def decode(self, exo):
		generated = idx2word(exo, i2w=self.get_i2w(),
							 pad_idx=self.get_w2i()['<pad>'],
							 eos_idx=self.get_w2i()['<eos>'])
		return generated

	def get_i2w(self):
		return self.vocab_object.get_itos()

	def __len__(self):
		return len(self.data)

	def __getitem__(self, item):
		input = self.data[item]['input'][:self.max_len]
		length= len(input)
		label = self.data[item]['label']
		input.extend([self.pad_idx] * (self.max_len - len(input)))
		target=input[1:]
		input=input[:-1]
		return {
			'sentence': self.data[item]['sentence'],
			'length': length,
			'input': np.asarray(input, dtype=int),
			'target': np.asarray(target, dtype=int),
			'label': label,
		}

	def get_texts(self):
		"""returns a list of the textual sentences in the dataset"""
		ll=[]
		for _, dat in self.data.items():
			ll.append(dat['sentence'])
		return ll

	def shape_for_loss_function(self, logp, target):
		target = target.contiguous().view(-1).cuda()
		logp = logp.view(-1, logp.size(2)).cuda()
		return logp, target

	def convert_tokens_to_string(self, tokens):
		if tokens==[]:
			return ""
		else:
			raise ValueError("Idk what this is supposed to return")

	def arr_to_sentences(self, array):
		sentences=[]
		for arr in array:
			arr=arr.int()
			sent=self.vocab_object.lookup_tokens(arr.tolist())
			ss=""
			for token in sent:
				if token== "<bos>":
					continue
				if token =="<eos>":
					break
				ss+=f" {token}"
			sentences.append(ss)
		return sentences

	def iterexamples(self):
		for i, ex in self.data.items():
			yield i, ex

	def return_pandas(self):
		"""Return a pandas version of the dataset"""
		dict={}
		for i, ex in self.iterexamples():
			dict[i]={'sentence':ex['sentence'], 'label':ex['label']}
		return pd.DataFrame.from_dict(dict, orient='index')