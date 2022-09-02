from torch.utils.data import Dataset
import pandas as pd
from nltk.tokenize import TweetTokenizer
from process_data import *
from torchtext.vocab import build_vocab_from_iterator
from torchvision import datasets, transforms
import copy
import torch
import matplotlib.pyplot as plt
#
class MNIST_dataset(Dataset):
	def __init__(self, data_ll):
		super().__init__()
		"""data: tsv of the data
		   tokenizer: tokenizer trained
		   vocabInput+Output: vocab trained on train"""
		self.data = {}
		self.loss_function=torch.nn.BCELoss(reduction='sum')
		index=0
		for i, row in enumerate(data_ll):
			input, label=row
			self.data[index] = {'input': input, 'label':label, 'target':input}
			index+=1

	def process_generated(self, exo):
		# print(exo)
		for i, img in enumerate(exo):
			plt.imsave(f'Temp/{i}.png', img.cpu().detach().view(28, 28))
		# fds

	def test_reconstruction(self, model):
		#First, get imgs of a 7 and a 0, which are fairly different
		# for key, item in self.data.items():
		# 	print(key, item['label'])

		dat=self.data[54993]
		dat1=self.data[54498]
		self.process_generated([dat['input'], dat1['input']])
		fds
		sentences = ["<bos> This is an excellent movie <eos>".lower(),
					 "<bos> I hated this movie so much I couldn't finish it <eos>".lower()]
		tokenized = self.batch_tokenize_and_pad(sentences)
		mean = model.encode(tokenized).squeeze(0)
		samples, z = model.inference(n=2, z=mean)
		self.process_generated(samples)


	def reset_index(self):
		new_dat = {}
		for i, (j, dat) in enumerate(self.data.items()):
			new_dat[i] = dat
		self.data = new_dat

	def __len__(self):
		return len(self.data)

	def __getitem__(self, item):
		input = self.data[item]['input']
		label = self.data[item]['label']
		return {
			'input': np.asarray(input, dtype=float),
			'target': np.asarray(input, dtype=float),
			'label': label,
		}

	def shape_for_loss_function(self, logp, target):
		return logp, target.view(target.shape[0], -1).float().cuda()

	def iterexamples(self):
		for i, ex in self.data.items():
			yield i, ex

	def return_pandas(self):
		"""Return a pandas version of the dataset"""
		dict={}
		for i, ex in self.iterexamples():
			dict[i]={'sentence':ex['sentence'], 'label':ex['label']}
		return pd.DataFrame.from_dict(dict, orient='index')