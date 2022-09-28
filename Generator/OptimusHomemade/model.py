import math
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForMaskedLM


class OptimusHomemade(nn.Module):
	"""VAE with normal prior"""

	def __init__(self, argdict):  #
		super(OptimusHomemade, self).__init__()
		self.argdict = argdict
		self.encoder=BertForMaskedLM.from_pretrained('bert-base-uncased')
		self.encoder_tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')


	def forward(self, batch):
		print(batch)
		sents=batch['sentence']
		sents=["[CLS] "+sent for sent in sents]
		encoded=self.encoder_tokenizer(sents, padding=True, truncation=True)
		print(encoded)


		fds

