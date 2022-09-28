import math
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel


class OptimusHomemade(nn.Module):
	"""VAE with normal prior"""

	def __init__(self, argdict):  #
		super(OptimusHomemade, self).__init__()
		self.argdict = argdict
		self.encoder=BertModel.from_pretrained('bert-base-uncased')
		self.encoder_tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')


	def forward(self, batch):
		print(batch)
		sents=batch['sentence']
		encoded=self.encoder_tokenizer(sents, padding=True, truncation=True)
		output=self.encoder(input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask'])
		print(output)

		fds

