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

		self.hidden_to_latent=nn.Linear(768, argdict['latent_size'])

	def forward(self, batch):
		sents=batch['sentence']
		encoded=self.encoder_tokenizer(sents, padding=True, truncation=True, return_tensors='pt')
		output=self.encoder(input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask'])
		print(encoded['input_ids'].shape)
		output=output['last_hidden_state'][:, 0, :]
		latent=self.hidden_to_latent(output)


		print(latent.shape)
		fds