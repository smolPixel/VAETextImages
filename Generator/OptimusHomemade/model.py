import math
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, GPT2Tokenizer
from Generator.OptimusHomemade.GPT2Latent import GPT2ModelLatent


class OptimusHomemade(nn.Module):
	"""VAE with normal prior"""

	def __init__(self, argdict):  #
		super(OptimusHomemade, self).__init__()
		self.argdict = argdict
		self.device = 'cuda'
		self.encoder=BertModel.from_pretrained('bert-base-uncased')
		self.encoder_tokenizer=BertTokenizer.from_pretrained('bert-base-uncased').to(self.device)

		self.hidden_to_latent=nn.Linear(768, argdict['latent_size'])

		# self.latent_to_hidden=nn.Linear

		self.decoder=GPT2ModelLatent.from_pretrained('gpt2', argdict)
		self.decoder_tokenizer=GPT2Tokenizer.from_pretrained('gpt2').to(self.device)
		self.decoder_tokenizer.pad_token = self.decoder_tokenizer.eos_token

		self.loss_function=torch.nn.CrossEntropyLoss(ignore_index=self.decoder_tokenizer.pad_token_id)

	def forward(self, batch):
		sents=batch['sentence']
		encoded=self.encoder_tokenizer(sents, padding=True, truncation=True, return_tensors='pt').to(self.device)
		output=self.encoder(input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask'])
		output=output['last_hidden_state'][:, 0, :]
		latent=self.hidden_to_latent(output)

		#TODO REPARAMETRIZATION

		#decoder
		encoded=self.decoder_tokenizer(sents, padding=True, truncation=True, return_tensors='pt').to(self.device)
		output = self.decoder(input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask'], z=latent, labels=encoded['input_ids'])
		return output