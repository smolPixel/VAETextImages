import torch.nn as nn


class encoder(nn.Module):

	def __init__(self, argdict, **kwargs):
		super().__init__()
		self.argdict=argdict

		encoder=self.argdict['encoder']
		if encoder.lower()=="gru":
			from Encoders.GRU import GRU_Encoder
			self.model=GRU_Encoder(argdict, kwargs['vocab_size'], kwargs['embedding_size'], hidden_size=kwargs['hidden_size'], latent_size=kwargs['latent_size'])


	def forward(self, input):
		return self.model(input)
