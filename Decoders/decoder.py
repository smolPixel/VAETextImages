import torch.nn as nn


class decoder(nn.Module):

	def __init__(self, argdict, **kwargs):
		super().__init__()
		self.argdict=argdict

		encoder=self.argdict['decoder']
		if encoder.lower()=="gru":
			from Decoders.GRU import GRU_Decoder
			self.model=GRU_Decoder(argdict, kwargs['vocab_size'], kwargs['embedding_size'], hidden_size=kwargs['hidden_size'])


	def forward(self, input):
		return self.model(input)
