import torch.nn as nn
import torch

class decoder(nn.Module):

	def __init__(self, argdict, **kwargs):
		super().__init__()
		self.argdict=argdict

		self.norm=[]


		decoder=self.argdict['decoder']
		if decoder.lower()=="gru":
			from Decoders.GRU import GRU_Decoder
			self.model=GRU_Decoder(argdict) #, kwargs['vocab_size'], kwargs['embedding_size'], hidden_size=kwargs['hidden_size'], latent_size=kwargs['latent_size'])
		elif decoder.lower()=='linear':
			from Decoders.Linear import Linear_Decoder
			self.model=Linear_Decoder(argdict)
		else:
			raise ValueError("unrecognized Encoder")

	def forward(self, input, z):
		norm=torch.clone(torch.linalg.norm(self.model.latent2hidden.weight))
		print(norm)
		fd
		return self.model(input, z)

	def generate(self, z, **kwargs):
		return self.model.generate(z, **kwargs)
