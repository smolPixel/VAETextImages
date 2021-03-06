import torch.nn as nn
import torch

class encoder(nn.Module):

	def __init__(self, argdict):
		super().__init__()
		self.argdict=argdict
		# self.datasets=datasets

		encoder=self.argdict['encoder']
		if encoder.lower()=="gru":
			from Encoders.GRU import GRU_Encoder
			self.model=GRU_Encoder(argdict) #, self.datasets['train'].vocab_size, self.argdict['embedding_size'],
								   # hidden_size=self.argdict['hidden_size'], latent_size=self.argdict['latent_size'])
		elif encoder.lower()=="cnn":
			from Encoders.CNN import CNN_Encoder
			self.model=CNN_Encoder(argdict, )
		elif encoder.lower()=='linear':
			from Encoders.Linear import Linear_Encoder
			self.model=Linear_Encoder(argdict)
		else:
			raise ValueError("unrecognized Encoder")

		self.means=[]
		self.logps=[]


	def forward(self, input):
		mean, logp=self.model(input)
		mean_norm=torch.norm(mean, dim=-1)
		self.means.append(torch.mean(mean_norm).item())
		logp_norm=torch.norm(logp, dim=-1)
		self.logps.append(torch.mean(logp_norm).item())
		return mean, logp
