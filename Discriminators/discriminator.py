import torch.nn as nn
import torch

class discriminator(nn.Module):
	"""Discriminator should you ever need one"""

	def __init__(self, argdict):
		super().__init__()
		self.argdict=argdict
		# self.datasets=datasets

		encoder=self.argdict['discriminator']
		if encoder.lower()=="gru":
			from Discriminators.GRU import GRU_Discriminator
			self.model=GRU_Discriminator(argdict)

		#
		# self.means=[]
		# self.logps=[]

	#
	# def forward(self, input):
	# 	mean, logp=self.model(input)
	# 	mean_norm=torch.norm(mean, dim=-1)
	# 	self.means.append(torch.mean(mean_norm).item())
	# 	logp_norm=torch.norm(logp, dim=-1)
	# 	self.logps.append(torch.mean(logp_norm).item())
	# 	return mean, logp
