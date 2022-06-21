import torch.nn as nn


class encoder(nn.Module):

	def __init__(self, argdict):
		self.argdict=argdict

		encoder=self.argdict['encoder']
		if encoder.lower()=="gru":
			from Encoders.GRU import GRU_Encoder
			self.model=GRU_Encoder


	def forward(self, input):
		return self.model(input)
