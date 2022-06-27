import torch.nn as nn
import torch.nn.functional as F

class Linear_Decoder(nn.Module):

	def __init__(self, argdict):
		super().__init__()
		self.argdict=argdict
		self.fc4 = nn.Linear(self.argdict['latent_size'], self.argdict['hidden_size_encoder'][1])
		self.fc5 = nn.Linear(self.argdict['hidden_size_encoder'][1], self.argdict['hidden_size_encoder'][0])
		self.fc6 = nn.Linear(self.argdict['hidden_size_encoder'][0], self.argdict['input_size'])

	def forward(self, input_sequence, z):
		h = F.relu(self.fc4(z))
		h = F.relu(self.fc5(h))
		return F.sigmoid(self.fc6(h))

	def generate(self, z):
		h = F.relu(self.fc4(z))
		h = F.relu(self.fc5(h))
		return F.sigmoid(self.fc6(h))