import torch.nn as nn
import torch.nn.functional as F

class Linear_Decoder(nn.Module):

	def __init__(self, argdict):
		super().__init__()
		self.argdict=argdict
		self.latent2hidden = nn.Linear(self.argdict['latent_size'], self.argdict['hidden_size_encoder'][1])
		self.fc5 = nn.Linear(self.argdict['hidden_size_encoder'][1], self.argdict['hidden_size_encoder'][0])
		self.fc6 = nn.Linear(self.argdict['hidden_size_encoder'][0], self.argdict['input_size'])

	def forward(self, batch, z, append_labels=False):
		if append_labels:
			labs = batch['label'].unsqueeze(0).cuda()
			z[:, -1] = labs
		h = F.relu(self.latent2hidden(z))
		h = F.relu(self.fc5(h))
		return F.sigmoid(self.fc6(h))

	def generate(self, z, labels=None):
		if labels is not None:
			labs = batch['label'].unsqueeze(0).cuda()
			z[:, -1] = labs
		h = F.relu(self.latent2hidden(z))
		h = F.relu(self.fc5(h))
		return F.sigmoid(self.fc6(h))