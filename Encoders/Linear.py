import torch.nn as nn
import torch.nn.functional as F

class Linear_Encoder(nn.Module):

	def __init__(self, argdict):
		super().__init__()
		self.argdict=argdict
		self.fc1 = nn.Linear(self.argdict['input_size'], self.argdict['hidden_size_encoder'][0])
		self.fc2 = nn.Linear(self.argdict['hidden_size_encoder'][0], self.argdict['hidden_size_encoder'][1])
		self.hidden2mean = nn.Linear(self.argdict['hidden_size_encoder'][1], self.argdict['latent_size'])
		self.hidden2logv = nn.Linear(self.argdict['hidden_size_encoder'][1], self.argdict['latent_size'])

	def forward(self, batch, append_labels=False):
		input_sequence=batch['input']
		input_sequence=input_sequence.view(-1, self.argdict['input_size']).to('cuda').float()
		h = F.relu(self.fc1(input_sequence))
		if append_labels:
			labs=batch['label'].unsqueeze(1).repeat(1, input_embedding.shape[1]).cuda()
			h[:, -1]=labs
			print(h)
		fds
		hidden = F.relu(self.fc2(h))
		mean = self.hidden2mean(hidden)
		logv = self.hidden2logv(hidden)
		return mean, logv