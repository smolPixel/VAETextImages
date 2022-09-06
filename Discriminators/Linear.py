import torch.nn as nn
import torch
import torch.nn.functional as F

class Linear_Discriminator(nn.Module):

	def __init__(self, argdict):#, vocab_size, embedding_size, hidden_size, latent_size):
		super().__init__()
		print("Think about why there aint no embedding dropout on Discriminator?")
		self.argdict=argdict

		self.fc1 = nn.Linear(self.argdict['input_size'], self.argdict['hidden_size_encoder'][0])
		self.fc2 = nn.Linear(self.argdict['hidden_size_encoder'][0], self.argdict['hidden_size_encoder'][1])
		self.linear = nn.Linear(self.argdict['hidden_size_encoder'][1], self.argdict['num_classes'])

	def forward(self, batch, append_labels):
		input_sequence=batch['input']
		input_sequence = input_sequence.view(-1, self.argdict['input_size']).to('cuda').float()
		h = F.relu(self.fc1(input_sequence))
		if append_labels:
			labs=batch['label'].unsqueeze(0).cuda()
			h[:, -1]=labs
		hidden = F.relu(self.fc2(h))
		logits = self.linear(hidden)
		return logits