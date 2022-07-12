import torch.nn as nn


class GRU_Discriminator(nn.Module):

	def __init__(self, argdict):#, vocab_size, embedding_size, hidden_size, latent_size):
		super().__init__()
		print("Think about why there aint no embedding dropout on Discriminator?")
		self.argdict=argdict

		self.embedding = nn.Embedding(argdict['input_size'], argdict['embedd_size'])
		self.rnn=nn.GRU(argdict['embedd_size'], argdict['hidden_size'], num_layers=1, bidirectional=False,
                               batch_first=True)

		self.linear = nn.Linear(argdict['hidden_size'], 1)

	def forward(self, input_sequence):
		input_sequence=input_sequence.to('cuda')
		input_embedding = self.embedding(input_sequence)
		_, hidden = self.rnn(input_embedding)
		logit = self.linear(hidden).squeeze(-1)
		return logit