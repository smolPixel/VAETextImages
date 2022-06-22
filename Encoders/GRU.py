import torch.nn as nn


class GRU_Encoder(nn.Module):

	def __init__(self, argdict, vocab_size, embedding_size, hidden_size, latent_size):
		super().__init__()
		print("Think about why there aint no embedding dropout on encoder?")
		self.argdict=argdict

		self.embedding = nn.Embedding(vocab_size, embedding_size)
		self.rnn=nn.GRU(embedding_size, hidden_size, num_layers=1, bidirectional=False,
                               batch_first=True)

		self.hidden2mean = nn.Linear(hidden_size, latent_size)
		self.hidden2logv = nn.Linear(hidden_size, latent_size)

	def forward(self, input_sequence):
		input_embedding = self.embedding(input_sequence)
		_, hidden = self.rnn(input_embedding)
		mean = self.hidden2mean(hidden)
		logv = self.hidden2logv(hidden)
		return mean, logv