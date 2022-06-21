import torch.nn as nn


class GRU_Encoder(nn.Module):

	def __init__(self, argdict, vocab_size, embedding_size):
		super().__init__()
		self.argdict=argdict

		self.embedding = nn.Embedding(vocab_size, embedding_size)
	def forward(self, input):
		pass