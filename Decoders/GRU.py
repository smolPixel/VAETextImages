import torch.nn as nn


class GRU_Decoder(nn.Module):

	def __init__(self, argdict, vocab_size, embedding_size, hidden_size):
		super().__init__()
		print("Think about why there aint no embedding dropout on encoder?")
		self.argdict=argdict

		self.embedding = nn.Embedding(vocab_size, embedding_size)
		self.rnn=nn.GRU(embedding_size, hidden_size, num_layers=1, bidirectional=False,
                               batch_first=True)
	def forward(self, input_sequence, latent_space):
		input_embedding = self.embedding(input_sequence)
		_, hidden = self.rnn(input_embedding)
		return _, hidden