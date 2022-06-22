import torch.nn as nn
import torch

class GRU_Decoder(nn.Module):

	def __init__(self, argdict, vocab_size, embedding_size, hidden_size, latent_size):
		super().__init__()
		print("Think about why there aint no embedding dropout on encoder?")
		self.argdict=argdict

		self.hidden_size=hidden_size

		self.latent2hidden=nn.Linear(latent_size, hidden_size)
		self.embedding = nn.Embedding(vocab_size, embedding_size)
		self.rnn=nn.GRU(embedding_size, hidden_size, num_layers=1, bidirectional=False,
                               batch_first=True)
		self.outputs2vocab = nn.Linear(hidden_size, vocab_size)

	def forward(self, input_sequence, latent_space):
		input_embedding = self.embedding(input_sequence)
		outputs, hidden = self.rnn(input_embedding)
		logp = nn.functional.log_softmax(self.outputs2vocab(outputs), dim=-1)
		return logp

	def generate(self, z):
		batch_size = z.size(0)

		hidden = self.latent2hidden(z)

		# if self.bidirectional or self.num_layers > 1:
		#     # unflatten hidden state
		#     hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
		#     #Added the else here otherwise it was always unsqueezing which made it bug for bidir
		# else:
		#     hidden = hidden.unsqueeze(0)
		# if self.num_layers > 1:
		# unflatten hidden state
		hidden = hidden.view(batch_size, 1, self.hidden_size)
		hidden = torch.transpose(hidden, 0, 1)
		hidden = hidden.contiguous()
		# hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
		# Added the else here otherwise it was always unsqueezing which made it bug for bidir
		# else:
		#     hidden = hidden.unsqueeze(0)

		generations = self.tensor(batch_size, self.max_sequence_length).fill_(self.pad_idx).long()

		t = 0
		while t < self.max_sequence_length:

			if t == 0:
				input_sequence = torch.Tensor(batch_size).fill_(self.sos_idx).long()
				input_sequence = input_sequence.unsqueeze(1)

			input_embedding = self.embedding(input_sequence)

			output, hidden = self.decoder_rnn(input_embedding, hidden)

			# output = self.outputs2embeds(output)

			# logits = self.embed2vocab(output)
			logits = self.outputs2vocab(output)

			input_sequence = torch.argmax(logits, dim=-1)
			generations[:, t] = input_sequence.squeeze(1)
			t += 1