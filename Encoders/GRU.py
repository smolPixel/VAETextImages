import torch.nn as nn
import torch

class GRU_Encoder(nn.Module):

	def __init__(self, argdict):#, vocab_size, embedding_size, hidden_size, latent_size):
		super().__init__()
		print("Think about why there aint no embedding dropout on encoder?")
		self.argdict=argdict

		self.embedding = nn.Embedding(argdict['input_size'], argdict['embedd_size'])
		self.rnn=nn.GRU(argdict['embedd_size'], argdict['hidden_size'], num_layers=1, bidirectional=False,
                               batch_first=True)

		self.hidden2mean = nn.Linear(argdict['hidden_size'], argdict['latent_size'])
		self.hidden2logv = nn.Linear(argdict['hidden_size'], argdict['latent_size'])

	def forward(self, batch, append_labels=False):

		input_sequence=batch['input'].to('cuda')
		batch_size=input_sequence.shape[0]

		if isinstance(input_sequence, torch.LongTensor) or (
				torch.cuda.is_available() and isinstance(input_sequence, torch.cuda.LongTensor)):
			input_embedding = self.embedding(input_sequence)
		else:
			input_embedding = torch.matmul(input_sequence, self.embedding.weight)
		if append_labels:
			labs=batch['label'].unsqueeze(1).repeat(1, input_embedding.shape[1]).cuda()
			input_embedding[:, :, -1]=labs

		hidden=torch.zeros((1, input_embedding.shape[0], self.argdict['hidden_size'])).cuda()
		len_seq=input_embedding.shape[1]

		hidden_states=torch.zeros(batch_size, len_seq, self.argdict['hidden_size'])

		for i in range(len_seq):
			inp=input_embedding[:, i, :].unsqueeze(1)
			_, hidden=self.rnn(inp, hidden)
			hidden_states[:, i, :]=hidden.squeeze(0)


		if self.argdict['pooling']=="mean":
			hidden=torch.mean(hidden_states, dim=1)

		print(hidden.shape)
		fds
		#For pooling, we need to do this one step at a time

		# _, hidden = self.rnn(input_embedding)

		mean = self.hidden2mean(hidden)
		logv = self.hidden2logv(hidden)
		return mean, logv