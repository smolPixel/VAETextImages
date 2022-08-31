import torch
import torch.nn as nn
from Generator.utils import to_var


class CVAE_model(nn.Module):
	def __init__(self, argdict, encoder, decoder):

		super().__init__()
		self.argdict = argdict
		self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

		self.encoder = encoder
		self.decoder = decoder
		self.num_classes=len(argdict["categories"])

	def forward(self, batch):

		input_sequence = batch['input']
		labels= batch['label']

		batch_size = input_sequence.size(0)
		mean, logv = self.encoder(input_sequence)
		std = torch.exp(0.5 * logv)

		z = to_var(torch.randn([batch_size, self.argdict['latent_size']]))
		z = z * std + mean
		print(labels)
		z[:, :, -1]=labels
		#last two dims
		logp = self.decoder(input_sequence, z)

		return logp, mean, logv, z


	def encode(self, input_sequence, labels):
		batch_size = input_sequence.size(0)
		# sorted_lengths, sorted_idx = torch.sort(length, descending=True)
		# input_sequence = input_sequence[sorted_idx]

		# ENCODER
		input_embedding = self.embedding(input_sequence)
		# labels = labels.cuda()

		_, hidden = self.encoder_rnn(input_embedding)

		if self.bidirectional or self.num_layers > 1:
			# flatten hidden state
			hidden = hidden.view(batch_size, self.hidden_size * self.hidden_factor)
		else:
			hidden = hidden.squeeze()

		# REPARAMETERIZATION
		mean = self.hidden2mean(hidden)
		logv = self.hidden2logv(hidden)
		std = torch.exp(0.5 * logv)

		z = to_var(torch.randn([batch_size, self.latent_size]))
		z = z * std + mean

		return z

	def inference(self,  n=4, z=None, cat=0):
		if z is None:
			batch_size = n
			z = to_var(torch.randn([batch_size, self.latent_size]))
		else:
			batch_size = z.size(0)

		if type(cat)==int:
			# Convert to one hot and concat
			ll = torch.zeros((batch_size, self.num_classes)).cuda()
			ll[:, cat] = 1
		elif len(cat.shape)==1:
			ll = torch.zeros((batch_size, self.num_classes)).cuda()
			for i, ind in enumerate(cat):
				ll[i, ind]=1
		else:
			#If cat is already a tensor
			ll=cat


		z = torch.cat([z, ll], dim=1)
		# print(z)

		hidden = self.latent2hidden(z)

		hidden = hidden.view(batch_size, self.hidden_factor, self.hidden_size)
		hidden = torch.transpose(hidden, 0, 1)
		hidden = hidden.contiguous()

		# required for dynamic stopping of sentence generation
		# sequence_idx = torch.arange(0, batch_size, out=self.tensor()).long()  # all idx of batch
		# all idx of batch which are still generating
		# sequence_running = torch.arange(0, batch_size, out=self.tensor()).long()
		# sequence_mask = torch.ones(batch_size, out=self.tensor()).bool()
		# idx of still generating sequences with respect to current loop
		# running_seqs = torch.arange(0, batch_size, out=self.tensor()).long()

		generations = self.tensor(batch_size, self.max_sequence_length).fill_(self.pad_idx).long()

		t = 0
		# input_sequence = input_sequence.unsqueeze(1)
		while t < self.max_sequence_length:

			if t == 0:
				input_sequence = to_var(torch.Tensor(batch_size).fill_(self.sos_idx).long())
				input_sequence = input_sequence.unsqueeze(1)

			input_embedding = self.embedding(input_sequence)
			if type(cat) == int:
				# Convert to one hot and concat
				ll = torch.zeros((input_embedding.shape[0], self.num_classes)).cuda()
				ll[:, cat] = 1
			elif len(cat.shape) == 1:
				ll = torch.zeros((batch_size, self.num_classes)).cuda()
				for i, ind in enumerate(cat):
					ll[i, ind] = 1
			else:
				# If cat is already a tensor
				ll = cat

			# print(input_embedding.shape)
			# print(ll.shape)

			input_embedding = torch.cat([input_embedding, ll.unsqueeze(1)], dim=2)

			output, hidden = self.decoder_rnn(input_embedding, hidden)

			logits = self.outputs2vocab(output)

			input_sequence = torch.argmax(logits, dim=-1)
			generations[:, t]=input_sequence.squeeze(1)
			t += 1

		return generations, z
