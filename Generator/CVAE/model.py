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
		mean, logv = self.encoder(batch)
		std = torch.exp(0.5 * logv)

		z = to_var(torch.randn([batch_size, self.argdict['latent_size']]))
		z = z * std + mean
		if len(z.shape)==3:
			z[:, :, -1]=labels
		elif len(z.shape)==2:
			z[:, -1]=labels
		else:
			raise ValueError()
		#last two dims
		logp = self.decoder(batch, z)

		return logp, mean, logv, z


	def encode(self, input_sequence, labels):
		batch_size = input_sequence.size(0)

		batch_size = input_sequence.size(0)
		mean, logv = self.encoder(input_sequence)
		std = torch.exp(0.5 * logv)

		z = to_var(torch.randn([batch_size, self.argdict['latent_size']]))
		z = z * std + mean
		if len(z.shape) == 3:
			z[:, :, -1] = labels
		elif len(z.shape) == 2:
			z[:, -1] = labels
		else:
			raise ValueError()
		return z, mean, logv

	def inference(self, z, labels):

		if len(z.shape) == 3:
			z[:, :, -1] = labels
		elif len(z.shape) == 2:
			z[:, -1] = labels
		else:
			raise ValueError()

		generated = self.decoder.generate(z)  # , pad_idx=self.pad_idx, sos_idx=self.sos_idx)

		return generated, z