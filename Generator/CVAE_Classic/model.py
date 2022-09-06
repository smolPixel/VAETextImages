import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from Generator.utils import to_var


class CVAE_Classic_model(nn.Module):
    def __init__(self, argdict, encoder, decoder):
        super().__init__()
        self.argdict = argdict
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, batch):
        input_sequence = batch['input']

        batch_size = input_sequence.size(0)
        mean, logv = self.encoder(batch, append_labels=True)
        std = torch.exp(0.5 * logv)

        z = to_var(torch.randn([batch_size, self.argdict['latent_size']]))
        z = z * std + mean
        logp = self.decoder(batch, z, append_labels=True)

        return logp, mean, logv, z

    def encode(self, batch):
        input_sequence = batch['input']

        batch_size = input_sequence.size(0)
        mean, logv = self.encoder(batch, append_labels=True)
        std = torch.exp(0.5 * logv)

        z = to_var(torch.randn([batch_size, self.argdict['latent_size']]))
        z = z * std + mean
        # print(z.shape)
        # labels=batch['label']
        # z[:, :, -1]=labels
        return z, mean, logv

    def inference(self, n=4, z=None, labels=None):
        # if z is None:
        #     batch_size = n
        #     z = to_var(torch.randn([batch_size, self.latent_size]))
        # else:
        #     batch_size = z.size(0)

        generated = self.decoder.generate(z, labels)  # , pad_idx=self.pad_idx, sos_idx=self.sos_idx)

        return generated, z
