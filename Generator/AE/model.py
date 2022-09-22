import torch
import torch.nn as nn
# import torch.nn.utils.rnn as rnn_utils
from Generator.utils import to_var


class AE_model(nn.Module):
    def __init__(self, argdict, encoder, decoder):

        super().__init__()
        self.argdict=argdict
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor



        self.encoder=encoder
        self.decoder=decoder

    def forward(self, batch):

        input_sequence=batch['input']
        batch_size = input_sequence.size(0)
        mean, logv=self.encoder(batch)
        logp = self.decoder(input_sequence, mean)
        return logp, mean

    def encode(self, input_sequence):
        batch_size = input_sequence.size(0)
        # ENCODER
        mean, logv=self.encoder(input_sequence)
        return mean

    def inference(self,  n=4, z=None):
        generated=self.decoder.generate(z)#, pad_idx=self.pad_idx, sos_idx=self.sos_idx)
        return generated, z
