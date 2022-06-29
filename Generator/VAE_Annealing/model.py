import torch
import torch.nn as nn
# import torch.nn.utils.rnn as rnn_utils
from Generator.utils import to_var


class VAE_model(nn.Module):
    def __init__(self, argdict, encoder, decoder):

        super().__init__()
        self.argdict=argdict
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor



        self.encoder=encoder
        self.decoder=decoder

    def forward(self, batch):

        input_sequence=batch['input']

        batch_size = input_sequence.size(0)
        mean, logv=self.encoder(input_sequence)
        std = torch.exp(0.5 * logv)

        z = to_var(torch.randn([batch_size, self.argdict['latent_size']]))
        z = z * std + mean
        logp = self.decoder(input_sequence, z)

        return logp, mean, logv, z

    def encode(self, input_sequence, length):
        # print("HIHIOHOHO")
        # print(input_sequence.shape)
        batch_size = input_sequence.size(0)
        # sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        # input_sequence = input_sequence[sorted_idx]

        # ENCODER
        input_embedding = self.embedding(input_sequence)

        # packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        _, hidden = self.encoder_rnn(input_embedding)

        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            hidden = hidden.view(batch_size, self.hidden_size*self.hidden_factor)
        else:
            hidden = hidden.squeeze()

        # REPARAMETERIZATION
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)

        z = to_var(torch.randn([batch_size, self.latent_size]))
        z = z * std + mean
        # print(z.shape)

        return z

    def inference(self,  n=4, z=None):


        if z is None:
            batch_size = n
            z = to_var(torch.randn([batch_size, self.latent_size]))
        else:
            batch_size = z.size(0)


        generated=self.decoder.generate(z)#, pad_idx=self.pad_idx, sos_idx=self.sos_idx)
        #
        # hidden = self.latent2hidden(z)
        #
        # # if self.bidirectional or self.num_layers > 1:
        # #     # unflatten hidden state
        # #     hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        # #     #Added the else here otherwise it was always unsqueezing which made it bug for bidir
        # # else:
        # #     hidden = hidden.unsqueeze(0)
        # # if self.num_layers > 1:
        #     # unflatten hidden state
        # hidden=hidden.view(batch_size, self.hidden_factor, self.hidden_size)
        # hidden=torch.transpose(hidden, 0, 1)
        # hidden=hidden.contiguous()
        #     # hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        #     #Added the else here otherwise it was always unsqueezing which made it bug for bidir
        # # else:
        # #     hidden = hidden.unsqueeze(0)
        #
        # generations = self.tensor(batch_size, self.max_sequence_length).fill_(self.pad_idx).long()
        #
        # t = 0
        # while t < self.max_sequence_length:
        #
        #     if t == 0:
        #         input_sequence = to_var(torch.Tensor(batch_size).fill_(self.sos_idx).long())
        #         input_sequence = input_sequence.unsqueeze(1)
        #
        #     input_embedding = self.embedding(input_sequence)
        #
        #     output, hidden = self.decoder_rnn(input_embedding, hidden)
        #
        #     # output = self.outputs2embeds(output)
        #
        #     # logits = self.embed2vocab(output)
        #     logits = self.outputs2vocab(output)
        #
        #     input_sequence = torch.argmax(logits, dim=-1)
        #     generations[:, t]=input_sequence.squeeze(1)
        #     t += 1

        return generated, z
