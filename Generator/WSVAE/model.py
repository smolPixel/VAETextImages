import torch
import torch.nn as nn
# import torch.nn.utils.rnn as rnn_utils
from Generator.utils import to_var
import torch.nn.functional as F

class WSVAE_model(nn.Module):
    def __init__(self, argdict, encoder, decoder, discriminator):

        super().__init__()
        self.argdict=argdict
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor



        self.encoder=encoder
        self.decoder=decoder
        self.discriminator=discriminator

    def forward(self, batch, pretraining=False):

        input_sequence=batch['input']

        batch_size = input_sequence.size(0)
        mean, logv=self.encoder(batch)
        # cmu, zmu = mean[:, :, -1], mean[:, :, :-1]
        # clogvar, zlogvar = logv[:, :, -1], logv[:, :, :-1]
        std = torch.exp(0.5 * logv)
        if pretraining:
            # c = torch.multinomial(torch.Tensor([0.5, 0.5]), batch_size, replacement=True)
            # c=nn.functional.one_hot(c, num_classes=2).cuda()
            c=torch.randint(self.argdict['num_classes'], ( 1, batch_size)).squeeze(0)

        else:
            c=F.gumbel_softmax(self.discriminator(batch), tau=1, hard=True, dim=-1).unsqueeze(0)
            c=torch.argmax(c, dim=-1)
            # c = torch.softmax(self.discriminator(batch['input']), dim=-1)
        #
        z = to_var(torch.randn([batch_size, std.shape[-1]]))
        z = z * std + mean
        if len(z.shape) == 3:
            z[:, :, -1] = c
        elif len(z.shape) == 2:
            # print(torch.argmax(c, dim=-1).shape)
            z[:, -1] = c
        else:
            raise ValueError()
        logp = self.decoder(batch, z)

        return logp, mean, logv, z

    def encode(self, input_sequence):
        # print("HIHIOHOHO")
        # print(input_sequence.shape)
        batch_size = input_sequence.size(0)
        # sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        # input_sequence = input_sequence[sorted_idx]

        # ENCODER
        mean, logv=self.encoder({'input':input_sequence})
        std = torch.exp(0.5 * logv)

        z = to_var(torch.randn([batch_size, self.argdict['latent_size']]))
        # cmu, zmu = mean[:, :, -1], mean[:, :, :-1]
        # if len(z.shape) == 3:
        #     cmu, zmu = mean[:, :, -1], mean[:, :, :-1]
        #     clogvar, zlogvar = logv[:, :, -1], logv[:, :, :-1]
        # elif len(z.shape) == 2:
        #     # print(torch.argmax(c, dim=-1).shape)
        #     cmu, zmu = mean[:, -1], mean[:, :-1]
        #     clogvar, zlogvar = logv[:, -1], logv[:, :-1]
        # else:
        #     raise ValueError()
        # clogvar, zlogvar = logv[:, :, -1], logv[:, :, :-1]
        z = to_var(torch.randn([batch_size, std.shape[-1]]))
        z = z * std + mean
        c = F.gumbel_softmax(self.discriminator(input_sequence), tau=1, hard=True, dim=-1).unsqueeze(0)
        c = torch.argmax(c, dim=-1)
        if len(z.shape) == 3:
            z[:, :, -1] = c
        elif len(z.shape) == 2:
            # print(torch.argmax(c, dim=-1).shape)
            z[:, -1] = c
        else:
            raise ValueError()
        # print(z.shape)

        return z

    def discriminate(self, input):
        return self.discriminator.forward({'input':input})

    def inference(self, z, labels):

        if len(z.shape) == 3:
            z[:, :, -1] = labels
        elif len(z.shape) == 2:
            z[:, -1] = labels
        else:
            raise ValueError()

        generated = self.decoder.generate(z, None)  # , pad_idx=self.pad_idx, sos_idx=self.sos_idx)

        return generated, z