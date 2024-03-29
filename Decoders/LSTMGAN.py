import torch
import torch.nn.functional as F
import math
import torch.nn as nn


class LSTMGenerator(nn.Module):

    def __init__(self, argdict, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu=False):
        super(LSTMGenerator, self).__init__()
        self.name = 'vanilla'
        self.argdict=argdict

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.gpu = gpu

        self.temperature = 1.0

        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.lstm2out = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

        self.init_params()

    def forward(self, inp, hidden, need_hidden=False):
        """
        Embeds input and applies LSTM
        :param inp: batch_size * seq_len
        :param hidden: (h, c)
        :param need_hidden: if return hidden, use for sampling
        """
        # print(inp.shape)
        # print(hidden[0].shape, hidden[1].shape)
        emb = self.embeddings(inp)  # batch_size * len * embedding_dim
        if len(inp.size()) == 1:
            emb = emb.unsqueeze(1)  # batch_size * 1 * embedding_dim
        # print(emb.shape)
        # print(hidden[0].shape, hidden[1].shape)
        # print(hidden.shape)
        out, hidden = self.lstm(emb, hidden)  # out: batch_size * seq_len * hidden_dim
        out = out.contiguous().view(-1, self.hidden_dim)  # out: (batch_size * len) * hidden_dim
        out = self.lstm2out(out)  # (batch_size * seq_len) * vocab_size
        # out = self.temperature * out  # temperature
        pred = self.softmax(out)

        if need_hidden:
            return pred, hidden
        else:
            return pred

    def sample(self, num_samples, batch_size, start_letter):
        """
        Samples the network and returns num_samples samples of length max_seq_len.
        :return samples: num_samples * max_seq_length (a sampled sequence in each row)
        """
        num_batch = num_samples // batch_size + 1 if num_samples != batch_size else 1
        samples = torch.zeros(num_batch * batch_size, self.max_seq_len).long()

        # Generate sentences with multinomial sampling strategy
        for b in range(num_batch):
            hidden = self.init_hidden(batch_size)
            inp = torch.LongTensor([start_letter] * batch_size).cuda()

            for i in range(self.max_seq_len):
                out, hidden = self.forward(inp, hidden, need_hidden=True)  # out: batch_size * vocab_size
                next_token = torch.multinomial(torch.exp(out), 1)  # batch_size * 1 (sampling from each row)
                samples[b * batch_size:(b + 1) * batch_size, i] = next_token.view(-1)
                inp = next_token.view(-1)
        samples = samples[:num_samples]

        return samples

    def init_params(self):
        for param in self.parameters():
            if param.requires_grad and len(param.shape) > 0:
                stddev = 1 / math.sqrt(param.shape[0])
                torch.nn.init.normal_(param, std=stddev)
                # if cfg.gen_init == 'uniform':
                #     torch.nn.init.uniform_(param, a=-0.05, b=0.05)
                # elif cfg.gen_init == 'normal':
                #     torch.nn.init.normal_(param, std=stddev)
                # elif cfg.gen_init == 'truncated_normal':
                #     truncated_normal_(param, std=stddev)

    def init_oracle(self):
        for param in self.parameters():
            if param.requires_grad:
                torch.nn.init.normal_(param, mean=0, std=1)

    def init_hidden(self, batch_size=8):
        h = torch.zeros(1, batch_size, self.hidden_dim)
        c = torch.zeros(1, batch_size, self.hidden_dim)

        return h.cuda(), c.cuda()


class SeqGAN_G(LSTMGenerator):
    def __init__(self, argdict, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu=False):
        super(SeqGAN_G, self).__init__(argdict, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu)
        self.name = 'seqgan'

    def batchPGLoss(self, inp, target, reward):
        """
        Returns a policy gradient loss

        :param inp: batch_size x seq_len, inp should be target with <s> (start letter) prepended
        :param target: batch_size x seq_len
        :param reward: batch_size (discriminator reward for each sentence, applied to each token of the corresponding sentence)
        :return loss: policy loss
        """

        batch_size, seq_len = inp.size()
        hidden = self.init_hidden(batch_size)

        out = self.forward(inp, hidden).view(batch_size, self.max_seq_len, self.vocab_size)
        target_onehot = F.one_hot(target, self.vocab_size).float()  # batch_size * seq_len * vocab_size
        pred = torch.sum(out * target_onehot, dim=-1)  # batch_size * seq_len
        loss = -torch.sum(pred * reward)

        return loss