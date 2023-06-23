import random
from torch.utils.data import Dataset, DataLoader

class Signal:
    """Running signal to control training process"""

    def __init__(self):
        self.pre_sig = True
        self.adv_sig = True

        self.update()

    def update(self):
        signal_dict = {'pre_sig': True, 'adv_sig': True}
        self.pre_sig = signal_dict['pre_sig']
        self.adv_sig = signal_dict['adv_sig']

class GenDataIter:
    def __init__(self, argdict, samples, path, start_letter, if_test_data=False, shuffle=None):
        self.batch_size = argdict['batch_size']
        self.max_seq_len = argdict['max_seq_len']
        self.start_letter = start_letter
        self.shuffle = True
        if cfg.if_real_data:
            self.word2idx_dict, self.idx2word_dict = load_dict(cfg.dataset, path)
        if if_test_data:  # used for the classifier
            self.word2idx_dict, self.idx2word_dict = load_test_dict(cfg.dataset, path)
        self.loader = DataLoader(
            dataset=GANDataset(self.__read_data__(samples)),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=True)
        # print(len(self.loader.dataset))
        self.input = self._all_data_('input')
        self.target = self._all_data_('target')


    def __read_data__(self, samples):
        """
        input: same as target, but start with start_letter.
        """
        # global all_data
        if isinstance(samples, torch.Tensor):  # Tensor
            inp, target = self.prepare(samples)
            all_data = [{'input': i, 'target': t} for (i, t) in zip(inp, target)]

        elif isinstance(samples, str):  # filename
            inp, target = self.load_data(samples)
            all_data = [{'input': i, 'target': t} for (i, t) in zip(inp, target)]
        else:
            all_data = None
        return all_data

    def random_batch(self):
        """Randomly choose a batch from loader, please note that the data should not be shuffled."""
        idx = random.randint(0, len(self.loader) - 1)
        return list(self.loader)[idx]

    def _all_data_(self, col):
        return torch.cat([data[col].unsqueeze(0) for data in self.loader.dataset.data], 0)

    @staticmethod
    def prepare(samples, gpu=False):
        """Add start_letter to samples as inp, target same as samples"""
        inp = torch.zeros(samples.size()).long()
        target = samples
        inp[:, 0] = cfg.start_letter
        inp[:, 1:] = target[:, :cfg.max_seq_len - 1]

        if gpu:
            return inp.cuda(), target.cuda()
        return inp, target

    def load_data(self, filename):
        """Load real data from local file"""
        self.tokens = get_tokenlized(filename)
        samples_index = tokens_to_tensor(self.tokens, self.word2idx_dict)
        return self.prepare(samples_index)