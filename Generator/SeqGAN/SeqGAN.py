

class SeqGAN():

    def __init__(self, argdict, datasets):
        self.argdict=argdict
        self.splits=['train', 'dev', 'test']
        self.datasets=datasets