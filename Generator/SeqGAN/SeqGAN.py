from Generator.SeqGAN.seqgan_instructor import SeqGANInstructor

class SeqGAN():

    def __init__(self, argdict, datasets):
        self.argdict=argdict
        self.splits=['train', 'dev', 'test']
        self.datasets=datasets
        self.model, self.params = self.init_model_dataset()

    def init_model_dataset(self):
        self.instructor=SeqGANInstructor(self.argdict, self.datasets)
        fds