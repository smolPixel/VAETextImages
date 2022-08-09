from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch


class GPT2():

	def __init__(self, argdict, train, dev, test):
		self.argdict=argdict
		self.splits=['train', 'dev']
		self.datasets={'train':train, 'dev':dev, 'test':test}
		self.model, self.params=self.init_model_dataset()
		# optimizers
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)  # self.argdict.learning_rate)
		self.loss_function_basic=train.loss_function

	def init_model_dataset(self):
		device = "cuda:0"
		model_id = "gpt2-large"
		model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
		tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
		tokenizer.pad_token = tokenizer.eos_token

	def train(self):
		pass

	def test(self):
		fds