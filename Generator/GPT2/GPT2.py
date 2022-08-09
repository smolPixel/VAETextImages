from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
import numpy as np

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
		self.device = "cuda:0"
		model_id = "gpt2"
		model = GPT2LMHeadModel.from_pretrained(model_id).to(self.device)
		self.tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
		self.tokenizer.pad_token = self.tokenizer.eos_token
		return model, None

	def train(self):
		pass

	def loss_fn(self, logp, target):

		# Negative Log Likelihood
		NLL_loss = self.loss_function_basic(logp, target)
		# BCE = torch.nelf.kl_anneal_function(anneal_function, step, k, self.dataset_length*self.argdict['x0'])

		return NLL_loss

	def test(self):
		data_loader = DataLoader(
			dataset=self.datasets['test'],
			batch_size=16,  # self.argdict.batch_size,
			shuffle=False,
			num_workers=cpu_count(),
			pin_memory=torch.cuda.is_available()
		)

		self.model.eval()


		Average_loss=[]
		Average_NLL=[]
		Average_KL_Div=[]
		for iteration, batch in enumerate(data_loader):
			encodings = self.tokenizer(batch['sentence'], return_tensors="pt", padding=True, truncation=True).to(self.device)
			target=encodings['input_ids']#[1:]
			batch_size=target.shape[0]
			# Forward pass
			outputs=self.model(encodings['input_ids'], labels=encodings['input_ids'].clone())
			logp=outputs[1]
			logp, target=self.datasets['train'].shape_for_loss_function(logp, target)
			NLL_loss= self.loss_fn(logp, target)
			loss = (NLL_loss) / batch_size
			Average_loss.append(loss.item())
			Average_NLL.append(NLL_loss.cpu().detach()/batch_size)


		print(Average_NLL)
		return {'Mean ELBO': np.mean(Average_loss), 'Mean LF' :np.mean(Average_NLL), 'Mean KL div' :np.mean(Average_KL_Div), 'PPL': {torch.exp(torch.mean(torch.Tensor(Average_NLL)))}}