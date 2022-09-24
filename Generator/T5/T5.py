from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
import numpy as np

class T5():

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
		model_id = "t5-small"
		model = T5ForConditionalGeneration.from_pretrained(model_id).to(self.device)
		self.tokenizer = T5Tokenizer.from_pretrained(model_id)
		self.tokenizer.pad_token = self.tokenizer.eos_token
		return model, None

	def run_epoch(self):
		for split in self.splits:

			data_loader = DataLoader(
				dataset=self.datasets[split],
				batch_size=8,  # self.argdict.batch_size,
				shuffle=split == 'train',
				num_workers=cpu_count(),
				pin_memory=False
			)

			# tracker = defaultdict(tensor)

			# Enable/Disable Dropout
			if split == 'train':
				self.model.train()
				self.dataset_length = len(data_loader)
			else:
				self.model.eval()

			Average_loss = []
			Average_NLL = []
			Average_KL_Div = []
			for iteration, batch in enumerate(data_loader):
				# Forward pass
				encodings = self.tokenizer(batch['sentence'], return_tensors="pt", padding=True, truncation=True).to(self.device)
				target = encodings['input_ids']
				batch_size = target.shape[0]
				outputs=self.model(encodings['input_ids'], labels=encodings['input_ids'].clone())
				# print(batch_size)
				loss=outputs[0]
				# backward + optimization
				if split == 'train':
					self.optimizer.zero_grad()
					loss.backward()
					self.optimizer.step()

				# Average_loss.append(loss.item().cpu())
				Average_NLL.append(outputs[0].cpu().detach() / batch_size)

			print(f"{split.upper()} Epoch {self.epoch}/{self.argdict['nb_epoch']}, Mean LF {np.mean(Average_NLL)}")

	def train_model(self):
		for epoch in range(self.argdict['nb_epoch']):
			self.epoch = epoch
			self.run_epoch()

	def loss_fn(self, logp, target):

		# Negative Log Likelihood
		NLL_loss = self.loss_function_basic(logp, target)
		# BCE = torch.nelf.kl_anneal_function(anneal_function, step, k, self.dataset_length*self.argdict['x0'])

		return NLL_loss

	def test_model(self):
		data_loader = DataLoader(
			dataset=self.datasets['test'],
			batch_size=4,  # self.argdict.batch_size,
			shuffle=False,
			num_workers=1,
			pin_memory=torch.cuda.is_available()
		)

		self.model.eval()


		Average_loss=[]
		Average_NLL=[]
		average_nll_gpt=[]
		Average_KL_Div=[]
		for iteration, batch in enumerate(data_loader):
			encodings = self.tokenizer(batch['sentence'], return_tensors="pt", padding=True, truncation=True).to(self.device)
			target=encodings['input_ids']
			batch_size=target.shape[0]
			# Forward pass
			outputs=self.model(encodings['input_ids'], labels=encodings['input_ids'].clone())
			logp=outputs[1]
			average_nll_gpt.append(outputs[0].detach())
			logp, target=self.datasets['train'].shape_for_loss_function(logp[:, :-1, :].contiguous(), target[:, 1:])
			NLL_loss= self.loss_fn(logp, target)


			loss = (NLL_loss) / batch_size
			Average_loss.append(loss.item())
			Average_NLL.append(NLL_loss.cpu().detach())


		# print(Average_NLL)
		# print(average_nll_gpt)
		return {'Mean ELBO': np.mean(Average_loss), 'Mean LF' :np.mean(Average_NLL), 'Mean KL div' :np.mean(Average_KL_Div), 'PPL': {torch.exp(torch.mean(torch.Tensor(Average_NLL)))},
				'PPL_GPT':torch.exp(torch.mean(torch.Tensor(average_nll_gpt)))}