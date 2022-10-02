# from transformers import (WEIGHTS_NAME, AdamW, WarmupLinearSchedule,
#                                   BertConfig, BertForLatentConnector, BertTokenizer,
#                                   GPT2Config, , GPT2Tokenizer,
#                                   OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
#                                   RobertaConfig, RobertaForMaskedLM, RobertaTokenizer)


from torch.utils.data import DataLoader
from multiprocessing import cpu_count
import numpy as np
from Generator.OptimusHomemade.model import OptimusHomemade
import torch

class OptimusVAE():
	def __init__(self, argdict, train, dev, test):
		self.datasets={'train':train, 'dev':dev, 'test': test}
		self.argdict=argdict
		self.splits=['train', 'dev']
		self.model=OptimusHomemade(argdict)
		self.device='cuda'

		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

	def run_epoch(self):
		for split in self.splits:

			data_loader = DataLoader(
				dataset=self.datasets[split],
				batch_size=self.argdict['batch_size'],
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
				# encodings_input = self.tokenizer_encoder(batch['sentence'], return_tensors="pt", padding=True, truncation=True).to(self.device)
				# target = encodings_input['input_ids']
				batch_size = len(batch['sentence'])
				outputs=self.model(batch)
				# print(batch_size)
				loss=outputs['loss']
				# backward + optimization
				if split == 'train':
					self.optimizer.zero_grad()
					loss.backward()
					self.optimizer.step()

				# Average_loss.append(loss.item().cpu())
				Average_NLL.append(outputs['loss'].cpu().detach() / batch_size)

			print(f"{split.upper()} Epoch {self.epoch}/{self.argdict['nb_epoch']}, Mean LF {np.mean(Average_NLL)}")

	def train_model(self):
		for epoch in range(self.argdict['nb_epoch']):
			self.epoch = epoch
			self.run_epoch()

		self.generate_from_train()

	def generate_from_train(self):
		data_loader = DataLoader(
			dataset=self.datasets['train'],
			batch_size=2,  # self.argdict.batch_size,
			shuffle=False,
			num_workers=cpu_count(),
			pin_memory=torch.cuda.is_available()
		)

		self.model.eval()

		for iteration, batch in enumerate(data_loader):

			batch_size = batch['input'].size(0)
			# Forward pass
			output, mean, logv, z = self.model(batch)
			samples, z = self.model.inference(z=z.squeeze(0))
			# print(gend)
			for sent, gen in zip(batch['sentence'], gend):
				print(f"Original sentence: {sent}, generated: {gen}")
			break


