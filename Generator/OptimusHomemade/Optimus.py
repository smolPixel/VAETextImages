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
from metrics import calc_au, calc_mi

from sklearn.svm import LinearSVC

class OptimusVAE():
	def __init__(self, argdict, train, dev, test):
		self.datasets={'train':train, 'dev':dev, 'test': test}
		self.argdict=argdict
		self.splits=['train', 'dev', 'test']
		self.model=OptimusHomemade(argdict)
		self.device='cuda'

		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
		self.loss_function_basic = train.loss_function
		self.step = 0
		self.epoch = 0
		if argdict['dataset'] in ['SST2']:
			self.loss_function_ppl = torch.nn.CrossEntropyLoss(reduction='mean')
		else:
			self.loss_function_ppl = self.loss_function_basic

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
				outputs, mean, logv, z=self.model(batch)
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
			samples, z = self.model.inference(z=z)
			# print(gend)
			for sent, gen in zip(batch['sentence'], samples):
				print(f"Original sentence: {sent}, generated: {gen}")
			break



	def encode(self):
		with torch.no_grad():
			dico = {}
			for split in self.splits:
				data_loader = DataLoader(
					dataset=self.datasets[split],
					batch_size=self.argdict['batch_size'],
					shuffle=False,
					num_workers=cpu_count(),
					pin_memory=torch.cuda.is_available()
				)
				# Enable/Disable Dropout

				self.model.eval()
				# print(f"The dataset length is {len(data_loader.dataset)}")
				dataset = torch.zeros(len(data_loader.dataset), self.argdict['latent_size'])
				labels = torch.zeros(len(data_loader.dataset))
				sentences = []
				counter = 0
				for iteration, batch in enumerate(data_loader):
					# print("Oh la la banana")
					batch_size = batch['input'].size(0)
					# print(batch['input'].shape)
					# for k, v in batch.items():
					# 	if torch.is_tensor(v):
					# 		batch[k] = to_var(v)
					#
					# print(batch['input'])
					# print(batch['input'].shape)
					z = self.model.encode(batch)
					# print(batch_size)
					# print(z.shape)
					dataset[counter:counter + batch_size] = z
					labels[counter:counter + batch_size] = batch['label']
					counter += batch_size
				# print(dataset)
				dico[f"labels_{split}"] = labels
				dico[f"encoded_{split}"] = dataset
			# torch.save(labels, f"bin/labels_{split}.pt")
			# torch.save(dataset, f"bin/encoded_{split}.pt")
			return dico

	def kl_anneal_function(self, anneal_function, step, k, x0):
		if anneal_function == 'logistic':
			return float(1 / (1 + np.exp(-k * (step - x0))))
		elif anneal_function == 'linear':
			return min(1, step / x0)

	def loss_fn(self, logp, target, mean, logv, anneal_function, step, k):
		NLL_loss = self.loss_function_basic(logp, target)
		# KL Divergence
		KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
		KL_weight = self.kl_anneal_function(anneal_function, step, k, self.dataset_length * self.argdict['x0'])

		return NLL_loss, KL_loss, KL_weight

	def test_model(self):
		data_loader = DataLoader(
			dataset=self.datasets['test'],
			batch_size=self.argdict['batch_size'],
			shuffle=False,
			num_workers=cpu_count(),
			pin_memory=torch.cuda.is_available()
		)

		self.model.eval()

		Average_loss = []
		Average_NLL = []
		Average_KL_Div = []
		MIs = []
		mus = []
		NLL_mean_for_ppl = []
		for iteration, batch in enumerate(data_loader):
			# Forward pass
			with torch.no_grad():
				outputs, mean, logv, z = self.model(batch)
			logp=outputs['logits']
			# Keeping track of the means for AU
			mus.append(mean.detach().squeeze(0))
			batch_size = logp.shape[0]
			target=outputs['encoded_output']['input_ids']
			# logp, target = self.datasets['train'].shape_for_loss_function(logp, batch['target'])
			# print(target)
			logp, target = self.datasets['train'].shape_for_loss_function(logp[:, :-1, :].contiguous(), target[:, 1:])
			NLL_loss, KL_loss, KL_weight = self.loss_fn(logp, target.to('cuda'), mean, logv, 'logistic', self.step, 0.0025)



			NLL_mean = self.loss_function_ppl(logp, target.to('cuda'))

			loss = (NLL_loss + KL_weight * KL_loss) / batch_size
			Average_loss.append(outputs['loss'].cpu())
			# Average_KL_Div.append(KL_loss.cpu().detach() / batch_size)
			# Average_NLL.append(NLL_loss.cpu().detach() / batch_size)
			# NLL_mean_for_ppl.append(NLL_mean.cpu().detach())
			# aggr=self.get_aggregate()
			MIs.append(calc_mi(z, mean, logv))
		# print(MIs)
		# fds

		# print(MIs)
		AU = calc_au(mus)
		encoded = self.encode()
		X = encoded['encoded_test']
		Y = encoded['labels_test']

		svc = LinearSVC()
		svc.fit(X, Y)
		sep = svc.score(X, Y)
		# print(AU)
		return {'Mean ELBO': np.mean(Average_loss), 'Mean LF': np.mean(Average_NLL),
				'Mean KL div': np.mean(Average_KL_Div), 'PPL': {torch.exp(torch.mean(torch.Tensor(NLL_mean_for_ppl)))},
				'Separability': sep, 'MI': {np.mean(MIs)}, 'Active Units': AU[0]}

