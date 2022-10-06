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
import math
from sklearn.svm import LinearSVC

class OptimusVAE():
	def __init__(self, argdict, datasets, datasetsPretraining=None):
		self.datasets=datasets
		self.argdict=argdict
		self.splits=['train', 'dev', 'test']
		self.model=OptimusHomemade(argdict)
		self.device='cuda'

		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
		self.loss_function_basic = datasets['train'].loss_function
		self.step = 0
		self.epoch = 0
		if argdict['dataset'] in ['SST2']:
			self.loss_function_ppl = torch.nn.CrossEntropyLoss(reduction='mean')
		else:
			self.loss_function_ppl = self.loss_function_basic

	def kl_anneal_function(self, anneal_function, step, k, x0):
		if anneal_function == 'logistic':
			return float(1 / (1 + np.exp(-k * (step - x0))))
		elif anneal_function == 'linear':
			return min(1, step / x0)
		elif anneal_function == 'beta':
			return k

	def loss_fn(self, logp, target, mean, logv, anneal_function, step, k, x0, lamb):
		NLL_loss = self.loss_function_basic(logp, target)
		# KL Divergence
		#TODO Check that this is correct
		dimensionwise_loss = -0.5 * (1 + logv - mean ** 2 - logv.exp())
		dimensionwise_loss[dimensionwise_loss < lamb] = lamb
		KL_loss = dimensionwise_loss.sum()
		# KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
		KL_weight = self.kl_anneal_function(anneal_function, step, k, x0)

		return NLL_loss, KL_loss, KL_weight


	def run_batch(self, batch, params_anneal):
		batch_size = len(batch['sentence'])
		outputs, mean, logv, z = self.model(batch)
		logp = outputs['logits']
		# print(batch_size)
		target = outputs['encoded_output']['input_ids']
		# logp, target = self.datasets['train'].shape_for_loss_function(logp, batch['target'])
		# print(target)
		logp, target = self.datasets['train'].shape_for_loss_function(logp[:, :-1, :].contiguous(), target[:, 1:])
		NLL_loss, KL_loss, KL_weight = self.loss_fn(logp, target, mean, logv, params_anneal['strategy'], params_anneal['step'], params_anneal['k'], params_anneal['x0'], params_anneal['lamb'])
		# print(NLL_loss)
		# batch_size = logp.shape[0]
		# print(batch_size)
		loss = (NLL_loss + KL_weight * KL_loss) / batch_size
		# backward + optimization
		return loss, KL_loss, NLL_loss, KL_weight
		# Average_loss.append(loss.item().cpu())
		# Average_loss.append(loss.item())
		# Average_KL_Div.append(KL_loss.cpu().detach() / batch_size)
		# Average_NLL.append(NLL_loss.cpu().detach() / batch_size)


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
				logp = outputs['logits']
				# print(batch_size)
				target = outputs['encoded_output']['input_ids']
				# logp, target = self.datasets['train'].shape_for_loss_function(logp, batch['target'])
				# print(target)
				logp, target = self.datasets['train'].shape_for_loss_function(logp[:, :-1, :].contiguous(), target[:, 1:])
				NLL_loss, KL_loss, KL_weight = self.loss_fn(logp, target, mean, logv, 'logistic', self.step,0.0025)
				# print(NLL_loss)
				# batch_size = logp.shape[0]
				# print(batch_size)
				loss = (NLL_loss + KL_weight * KL_loss) / batch_size
				# backward + optimization
				if split == 'train':
					self.optimizer.zero_grad()
					loss.backward()
					self.optimizer.step()

				# Average_loss.append(loss.item().cpu())
				Average_loss.append(loss.item())
				Average_KL_Div.append(KL_loss.cpu().detach() / batch_size)
				Average_NLL.append(NLL_loss.cpu().detach() / batch_size)
			print(f"{split.upper()} Epoch {self.epoch}/{self.argdict['nb_epoch']}, Mean ELBO {np.mean(Average_loss)}, Mean NLL {np.mean(Average_NLL)}, Mean KL div {np.mean(Average_KL_Div)} KL Weight {KL_weight}")

	def train_model(self):
		#Pretraining
		for epoch in range(self.argdict['nb_epoch_pretraining']):
			#How many batches
			data_loader = DataLoader(
				dataset=self.datasets['train'],
				batch_size=self.argdict['batch_size'],
				shuffle= True,
				num_workers=cpu_count(),
				pin_memory=False
			)

			num_batches=len(data_loader)
			ratios=self.argdict['ratios']
			ratios=[math.floor(r*num_batches) for r in ratios]
			num_iter_anneal=ratios[1]
			#TODO there has to be a more eleguant way to do this lmao
			ratios[1]=ratios[0]+ratios[1]
			ratios[2]=ratios[2]+ratios[1]

			train_loss, train_KL, train_NLL=[], [], []
			dev_loss, dev_KL, dev_NLL=[], [], []

			for i, batch in enumerate(data_loader):
				if i<ratios[0]:
					#AE objective
					args_KL={'strategy': 'beta', 'k': 0, 'step':0, 'x0':0, 'lamb':5}
				elif i<ratios[1]:
					#annealing
					args_KL={'strategy': 'logistic', 'step':i-ratios[0] , 'k':1, 'x0':math.floor(num_iter_anneal)/2, 'lamb':5}
				else:
					#beta=1
					args_KL = {'strategy': 'beta', 'k': 1, 'step':0, 'x0':0, 'lamb':5}
				loss, KL_loss, NLL_loss, KL_weight=self.run_batch(batch, args_KL)
				train_loss.append(loss.detach().cpu())
				train_NLL.append(NLL_loss.detach().cpu())
				train_KL.append(KL_loss.detach().cpu())
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
			print(f"Pretraining Train Epoch {epoch}/{self.argdict['nb_epoch_pretraining']}, Mean ELBO {np.mean(train_loss)},"
				  f" Mean NLL {np.mean(train_NLL)}, Mean KL div {np.mean(train_KL)}")

			#dev
			data_loader = DataLoader(
				dataset=self.datasets['dev'],
				batch_size=self.argdict['batch_size'],
				shuffle=False,
				num_workers=cpu_count(),
				pin_memory=False
			)

			num_batches = len(data_loader)
			for i, batch in enumerate(data_loader):
				with torch.no_grad():
					args_KL = {'strategy': 'beta', 'k': 1, 'step': 0, 'x0': 0, 'lamb': 5}
					loss, KL_loss, NLL_loss, KL_weight = self.run_batch(batch, args_KL)
					dev_loss.append(loss.detach().cpu())
					dev_NLL.append(NLL_loss.detach().cpu())
					dev_KL.append(KL_loss.detach().cpu())
			print(f"Pretraining Dev Epoch {epoch}/{self.argdict['nb_epoch_pretraining']}, Mean ELBO {np.mean(dev_loss)},"
				  f" Mean NLL {np.mean(dev_NLL)}, Mean KL div {np.mean(dev_KL)}")


		#Fine Tuning
		for epoch in range(self.argdict['nb_epoch']):
			#How many batches
			data_loader = DataLoader(
				dataset=self.datasets['train'],
				batch_size=self.argdict['batch_size'],
				shuffle= True,
				num_workers=cpu_count(),
				pin_memory=False
			)

			num_batches=len(data_loader)
			ratios=self.argdict['ratios']
			ratios=[math.floor(r*num_batches) for r in ratios]
			num_iter_anneal=ratios[1]
			#TODO there has to be a more eleguant way to do this lmao
			ratios[1]=ratios[0]+ratios[1]
			ratios[2]=ratios[2]+ratios[1]

			train_loss, train_KL, train_NLL=[], [], []
			dev_loss, dev_KL, dev_NLL=[], [], []

			for i, batch in enumerate(data_loader):
				if i<ratios[0]:
					#AE objective
					args_KL={'strategy': 'beta', 'k': 0, 'step':0, 'x0':0, 'lamb':5}
				elif i<ratios[1]:
					#annealing
					args_KL={'strategy': 'logistic', 'step':i-ratios[0] , 'k':1, 'x0':math.floor(num_iter_anneal)/2, 'lamb':5}
				else:
					#beta=1
					args_KL = {'strategy': 'beta', 'k': 1, 'step':0, 'x0':0, 'lamb':5}
				loss, KL_loss, NLL_loss, KL_weight=self.run_batch(batch, args_KL)
				train_loss.append(loss.detach().cpu())
				train_NLL.append(NLL_loss.detach().cpu())
				train_KL.append(KL_loss.detach().cpu())
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
			print(f"training Train Epoch {epoch}/{self.argdict['nb_epoch_pretraining']}, Mean ELBO {np.mean(train_loss)},"
				  f" Mean NLL {np.mean(train_NLL)}, Mean KL div {np.mean(train_KL)}")

			#dev
			data_loader = DataLoader(
				dataset=self.datasets['dev'],
				batch_size=self.argdict['batch_size'],
				shuffle=False,
				num_workers=cpu_count(),
				pin_memory=False
			)

			num_batches = len(data_loader)
			for i, batch in enumerate(data_loader):
				with torch.no_grad():
					args_KL = {'strategy': 'beta', 'k': 1, 'step': 0, 'x0': 0, 'lamb': 5}
					loss, KL_loss, NLL_loss, KL_weight = self.run_batch(batch, args_KL)
					dev_loss.append(loss.detach().cpu())
					dev_NLL.append(NLL_loss.detach().cpu())
					dev_KL.append(KL_loss.detach().cpu())
			print(f"training Dev Epoch {epoch}/{self.argdict['nb_epoch_pretraining']}, Mean ELBO {np.mean(dev_loss)},"
				  f" Mean NLL {np.mean(dev_NLL)}, Mean KL div {np.mean(dev_KL)}")


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
			# args_KL = {'strategy': 'beta', 'k': 1, 'step': 0, 'x0': 0, 'lamb': 5}
			NLL_loss, KL_loss, KL_weight = self.loss_fn(logp, target.to('cuda'), mean, logv, 'beta', 0, 1, 0, 5)



			NLL_mean = self.loss_function_ppl(logp, target.to('cuda'))

			loss = (NLL_loss + KL_weight * KL_loss) / batch_size
			Average_loss.append(loss.item())
			Average_KL_Div.append(KL_loss.cpu().detach() / batch_size)
			Average_NLL.append(NLL_loss.cpu().detach() / batch_size)
			NLL_mean_for_ppl.append(NLL_mean.cpu().detach())
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

