import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
# from metrics import calc_all, calc_batch_mi
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from pytorch_lightning.plugins import DDPPlugin
from multiprocessing import cpu_count
from metrics import calc_mi, calc_au

from sklearn.svm import LinearSVC

from torch import optim
from transformers import (
	AdamW,
	T5Config,
	T5ForConditionalGeneration,
	get_linear_schedule_with_warmup,
	T5Tokenizer
)
# from transformers.generation_stopping_criteria import (
#     MaxLengthCriteria,
#     StoppingCriteriaList,
# )
from Generator.T5VAE.model import ModifiedT5ForConditionalGeneration

# logging.getLogger("transformers").setLevel(logging.ERROR)

# self,
# tokenizer,
# iterations_per_training_epoch,
# latent_dim,
# pooling_strategy,
# min_z = None,
# fixed_reg_weight = None,
# denoise_percentage = 0,
# base_model = "t5-base",
# ):
# super().__init__()
# self.config = T5Config.from_pretrained(base_model)
# self.t5 = ModifiedT5ForConditionalGeneration.from_pretrained(
# base_model,
# config = self.config,
# latent_dim = latent_dim,
# pooling_strategy = pooling_strategy,
class T5VAE(LightningModule):
	def __init__(self, argdict, train, dev, test):
		super().__init__()
		self.config = T5Config.from_pretrained(argdict['base_model'], add_special_tokens=True)
		self.t5 = ModifiedT5ForConditionalGeneration.from_pretrained(
			argdict['base_model'],
			config=self.config,
			argdict=argdict
		)
		self.tokenizer = T5Tokenizer.from_pretrained(argdict['base_model'], add_special_tokens=True)
		self.tokenizer.add_special_tokens({'bos_token': '[EOS]'})

		self.resize_token_embeddings(len(tokenizer))
		print(self.tokenizer.bos_token_id)
		fds


		self.argdict=argdict
		self.splits=['train', 'dev', 'test']
		self.datasets={'train':train, 'dev':dev, 'test':test}
		self.latent_dim = self.argdict['latent_size']
		self.decoder_unfreeze_step = None
		self.min_z = self.argdict['lambda']
		self.denoise_percentage = self.argdict['denoise_percentage']
		self.fixed_reg_weight = self.argdict['fixed_reg_weight']
		# self.fixed_reg_weight = fixed_reg_weight
		# self.denoise_percentage = denoise_percentage
		self.loss_function_basic = train.loss_function
		if argdict['dataset'] in ['SST2']:
			self.loss_function_ppl = torch.nn.CrossEntropyLoss(ignore_index=train.pad_idx, reduction='mean')
		else:
			self.loss_function_ppl = self.loss_function_basic

	def freeze_decoder(self):
		for param in self.t5.memory_projection.parameters():
			param.requires_grad = False
		for param in self.t5.decoder.parameters():
			param.requires_grad = False
		for param in self.t5.lm_head.parameters():
			param.requires_grad = False

	def unfreeze_decoder(self):
		for param in self.t5.memory_projection.parameters():
			param.requires_grad = True
		for param in self.t5.decoder.parameters():
			param.requires_grad = True
		for param in self.t5.lm_head.parameters():
			param.requires_grad = True

	def loss_fn(self, logp, target, mean, logv):
		# NLL = torch.nn.NLLLoss(ignore_index=self.datasets['train'].pad_idx, reduction='sum')
		# cut-off unnecessary padding from target, and flatten
		# target = target[:, :torch.max(length).item()].contiguous().view(-1)
		# target = target.contiguous().view(-1)
		# logp = logp.view(-1, logp.size(2))

		# Negative Log Likelihood
		NLL_loss = self.loss_function_basic(logp, target)
		# BCE = torch.nn.functional.binary_cross_entropy(logp, target.view(-1, 784), reduction='sum')
		# KL Divergence
		KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())

		# KL_weight = self.kl_anneal_function(anneal_function, step, k, self.dataset_length*self.argdict['x0'])

		return NLL_loss, KL_loss

	def forward(self, encoder_input, encoder_mask, labels, **kwargs):
		output = self.t5(
			input_ids=encoder_input,
			attention_mask=encoder_mask,
			labels=labels,
			output_hidden_states=True,
			**kwargs
		)
		return (output.logits, output.z, output.mu, output.logvar)

	#####
	# Torch lightning
	#####

	def run_batch(self, batch, batch_idx, training=False):
		encoder_inputs, encoder_masks=batch['input_ids'].to(self.device), batch['attention_mask'].to(self.device)
		decoder_targets=batch['input_ids'].to(self.device)

		# if training and self.denoise_percentage:
		# 	for i, (inp, msk) in enumerate(zip(encoder_inputs, encoder_masks)):
		# 		#for each sentence
		# 		#Length= nb of 1 minus 1
		# 		token_length = (msk.sum() - 1).item()
		# 		max_drop = int(token_length * self.denoise_percentage)
		# 		#Max drop = number we are dropping for this sentence
		# 		if max_drop > 1:
		# 			drop_count = torch.randint(max_drop, size=(1,)).item()
		# 		else:
		# 			drop_count = 0
		# 		drop_index = torch.randperm(token_length)[:drop_count]
		# 		inp = torch.tensor(
		# 			[t for n, t in enumerate(inp) if n not in drop_index]
		# 		)
		# 		msk = torch.tensor(
		# 			[t for n, t in enumerate(msk) if n not in drop_index]
		# 		)
		# 		inp = torch.cat(
		# 			(inp, torch.tensor([self.tokenizer.pad_token_id] * drop_count))
		# 		)
		# 		msk = torch.cat((msk, torch.tensor([0] * drop_count)))
		# 		encoder_inputs[i] = inp
		# 		encoder_masks[i] = msk

		batch_size = encoder_inputs.shape[0]

		x, z, mu, logvar = self(
			encoder_input=encoder_inputs,
			encoder_mask=encoder_masks,
			labels=decoder_targets,
		)

		recon_loss = self.reconstruction_loss(x, decoder_targets)
		reg_loss = self.regularization_loss(mu, logvar, training)

		return recon_loss.mean(), reg_loss.mean()

	def kld_weight(self, start=0.0, stop=1, n_cycle=1, ratio=1, linear_ratio=1):
		if self.fixed_reg_weight is not None:
			return self.fixed_reg_weight
		# cycle_size = self.iterations_per_training_epoch // n_cycle
		cycle_size = self.iterations_per_training_epoch * 100
		vae_steps = int(cycle_size * ratio)
		ae_steps = cycle_size - vae_steps
		linear_steps = int(vae_steps * linear_ratio)  # 25%
		full_steps = cycle_size - ae_steps - linear_steps  # 25%
		step = self.global_step % cycle_size
		if step <= ae_steps:
			return 0
		vae_step = step - ae_steps
		weight = (
			vae_step / linear_steps * (stop - start)
			if vae_step <= linear_steps
			else stop
		)
		return weight

	def training_step(self, batch, batch_idx):
		tokenized = self.tokenizer(batch['sentence'], padding=True, truncation=True, return_tensors='pt')
		recon_loss, reg_loss = self.run_batch(tokenized, batch_idx, training=True)
		reg_weight = self.kld_weight()
		loss = recon_loss + reg_weight * reg_loss
		self.log("train_reg_weight", reg_weight)
		self.log("train_recon_loss", recon_loss)
		self.log("train_reg_loss", reg_weight * reg_loss)
		self.log("train_unweighted_reg_loss", reg_loss)
		self.log("train_loss", loss)
		return loss

	def training_epoch_end(self, outputs):
		# if self.current_epoch == 2:
		# self.decoder_unfreeze_step = self.global_step
		# for param in self.t5.decoder.parameters():
		#    param.requires_grad = True
		# for param in self.t5.lm_head.parameters():
		# param.requires_grad = True
		self.log("finished_epoch", self.current_epoch)
		return

	def validation_step(self, batch, batch_idx):




		tokenized= self.tokenizer(batch['sentence'], padding=True, truncation=True, return_tensors='pt')
		recon_loss, reg_loss = self.run_batch(tokenized, batch_idx)
		loss = recon_loss + reg_loss
		# mi = calc_batch_mi(self, batch)
		self.log("val_recon_loss", recon_loss)
		self.log("val_reg_loss", reg_loss)
		self.log("val_loss", loss)
		# self.log("finished_epoch", self.current_epoch)
		return loss

	def validation_epoch_end(self, outputs):
		pass
		# ppl, nll, elbo, rec, kl, mi, au = calc_all(self, self.val_dataloader())
		# self.log("val_ppl", ppl)
		# self.log("val_nll", nll)
		# self.log("val_elbo", elbo)
		# self.log("val_rec", rec)
		# self.log("val_kl", kl)
		# self.log("val_mi", mi)
		# self.log("val_au", au)

	def test_step(self, batch, batch_idx):
		recon_loss, reg_loss, _ = self.run_batch(batch, batch_idx)
		loss = recon_loss + reg_loss
		self.log("test_loss", recon_loss)
		self.log("test_reg_loss", reg_loss)
		self.log("test_loss", loss)
		self.log("finished_epoch", self.current_epoch)
		return loss

	# https://github.com/PyTorchLightning/pytorch-lightning/issues/3095

	def configure_optimizers(self):
		optimizer = torch.optim.AdamW(self.t5.parameters(), lr=1e-4)
		# optimizer = optim.SGD(self.t5.parameters(), lr=0.01, momentum=0.9)
		# scheduler = get_linear_schedule_with_warmup(
		#    optimizer,
		#    num_warmup_steps=5000,
		# num_warmup_steps=200,
		#    num_training_steps=130000,
		# num_training_steps=2400,
		# )
		# return [optimizer], [scheduler]
		return optimizer

	def reconstruction_loss(self, x, target):
		loss = F.cross_entropy(
			x.transpose(1, 2),
			target,
			ignore_index=self.tokenizer.pad_token_id,
			reduction="none",
		)
		return loss

	def regularization_loss(self, mu, logvar, training=False):
		dimensionwise_loss = -0.5 * (1 + logvar - mu ** 2 - logvar.exp())
		#KL thresholding, 2.2.4
		if self.min_z and training:
			dimensionwise_loss[dimensionwise_loss < self.min_z] = self.min_z
		loss = dimensionwise_loss.sum(-1)
		return loss

	def train_model(self):
		#Phase 1 encoder training
		self.freeze_decoder()


		early_stop_callback = EarlyStopping(
			monitor="val_recon_loss",
			min_delta=0.001,
			patience=15,
			verbose=True,
			mode="min",
			strict=True,
		)

		checkpoint_callback = ModelCheckpoint(
			monitor="val_recon_loss",
			mode="min",
			save_weights_only=True,
			save_top_k=15,
		)

		trainer = pl.Trainer(
			gpus=1,
			callbacks=[early_stop_callback, checkpoint_callback],
			max_epochs= self.argdict['nb_epoch_pretraining']#15
		)

		train_loader = DataLoader(
			dataset=self.datasets['train'],
			batch_size=64,  # self.argdict.batch_size,
			shuffle=True,
		)

		dev_loader = DataLoader(
			dataset=self.datasets['dev'],
			batch_size=64,  # self.argdict.batch_size,
			shuffle=True,
		)


		trainer.fit(self, train_loader, dev_loader)

		self = T5VAE.load_from_checkpoint(
			checkpoint_callback.best_model_path,
			argdict=self.argdict,
			train=self.datasets['train'],
			dev=self.datasets['dev'],
			test=self.datasets['test']
		)

		print(
			"Finished preliminary encoder training.",
			f"Checkpoint saved at: {checkpoint_callback.best_model_path}",
		)
		# Phase 2 full fine tuning
		self.unfreeze_decoder()

		# Run regular training.
		early_stop_callback = EarlyStopping(
			# monitor="val_loss",
			monitor="finished_epoch",
			min_delta=0.00,
			patience=10,
			verbose=True,
			mode="min",
			strict=True,
		)

		checkpoint_callback = ModelCheckpoint(
			monitor="finished_epoch",
			mode="max",
			save_weights_only=True,
			save_top_k=10,
		)

		trainer = pl.Trainer(
			gpus=-1,
			callbacks=[early_stop_callback, checkpoint_callback],
			max_epochs= self.argdict['nb_epoch']#10,
			# plugins=DDPPlugin(
			# 	find_unused_parameters=True
			# ),  # We ignore params from cross-attention.
		)

		train_loader = DataLoader(
			dataset=self.datasets['train'],
			batch_size=64,  # self.argdict.batch_size,
			shuffle=True,
		)

		dev_loader = DataLoader(
			dataset=self.datasets['dev'],
			batch_size=64,  # self.argdict.batch_size,
			shuffle=True,
		)

		trainer.fit(self, train_loader, dev_loader)
		self.interpolate()

	def interpolate(self, n=5):
		p0 = torch.randn([1, self.argdict['latent_size']])
		p1 = torch.randn([1, self.argdict['latent_size']])
		points = torch.zeros(n, self.argdict['latent_size'])
		points[0] = p0
		points[n - 1] = p1
		for i in range(n):
			ratio = i / n
			px = (1 - ratio) * p0 + ratio * p1
			points[i] = px
		points = points.cuda()
		print(self.tokenizer.bos_token_id)
		print(points.shape)
		samples, z = self.t5.inference(n=n, z=points)
		print(samples)
		self.datasets['train'].process_generated(samples)

	def encode(self):
		with torch.no_grad():
			dico = {}
			for split in self.splits:
				data_loader = DataLoader(
					dataset=self.datasets[split],
					batch_size=64,  # self.argdict.batch_size,
					shuffle=False,
					num_workers=cpu_count(),
					pin_memory=torch.cuda.is_available()
				)
				# Enable/Disable Dropout

				self.t5.eval()
				# print(f"The dataset length is {len(data_loader.dataset)}")
				dataset = torch.zeros(len(data_loader.dataset), self.argdict['latent_size'])
				labels = torch.zeros(len(data_loader.dataset))
				sentences = []
				counter = 0
				for iteration, batch in enumerate(data_loader):
					# print("Oh la la banana")
					batch_size = batch['input'].size(0)
					tokenized = self.tokenizer(batch['sentence'], padding=True, truncation=True, return_tensors='pt')

					encoder_inputs, encoder_masks = tokenized['input_ids'].to(self.device), tokenized['attention_mask'].to(self.device)
					decoder_targets = tokenized['input_ids'].to(self.device)
					#
					# print(batch['input'])
					# print(batch['input'].shape)
					# print(encoder_inputs)
					# print(encoder_masks)
					# self.t5.forward(encoder_inputs, encoder_masks, labels=decoder_targets)
					# fds
					z, _, _ = self.t5.encode(encoder_inputs, encoder_masks)
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
			batch_size=64,  # self.argdict.batch_size,
			shuffle=False,
			num_workers=cpu_count(),
			pin_memory=torch.cuda.is_available()
		)

		self.t5.eval()

		Average_loss = []
		Average_NLL = []
		Average_KL_Div = []
		MIs = []
		mus = []
		NLL_mean_for_ppl = []
		for iteration, batch in enumerate(data_loader):
			# Forward pass
			# logp, mean, logv, z = self.t5(batch)
			# continue
			tokenized = self.tokenizer(batch['sentence'], padding=True, truncation=True, return_tensors='pt')

			encoder_inputs, encoder_masks = tokenized['input_ids'].to(self.device), tokenized['attention_mask'].to(self.device)
			decoder_targets = tokenized['input_ids'].to(self.device)

			logp, z, mean, logv = self(
				encoder_inputs,
				encoder_masks,
				labels=decoder_targets,
			)

			# print(logp.shape)
			# print(decoder_targets.shape)

			# Keeping track of the means for AU
			mus.append(mean.detach().squeeze(0))
			batch_size = logp.shape[0]
			logp, target = self.datasets['train'].shape_for_loss_function(logp, decoder_targets)
			NLL_loss, KL_loss = self.loss_fn(logp, target.to('cuda'), mean, logv)

			NLL_mean = self.loss_function_ppl(logp, target.to('cuda'))

			loss = (NLL_loss + KL_loss) / batch_size
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

		# Reconstruction
		# This part should be handled in the dataset, as it is heavily dataset dependant (duh)
		self.datasets['train'].test_reconstruction(self.model)

		# print(AU)
		return {'Mean ELBO': np.mean(Average_loss), 'Mean LF': np.mean(Average_NLL),
				'Mean KL div': np.mean(Average_KL_Div), 'PPL': {torch.exp(torch.mean(torch.Tensor(NLL_mean_for_ppl)))},
				'Separability': sep, 'MI': {np.mean(MIs)}, 'Active Units': AU[0]}
