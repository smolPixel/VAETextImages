import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
# from metrics import calc_all, calc_batch_mi
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from torch import optim
from transformers import (
	AdamW,
	T5Config,
	T5ForConditionalGeneration,
	get_linear_schedule_with_warmup,
	T5TokenizerFast
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
		self.config = T5Config.from_pretrained(argdict['base_model'])
		self.t5 = ModifiedT5ForConditionalGeneration.from_pretrained(
			argdict['base_model'],
			config=self.config,
			argdict=argdict
		)
		self.tokenizer = T5TokenizerFast(argdict['base_model'])
		self.argdict=argdict
		self.splits=['train', 'dev', 'test']
		self.datasets={'train':train, 'dev':dev, 'test':test}
		self.latent_dim = self.argdict['latent_size']
		self.decoder_unfreeze_step = None
		self.min_z = self.argdict['lambda']
		# self.fixed_reg_weight = fixed_reg_weight
		# self.denoise_percentage = denoise_percentage

	def freeze_decoder(self):
		for param in self.t5.memory_projection.parameters():
			param.requires_grad = False
		for param in self.t5.decoder.parameters():
			param.requires_grad = False
		for param in self.t5.lm_head.parameters():
			param.requires_grad = False

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
		encoder_inputs, encoder_masks, decoder_targets = batch

		if training and self.denoise_percentage:
			for i, (inp, msk) in enumerate(zip(encoder_inputs, encoder_masks)):
				token_length = (msk.sum() - 1).item()
				max_drop = int(token_length * self.denoise_percentage)
				if max_drop > 1:
					drop_count = torch.randint(max_drop, size=(1,)).item()
				else:
					drop_count = 0
				drop_index = torch.randperm(token_length)[:drop_count]
				inp = torch.tensor(
					[t for n, t in enumerate(inp) if n not in drop_index]
				)
				msk = torch.tensor(
					[t for n, t in enumerate(msk) if n not in drop_index]
				)
				inp = torch.cat(
					(inp, torch.tensor([self.tokenizer.pad_token_id] * drop_count))
				)
				msk = torch.cat((msk, torch.tensor([0] * drop_count)))
				encoder_inputs[i] = msk
				encoder_masks[i] = inp

		batch_size = encoder_inputs.shape[0]

		x, z, mu, logvar = self(
			encoder_inputs,
			encoder_masks,
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
		recon_loss, reg_loss = self.run_batch(batch, batch_idx, training=True)
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
		tokenized= self.tokenizer(batch[self.field_input], padding=True, truncation=True)
		recon_loss, reg_loss = self.run_batch(tokenized, batch_idx)
		loss = recon_loss + reg_loss
		# mi = calc_batch_mi(self, batch)
		self.log("val_recon_loss", recon_loss)
		self.log("val_reg_loss", reg_loss)
		self.log("val_loss", loss)
		# self.log("finished_epoch", self.current_epoch)
		return loss

	def validation_epoch_end(self, outputs):
		ppl, nll, elbo, rec, kl, mi, au = calc_all(self, self.val_dataloader())
		self.log("val_ppl", ppl)
		self.log("val_nll", nll)
		self.log("val_elbo", elbo)
		self.log("val_rec", rec)
		self.log("val_kl", kl)
		self.log("val_mi", mi)
		self.log("val_au", au)

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
			callbacks=[early_stop_callback, checkpoint_callback],
			max_epochs=15
		)

		trainer.fit(self, self.datasets['train'], self.datasets['dev'])

		model = model_class.load_from_checkpoint(
			checkpoint_callback.best_model_path,
			strict=False,
			tokenizer=tokenizer,
			iterations_per_training_epoch=iterations_per_training_epoch,
			latent_dim=conf.latent_dim,
			pooling_strategy=conf.pooling_strategy,
			min_z=conf.min_z,
			fixed_reg_weight=None,
			denoise_percentage=conf.denoise_percentage,
			base_model=conf.base_model_name,
		)

		print(
			"Finished preliminary encoder training.",
			f"Checkpoint saved at: {checkpoint_callback.best_model_path}",
		)
		#Phase 2 full fine tuning
