# from transformers import (WEIGHTS_NAME, AdamW, WarmupLinearSchedule,
#                                   BertConfig, BertForLatentConnector, BertTokenizer,
#                                   GPT2Config, , GPT2Tokenizer,
#                                   OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
#                                   RobertaConfig, RobertaForMaskedLM, RobertaTokenizer)


from torch.utils.data import DataLoader
from multiprocessing import cpu_count
import numpy as np

class OptimusVAE():
	def __init__(self, argdict, train, dev, test):
		self.datasets={'train':train, 'dev':dev, 'test': test}
		self.argdict=argdict
		self.splits=['train', 'dev']
		decoder=GPT2ForLatentConnector(argdict)
		self.encoder=BertForLatentConnector(argdict)
		tokenizer_decoder=GPT2Tokenizer.from_pretrained('gpt2')
		self.tokenizer_encoder=BertTokenizer.from_pretrained('bert-base-uncased')
		self.model=Optimus(self.encoder, decoder, self.tokenizer_encoder, tokenizer_decoder, argdict)
		self.device='cuda'


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
				# encodings_input = self.tokenizer_encoder(batch['sentence'], return_tensors="pt", padding=True, truncation=True).to(self.device)
				# target = encodings_input['input_ids']
				# batch_size = target.shape[0]
				outputs=self.model(batch)
				print(outputs)
				fds
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

		self.generate_from_dataset()


