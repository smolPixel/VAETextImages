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
		self.tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
		self.tokenizer.pad_token = self.tokenizer.eos_token
		return model, None

	def train(self):
		pass

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
			print(batch)
			fds
			encodings = tokenizer(btext, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
			# Forward pass
			logp, mean, logv, z = self.model(batch)
			batch_size = logp.shape[0]
			logp, target=self.datasets['train'].shape_for_loss_function(logp, batch['target'])
			NLL_loss, KL_loss= self.loss_fn(logp, target.to('cuda'),  mean, logv)

			loss = (NLL_loss +  KL_loss) / batch_size
			Average_loss.append(loss.item())
			Average_KL_Div.append(KL_loss.cpu().detach()/batch_size)
			Average_NLL.append(NLL_loss.cpu().detach()/batch_size)


		print(Average_NLL)
		return {'Mean ELBO': np.mean(Average_loss), 'Mean LF' :np.mean(Average_NLL), 'Mean KL div' :np.mean(Average_KL_Div), 'PPL': {torch.exp(torch.mean(torch.Tensor(Average_NLL)))}}