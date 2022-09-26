# from transformers import (WEIGHTS_NAME, AdamW, WarmupLinearSchedule,
#                                   BertConfig, BertForLatentConnector, BertTokenizer,
#                                   GPT2Config, , GPT2Tokenizer,
#                                   OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
#                                   RobertaConfig, RobertaForMaskedLM, RobertaTokenizer)


from Generator.Optimus.model import Optimus
from Generator.Optimus.GPTLatent import GPT2ForLatentConnector
from Generator.Optimus.BertLatent import BertForLatentConnector
from transformers import GPT2Tokenizer, BertTokenizer

class OptimusVAE():
	def __init__(self, argdict, train, dev, test):
		decoder=GPT2ForLatentConnector(argdict)
		encoder=BertForLatentConnector(argdict)
		tokenizer_decoder=GPT2Tokenizer.from_pretrained('gpt2')
		tokenizer_encoder=BertTokenizer.from_pretrained('bert-base-uncased')
		self.model=Optimus(encoder, decoder, tokenizer_encoder, tokenizer_decoder, argdict)


