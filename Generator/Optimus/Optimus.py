# from transformers import (WEIGHTS_NAME, AdamW, WarmupLinearSchedule,
#                                   BertConfig, BertForLatentConnector, BertTokenizer,
#                                   GPT2Config, , GPT2Tokenizer,
#                                   OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
#                                   RobertaConfig, RobertaForMaskedLM, RobertaTokenizer)

from Generator.Optimus.GPTLatent import GPT2ForLatentConnector


class OptimusVAE():
	def __init__(self, argdict, train, dev, test):
		encoder=GPT2ForLatentConnector()