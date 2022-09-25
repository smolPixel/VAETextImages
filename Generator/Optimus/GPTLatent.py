import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss




class GPT2ForLatentConnector():
	r"""
		**labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
			Labels for language modeling.
			Note that the labels **are shifted** inside the model, i.e. you can set ``lm_labels = input_ids``
			Indices are selected in ``[-1, 0, ..., config.vocab_size]``
			All labels set to ``-1`` are ignored (masked), the loss is only
			computed for labels in ``[0, ..., config.vocab_size]``

	Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
		**loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
			Language modeling loss.
		**prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
			Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
		**past**:
			list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
			that contains pre-computed hidden-states (key and values in the attention blocks).
			Can be used (see `past` input) to speed up sequential decoding.
		**hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
			list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
			of shape ``(batch_size, sequence_length, hidden_size)``:
			Hidden-states of the model at the output of each layer plus the initial embedding outputs.
		**attentions**: (`optional`, returned when ``config.output_attentions=True``)
			list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
			Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

	Examples::

		import torch
		from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel

		tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
		model = GPT2LMHeadModel.from_pretrained('gpt2')

		input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
		outputs = model(input_ids, labels=input_ids)
		loss, logits = outputs[:2]

	"""

	def __init__(self, argdict):# latent_as_gpt_emb=True, latent_as_gpt_memory=True):
		# super(GPT2ForLatentConnector, self).__init__(config)

		# self.transformer = GPT2Model(config)
		# self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

		# self.init_weights()
		# self.tie_weights()


		# self.latent_as_gpt_emb = latent_as_gpt_emb
		# self.latent_as_gpt_memory = latent_as_gpt_memory

	def tie_weights(self):
		""" Make sure we are sharing the input and output embeddings.
			Export to TorchScript can't handle parameter sharing so we are cloning them instead.
		"""
		self._tie_or_clone_weights(self.lm_head,
								   self.transformer.wte)

	def forward(self, input_ids, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
				labels=None, label_ignore=None):
		transformer_outputs = self.transformer(input_ids,
											   past=past,
											   attention_mask=attention_mask,
											   token_type_ids=token_type_ids,
											   position_ids=position_ids,
											   head_mask=head_mask,
											   latent_as_gpt_emb=self.latent_as_gpt_emb,
											   latent_as_gpt_memory=self.latent_as_gpt_memory)
		hidden_states = transformer_outputs[0]

		lm_logits = self.lm_head(hidden_states)

		outputs = (lm_logits,) + transformer_outputs[1:]
		if labels is not None:
			# Shift so that tokens < n predict n
			shift_logits = lm_logits[..., :-1, :].contiguous()
			shift_labels = labels[..., 1:].contiguous()
			# Flatten the tokens
			loss_fct = CrossEntropyLoss(ignore_index=label_ignore,
										reduce=False)  # 50258 is the padding id, otherwise -1 is used for masked LM.
			loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
							shift_labels.view(-1))
			loss = torch.sum(loss.view(-1, shift_labels.shape[-1]), -1)
			outputs = (loss,) + outputs

		return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)