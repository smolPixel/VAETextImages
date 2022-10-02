from transformers import GPT2PreTrainedModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import (
	BaseModelOutputWithPastAndCrossAttentions,
	CausalLMOutputWithCrossAttentions,
	SequenceClassifierOutputWithPast,
	TokenClassifierOutput,
)

from transformers.generation_logits_process import LogitsProcessorList
from transformers.generation_stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)


class GPT2ModelLatent(GPT2PreTrainedModel):
	_keys_to_ignore_on_load_missing = ["attn.masked_bias"]

	def __init__(self, config, argdict):
		super().__init__(config)

		self.embed_dim = config.hidden_size

		self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
		self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
		self.latent_to_embed=nn.Linear(argdict['latent_size'], self.embed_dim)
		self.strategies=argdict['strategy']

		self.drop = nn.Dropout(config.embd_pdrop)
		self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
		self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

		# Model parallel
		self.model_parallel = False
		self.device_map = None
		self.gradient_checkpointing = False

		self.lm_head=nn.Linear(config.n_embd, config.vocab_size, bias=False)

		# Initialize weights and apply final processing
		self.post_init()

	def parallelize(self, device_map=None):
		# Check validity of device_map
		self.device_map = (
			get_device_map(len(self.h), range(torch.cuda.device_count())) if device_map is None else device_map
		)
		assert_device_map(self.device_map, len(self.h))
		self.model_parallel = True
		self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
		self.last_device = "cuda:" + str(max(self.device_map.keys()))
		self.wte = self.wte.to(self.first_device)
		self.wpe = self.wpe.to(self.first_device)
		# Load onto devices
		for k, v in self.device_map.items():
			for block in v:
				cuda_device = "cuda:" + str(k)
				self.h[block] = self.h[block].to(cuda_device)
		# ln_f to last
		self.ln_f = self.ln_f.to(self.last_device)

	def deparallelize(self):
		self.model_parallel = False
		self.device_map = None
		self.first_device = "cpu"
		self.last_device = "cpu"
		self.wte = self.wte.to("cpu")
		self.wpe = self.wpe.to("cpu")
		for index in range(len(self.h)):
			self.h[index] = self.h[index].to("cpu")
		self.ln_f = self.ln_f.to("cpu")
		torch.cuda.empty_cache()

	def get_input_embeddings(self):
		return self.wte

	def set_input_embeddings(self, new_embeddings):
		self.wte = new_embeddings

	def _prune_heads(self, heads_to_prune):
		"""
		Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
		"""
		for layer, heads in heads_to_prune.items():
			self.h[layer].attn.prune_heads(heads)

	# @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
	# @add_code_sample_docstrings(
	#     processor_class=_TOKENIZER_FOR_DOC,
	#     checkpoint=_CHECKPOINT_FOR_DOC,
	#     output_type=BaseModelOutputWithPastAndCrossAttentions,
	#     config_class=_CONFIG_FOR_DOC,
	# )

	def get_output_embeddings(self):
		return self.lm_head

	def set_output_embeddings(self, new_embeddings):
		self.lm_head = new_embeddings

	def prepare_inputs_for_generation(self, input_ids, z, past=None, **kwargs):
		token_type_ids = kwargs.get("token_type_ids", None)
		# only last token for inputs_ids if past is defined in kwargs
		if past:
			input_ids = input_ids[:, -1].unsqueeze(-1)
			if token_type_ids is not None:
				token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

		attention_mask = kwargs.get("attention_mask", None)
		position_ids = kwargs.get("position_ids", None)

		if attention_mask is not None and position_ids is None:
			# create position_ids on the fly for batch generation
			position_ids = attention_mask.long().cumsum(-1) - 1
			position_ids.masked_fill_(attention_mask == 0, 1)
			if past:
				position_ids = position_ids[:, -1].unsqueeze(-1)
		else:
			position_ids = None
		return {
			"input_ids": input_ids,
			"z": z,
			"past_key_values": past,
			"use_cache": kwargs.get("use_cache"),
			"position_ids": position_ids,
			"attention_mask": attention_mask,
			"token_type_ids": token_type_ids,
		}

	def forward(
		self,
		input_ids: Optional[torch.LongTensor] = None,
		z = None,
		past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
		attention_mask: Optional[torch.FloatTensor] = None,
		token_type_ids: Optional[torch.LongTensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		head_mask: Optional[torch.FloatTensor] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		encoder_hidden_states: Optional[torch.Tensor] = None,
		encoder_attention_mask: Optional[torch.FloatTensor] = None,
		use_cache: Optional[bool] = None,
			labels: Optional[torch.LongTensor] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
	) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
		output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
		output_hidden_states = (
			output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
		)
		use_cache = use_cache if use_cache is not None else self.config.use_cache
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		if input_ids is not None and inputs_embeds is not None:
			raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
		elif input_ids is not None:
			input_shape = input_ids.size()
			input_ids = input_ids.view(-1, input_shape[-1])
			batch_size = input_ids.shape[0]
		elif inputs_embeds is not None:
			input_shape = inputs_embeds.size()[:-1]
			batch_size = inputs_embeds.shape[0]
		else:
			raise ValueError("You have to specify either input_ids or inputs_embeds")

		device = input_ids.device if input_ids is not None else inputs_embeds.device

		if token_type_ids is not None:
			token_type_ids = token_type_ids.view(-1, input_shape[-1])
		if position_ids is not None:
			position_ids = position_ids.view(-1, input_shape[-1])

		if past_key_values is None:
			past_length = 0
			past_key_values = tuple([None] * len(self.h))
		else:
			past_length = past_key_values[0][0].size(-2)
		if position_ids is None:
			position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
			position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])


		# GPT2Attention mask.
		if attention_mask is not None:
			if batch_size <= 0:
				raise ValueError("batch_size has to be defined and > 0")
			attention_mask = attention_mask.view(batch_size, -1)
			# We create a 3D attention mask from a 2D tensor mask.
			# Sizes are [batch_size, 1, 1, to_seq_length]
			# So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
			# this attention mask is more simple than the triangular masking of causal attention
			# used in OpenAI GPT, we just need to prepare the broadcast dimension here.
			attention_mask = attention_mask[:, None, None, :]

			# Since attention_mask is 1.0 for positions we want to attend and 0.0 for
			# masked positions, this operation will create a tensor which is 0.0 for
			# positions we want to attend and -10000.0 for masked positions.
			# Since we are adding it to the raw scores before the softmax, this is
			# effectively the same as removing these entirely.
			attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
			attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

		# If a 2D or 3D attention mask is provided for the cross-attention
		# we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
		if self.config.add_cross_attention and encoder_hidden_states is not None:
			encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
			encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
			if encoder_attention_mask is None:
				encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
			encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
		else:
			encoder_attention_mask = None

		# Prepare head mask if needed
		# 1.0 in head_mask indicate we keep the head
		# attention_probs has shape bsz x n_heads x N x N
		# head_mask has shape n_layer x batch x n_heads x N x N
		head_mask = self.get_head_mask(head_mask, self.config.n_layer)


		if inputs_embeds is None:
			inputs_embeds = self.wte(input_ids)
		position_embeds = self.wpe(position_ids)
		hidden_states = inputs_embeds + position_embeds
		if 'Embedding' in self.strategies:
			z=self.latent_to_embed(z).unsqueeze(1).repeat(1, hidden_states.shape[1], 1)
			hidden_states=hidden_states+z

		if token_type_ids is not None:
			token_type_embeds = self.wte(token_type_ids)
			hidden_states = hidden_states + token_type_embeds

		hidden_states = self.drop(hidden_states)

		output_shape = input_shape + (hidden_states.size(-1),)

		presents = () if use_cache else None
		all_self_attentions = () if output_attentions else None
		all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
		all_hidden_states = () if output_hidden_states else None
		for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

			# Model parallel
			if self.model_parallel:
				torch.cuda.set_device(hidden_states.device)
				# Ensure layer_past is on same device as hidden_states (might not be correct)
				if layer_past is not None:
					layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
				# Ensure that attention_mask is always on the same device as hidden_states
				if attention_mask is not None:
					attention_mask = attention_mask.to(hidden_states.device)
				if isinstance(head_mask, torch.Tensor):
					head_mask = head_mask.to(hidden_states.device)
			if output_hidden_states:
				all_hidden_states = all_hidden_states + (hidden_states,)

			if self.gradient_checkpointing and self.training:

				if use_cache:
					logger.warning(
						"`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
					)
					use_cache = False

				def create_custom_forward(module):
					def custom_forward(*inputs):
						# None for past_key_value
						return module(*inputs, use_cache, output_attentions)

					return custom_forward

				outputs = torch.utils.checkpoint.checkpoint(
					create_custom_forward(block),
					hidden_states,
					None,
					attention_mask,
					head_mask[i],
					encoder_hidden_states,
					encoder_attention_mask,
				)
			else:
				outputs = block(
					hidden_states,
					layer_past=layer_past,
					attention_mask=attention_mask,
					head_mask=head_mask[i],
					encoder_hidden_states=encoder_hidden_states,
					encoder_attention_mask=encoder_attention_mask,
					use_cache=use_cache,
					output_attentions=output_attentions,
				)

			hidden_states = outputs[0]
			if use_cache is True:
				presents = presents + (outputs[1],)

			if output_attentions:
				all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
				if self.config.add_cross_attention:
					all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

			# Model Parallel: If it's the last layer for that device, put things on the next device
			if self.model_parallel:
				for k, v in self.device_map.items():
					if i == v[-1] and "cuda:" + str(k) != self.last_device:
						hidden_states = hidden_states.to("cuda:" + str(k + 1))

		hidden_states = self.ln_f(hidden_states)

		hidden_states = hidden_states.view(output_shape)
		# Add last hidden state
		if output_hidden_states:
			all_hidden_states = all_hidden_states + (hidden_states,)

		lm_logits = self.lm_head(hidden_states)

		loss = None
		if labels is not None:
			# Shift so that tokens < n predict n
			shift_logits = lm_logits[..., :-1, :].contiguous()
			shift_labels = labels[..., 1:].contiguous()
			# Flatten the tokens
			loss_fct = CrossEntropyLoss()
			loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

		if not return_dict:
			output = (lm_logits,) + transformer_outputs[1:]
			return ((loss,) + output) if loss is not None else output

		return CausalLMOutputWithCrossAttentions(
			loss=loss,
			logits=lm_logits,
			past_key_values=presents,
			hidden_states=all_hidden_states,
			attentions=all_self_attentions,
			cross_attentions=all_cross_attentions,
		)


	def generate(self, input_ids, z, max_length=None, model_kwargs=None):
		logits_processor = LogitsProcessorList()
		stopping_criteria = StoppingCriteriaList()
		if max_length is not None:
			warnings.warn(
				"`max_length` is deprecated in this function, use"
				" `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
				UserWarning,
			)
			stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)


		bos_token_id = self.config.bos_token_id
		pad_token_id = self.config.pad_token_id
		eos_token_id = None
		scores= None


		while True:
			# prepare model inputs
			model_inputs = self.prepare_inputs_for_generation(input_ids, z)

			# forward pass to get next token
			outputs = self(
				**model_inputs,
				return_dict=True,
				output_attentions=False,
				output_hidden_states=False,
			)


			next_token_logits = outputs.logits[:, -1, :]

			# pre-process distribution
			next_tokens_scores = logits_processor(input_ids, next_token_logits)
			# argmax
			next_tokens = torch.argmax(next_tokens_scores, dim=-1)


			# finished sentences should have their next token be a padding token
			if eos_token_id is not None:
				if pad_token_id is None:
					raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
				next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

			# update generated ids, model inputs, and length for next step
			input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
			model_kwargs = self._update_model_kwargs_for_generation(
				outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
			)
			cur_len = cur_len + 1

			# if eos_token was found in one sentence, set sentence to finished
			if eos_token_id is not None:
				unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

			# stop when each sentence is finished, or if we exceed the maximum length
			if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
				break
				# if not synced_gpus:
				# 	break
				# else:
				# 	this_peer_finished = True

		# if not return_dict:
		# 	return tuple(
		# 		v
		# 		for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
		# 		if v is not None
		# 	)
		#
		#
		#
		# return BaseModelOutputWithPastAndCrossAttentions(
		# 	last_hidden_state=hidden_states,
		# 	past_key_values=presents,
		# 	hidden_states=all_hidden_states,
		# 	attentions=all_self_attentions,
		# 	cross_attentions=all_cross_attentions,
		# )
