import torch.nn as nn
import torch


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(30522, 768, padding_idx=0)
        self.position_embeddings = nn.Embedding(512, 768)
        self.token_type_embeddings = nn.Embedding(2, 768)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = torch.nn.LayerNorm(768, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertLayer(nn.Module):
    def __init__(self):
        super(BertLayer, self).__init__()
        self.attention = BertAttention()
        self.intermediate = BertIntermediate()
        self.output = BertOutput()

    def forward(self, hidden_states, attention_mask, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs

class BertEncoder(nn.Module):
    def __init__(self):
        super(BertEncoder, self).__init__()
        self.output_attentions = True
        self.output_hidden_states = True
        self.layer = nn.ModuleList([BertLayer() for _ in range(12)])

    def forward(self, hidden_states, attention_mask, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

class BertForLatentConnector(nn.Module):
	r"""
	Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
		**last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
			Sequence of hidden-states at the output of the last layer of the model.
		**pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
			Last layer hidden-state of the first token of the sequence (classification token)
			further processed by a Linear layer and a Tanh activation function. The Linear
			layer weights are trained from the next sentence prediction (classification)
			objective during Bert pretraining. This output is usually *not* a good summary
			of the semantic content of the input, you're often better with averaging or pooling
			the sequence of hidden-states for the whole input sequence.
		**hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
			list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
			of shape ``(batch_size, sequence_length, hidden_size)``:
			Hidden-states of the model at the output of each layer plus the initial embedding outputs.
		**attentions**: (`optional`, returned when ``config.output_attentions=True``)
			list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
			Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

	Examples::

		tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		model = BertModel.from_pretrained('bert-base-uncased')
		input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
		outputs = model(input_ids)
		last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

	"""
	def __init__(self, argdict):
		super(BertForLatentConnector, self).__init__()
		self.argdict=argdict
		self.embeddings = BertEmbeddings().to(argdict['device'])
		self.encoder = BertEncoder()
		# self.pooler = BertPooler(config)
		#
		# self.linear = nn.Linear(config.hidden_size, 2 * latent_size, bias=False)
		#
		# self.init_weights()

	def _resize_token_embeddings(self, new_num_tokens):
		old_embeddings = self.embeddings.word_embeddings
		new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
		self.embeddings.word_embeddings = new_embeddings
		return self.embeddings.word_embeddings

	def _prune_heads(self, heads_to_prune):
		""" Prunes heads of the model.
			heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
			See base class PreTrainedModel
		"""
		for layer, heads in heads_to_prune.items():
			self.encoder.layer[layer].attention.prune_heads(heads)

	def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
		if attention_mask is None:
			attention_mask = torch.ones_like(input_ids)
		if token_type_ids is None:
			token_type_ids = torch.zeros_like(input_ids)

		# We create a 3D attention mask from a 2D tensor mask.
		# Sizes are [batch_size, 1, 1, to_seq_length]
		# So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
		# this attention mask is more simple than the triangular masking of causal attention
		# used in OpenAI GPT, we just need to prepare the broadcast dimension here.
		extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

		# Since attention_mask is 1.0 for positions we want to attend and 0.0 for
		# masked positions, this operation will create a tensor which is 0.0 for
		# positions we want to attend and -10000.0 for masked positions.
		# Since we are adding it to the raw scores before the softmax, this is
		# effectively the same as removing these entirely.
		extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
		extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0


		# Prepare head mask if needed
		# 1.0 in head_mask indicate we keep the head
		# attention_probs has shape bsz x n_heads x N x N
		# input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
		# and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
		if head_mask is not None:
			if head_mask.dim() == 1:
				head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
				head_mask = head_mask.expand(12, -1, -1, -1, -1)
			elif head_mask.dim() == 2:
				head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
			head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
		else:
			head_mask = [None] * 12

		embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
		encoder_outputs = self.encoder(embedding_output,
									   extended_attention_mask,
									   head_mask=head_mask)
		sequence_output = encoder_outputs[0]
		pooled_output = self.pooler(sequence_output)

		outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
		return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)