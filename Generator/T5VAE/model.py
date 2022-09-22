import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)


class ModifiedT5ForConditionalGeneration(T5ForConditionalGeneration):
    def __init__(self, config, argdict):
        super().__init__(config)
        self.latent_dim = argdict['latent_size']
        self.mu = nn.Linear(config.d_model, self.latent_dim, bias=False)
        self.logvar = nn.Linear(config.d_model, self.latent_dim, bias=False)
        self.embed_size_per_head = config.d_model // config.num_heads
        self.memory_projection = nn.Linear(
            self.latent_dim,
            config.num_decoder_layers * config.num_heads * self.embed_size_per_head,
            bias=False,
        )
        self.pooling_strategy = argdict['pooling_strategy']

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sampled_z=None,
    ):

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        z, mu, logvar = None, None, None
        if sampled_z is not None:
            z = sampled_z
            encoder_outputs = BaseModelOutput(
                last_hidden_state=None,
                hidden_states=None,
                attentions=None,
            )
        elif encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.run_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            # print('bish')
            pooled = self.pool(encoder_outputs.hidden_states)
            z, mu, logvar = self.calculate_latent(pooled)
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if (
            labels is not None
            and decoder_input_ids is None
            and decoder_inputs_embeds is None
        ):
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        if past_key_values is None:
            past_key_values = self.build_past(z)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None and labels is None:
            # assert (
            #    labels is None
            # ), "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            # hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(
                    self.decoder.first_device
                )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            # loss_fct = CrossEntropyLoss(ignore_index=-100)
            # loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
            pass

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        out = Seq2SeqLMOutput(
            # loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
        out.mu = mu
        out.logvar = logvar
        out.z = z
        return out

    def encode(self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sampled_z=None,
    ):
        # print(self.encoder(input_ids, attention_mask))
        # print(self.run_encoder(input_ids, attention_mask))
        # use_cache = use_cache if use_cache is not None else self.config.use_cache
        # return_dict = (
        #     return_dict if return_dict is not None else self.config.use_return_dict
        # )
        #
        # # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        # if head_mask is not None and decoder_head_mask is None:
        #     if self.config.num_layers == self.config.num_decoder_layers:
        #         warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
        #         decoder_head_mask = head_mask
        #
        # # Encode if needed (training, first prediction pass)
        # z, mu, logvar = None, None, None
        # if sampled_z is not None:
        #     z = sampled_z
        #     encoder_outputs = BaseModelOutput(
        #         last_hidden_state=None,
        #         hidden_states=None,
        #         attentions=None,
        #     )
        # elif encoder_outputs is None:
        #     # Convert encoder inputs in embeddings if needed
        encoder_outputs = self.run_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # inputs_embeds=inputs_embeds,
            # head_mask=head_mask,
            # output_attentions=output_attentions,
            output_hidden_states=True,
            # return_dict=return_dict,
        )
        pooled = self.pool(encoder_outputs.hidden_states)
        z, mu, logvar = self.calculate_latent(pooled)
        return z, mu, logvar

    def inference(self, z, bos_token):
        bs=z.shape[0]
        device='cpu'
        gend=torch.zeros((bs, 500))

        for i in range(bs):
            generated = torch.tensor([bos_token for i in range(bs)]).unsqueeze(0).to(device)
            sampled_z=z[i].unsqueeze(0).to(device)
            # print(torch.zeros((1, z.shape[1])).normal_(mean=0, std=1).shape)
            # z=z[0].unsqueeze(0)
            # print(z.shape)
            # fds


            output, encoder_outputs = None, None
            while generated.shape[1] < 500:

                # decoder_inputs = self.t5.prepare_inputs_for_generation(generated, past=past)

                sampled_z = z[0]

                with torch.no_grad():
                    output = self.forward(
                        input_ids=None,
                        attention_mask=None,
                        # attention_mask=torch.ones((generated.shape[0], generated.shape[1] + 1)),
                        # encoder_hidden_states=None, #new_encoder_hidden_states,  # Modified.
                        # encoder_attention_mask=None, #new_attention_mask,  # Modified.
                        # attention_mask=encoder_mask,
                        encoder_outputs=None,
                        decoder_input_ids=generated[:, -1].unsqueeze(0),
                        # encoder_hidden_states=encoder_outputs[0],  # Modified.
                        # encoder_attention_mask=attention_mask,  # Modified.
                        # head_mask=kwargs.get("decoder_head_mask"),
                        # cross_attn_head_mask=kwargs.get("cross_attn_head_mask"),
                        past_key_values= None,
                        # inputs_embeds=decoder_inputs_embeds,
                        use_cache=True,
                        # output_attentions=output_attentions,
                        output_hidden_states=True,
                        return_dict=True,
                        sampled_z=sampled_z#torch.zeros((1, z.shape[1])).normal_(mean=0, std=1)#sampled_z,
                    )

                # print(output)
                # temperature = kwargs.get("temperature") if "temperature" in kwargs else 1.0
                # top_k = kwargs.get("top_k") if "top_k" in kwargs else 0
                # top_p = kwargs.get("top_p") if "top_p" in kwargs else 0

                logits = output.logits[0, -1, :] #/ temperature
                # filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                # print(logits.shape)
                # probabilities = F.softmax(filtered_logits, dim=-1)
                # next_token_id = torch.multinomial(probabilities, 1)
                next_token_id = torch.argmax(logits, dim=-1).unsqueeze(0)

                generated = torch.cat((generated, next_token_id.unsqueeze(0)), dim=1)
                past = output.past_key_values
                encoder_outputs = BaseModelOutput(
                    last_hidden_state=output.encoder_last_hidden_state,
                    hidden_states=output.encoder_hidden_states,
                    attentions=output.encoder_attentions,
                )
                # if next_token_id == model.tokenizer.eos_token_id:
                #     break
            gend[i]=generated
        return gend, z

    def run_encoder(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):



        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # print(input_ids)
        # print(encoder_outputs)
        # print(self.encoder(input_ids, attention_mask))
        # fds


        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        return encoder_outputs

    def pool(self, x):
        # Shape of x - (layer_count, batch_size, seq_length, hidden_size)
        x = torch.stack(x[1:])
        x = x.transpose(0, 1)
        if self.pooling_strategy == "mean":
            return x[:, -1, :, :].mean(dim=1)
        elif self.pooling_strategy == "max":
            return torch.max(x[:, -1, :, :], dim=1)[0]  # Pool from last layer.
        else:
            raise Exception("Wrong pooling strategy!")

    def calculate_latent(self, pooled):
        mu, logvar = self.mu(pooled), self.logvar(pooled)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def build_past(self, z):
        projection = self.memory_projection(z)
        cross_attn = projection.reshape(
            self.config.num_decoder_layers,
            projection.shape[0],
            self.config.num_heads,
            1,
            self.embed_size_per_head,
        )
        past_key_values = tuple((ca, ca) for ca in cross_attn)
        return past_key_values

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
