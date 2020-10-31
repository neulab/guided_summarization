"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn
import numpy as np

from models.encoder import PositionalEncoding
from models.neural import MultiHeadedAttention, PositionwiseFeedForward, DecoderState

MAX_SIZE = 5000

COPY=False
REVERSE=False

class Z_TransformerDecoderLayer(nn.Module):
    """
    Args:
      d_model (int): the dimension of keys/values/queries in
                       MultiHeadedAttention, also the input size of
                       the first-layer of the PositionwiseFeedForward.
      heads (int): the number of heads for MultiHeadedAttention.
      d_ff (int): the second-layer of the PositionwiseFeedForward.
      dropout (float): dropout probability(0-1.0).
      self_attn_type (string): type of self-attention scaled-dot, average
    """

    def __init__(self, d_model, heads, d_ff, dropout):
        super(Z_TransformerDecoderLayer, self).__init__()


        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)

        self.z_context_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.context_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_z = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)
        mask = self._get_attn_subsequent_mask(MAX_SIZE)
        # Register self.mask as a buffer in TransformerDecoderLayer, so
        # it gets TransformerDecoderLayer's cuda behavior automatically.
        self.register_buffer('mask', mask)

    def forward(self, inputs, memory_bank, z_memory_bank, src_pad_mask, tgt_pad_mask, z_pad_mask,
                previous_input=None, layer_cache=None, step=None):
        """
        Args:
            inputs (`FloatTensor`): `[batch_size x 1 x model_dim]`
            memory_bank (`FloatTensor`): `[batch_size x src_len x model_dim]`
            src_pad_mask (`LongTensor`): `[batch_size x 1 x src_len]`
            tgt_pad_mask (`LongTensor`): `[batch_size x 1 x 1]`

        Returns:
            (`FloatTensor`, `FloatTensor`, `FloatTensor`):

            * output `[batch_size x 1 x model_dim]`
            * attn `[batch_size x 1 x src_len]`
            * all_input `[batch_size x current_step x model_dim]`

        """
        dec_mask = torch.gt(tgt_pad_mask +
                            self.mask[:, :tgt_pad_mask.size(1),
                                      :tgt_pad_mask.size(1)], 0)
        input_norm = self.layer_norm_1(inputs)
        all_input = input_norm
        if previous_input is not None:
            #print(previous_input.size())
            #print(input_norm.size())
            all_input = torch.cat((previous_input, input_norm), dim=1)
            dec_mask = None

        query, _ = self.self_attn(all_input, all_input, input_norm,
                                     mask=dec_mask,
                                     layer_cache=layer_cache,
                                     type="self")


        query = self.drop(query) + inputs

        if not REVERSE:
            query_norm = self.layer_norm_2(query)
            z_mid, attn = self.z_context_attn(z_memory_bank, z_memory_bank, query_norm,
                                          mask=z_pad_mask,
                                          layer_cache=layer_cache,
                                          type="z_context")

            z_query = self.drop(z_mid) + query
            z_query_norm = self.layer_norm_z(z_query)
            mid, z_attn = self.context_attn(memory_bank, memory_bank, z_query_norm,
                                          mask=src_pad_mask,
                                          layer_cache=layer_cache,
                                          type="context")
            


            output = self.feed_forward(self.drop(mid) + z_query)
        else:
            query_norm = self.layer_norm_z(query)
            z_mid, z_attn = self.context_attn(memory_bank, memory_bank, query_norm,
                                          mask=src_pad_mask,
                                          layer_cache=layer_cache,
                                          type="context")

            z_query = self.drop(z_mid) + query
            z_query_norm = self.layer_norm_2(z_query)
            mid, attn = self.z_context_attn(z_memory_bank, z_memory_bank, z_query_norm,
                                          mask=z_pad_mask,
                                          layer_cache=layer_cache,
                                          type="z_context")
            


            output = self.feed_forward(self.drop(mid) + z_query)

        return output, all_input, attn

    def _get_attn_subsequent_mask(self, size):
        """
        Get an attention mask to avoid using the subsequent info.

        Args:
            size: int

        Returns:
            (`LongTensor`):

            * subsequent_mask `[1 x size x size]`
        """
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        subsequent_mask = torch.from_numpy(subsequent_mask)
        return subsequent_mask



class Z_TransformerDecoder(nn.Module):
    """
    The Transformer decoder from "Attention is All You Need".


    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          BB[multi-head src-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> BB
          BB --> C
          C --> O


    Args:
       num_layers (int): number of encoder layers.
       d_model (int): size of the model
       heads (int): number of heads
       d_ff (int): size of the inner FF layer
       dropout (float): dropout parameters
       embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings
       attn_type (str): if using a seperate copy attention
    """

    def __init__(self, num_layers, d_model, heads, d_ff, dropout, embeddings, vocab_size):
        super(Z_TransformerDecoder, self).__init__()

        # Basic attributes.
        self.decoder_type = 'transformer'
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.pos_emb = PositionalEncoding(dropout,self.embeddings.embedding_dim)
        self.vocab_size = vocab_size

        if COPY:
            self.copy_attn = MultiHeadedAttention(
                1, d_model, dropout=dropout)

        # Build TransformerDecoder.
        self.transformer_layers = nn.ModuleList(
            [Z_TransformerDecoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_layers)])

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, tgt, memory_bank, z_memory_bank, state, memory_lengths=None, z_memory_lengths=None,
                step=None, cache=None,memory_masks=None, z_memory_masks=None):
        """
        See :obj:`onmt.modules.RNNDecoderBase.forward()`
        """

        src_words = state.src
        tgt_words = tgt
        src_batch, src_len = src_words.size()
        tgt_batch, tgt_len = tgt_words.size()


        z_words = state.z
        z_batch, z_len = z_words.size()

        # Run the forward pass of the TransformerDecoder.
        # emb = self.embeddings(tgt, step=step)
        emb = self.embeddings(tgt)
        assert emb.dim() == 3  # len x batch x embedding_dim

        output = self.pos_emb(emb, step)

        src_memory_bank = memory_bank
        padding_idx = self.embeddings.padding_idx
        tgt_pad_mask = tgt_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(tgt_batch, tgt_len, tgt_len)

        if (not memory_masks is None):
            src_len = memory_masks.size(-1)
            src_pad_mask = memory_masks.expand(src_batch, tgt_len, src_len)

        else:
            src_pad_mask = src_words.data.eq(padding_idx).unsqueeze(1) \
                .expand(src_batch, tgt_len, src_len)

        if (not z_memory_masks is None):
            z_len = z_memory_masks.size(-1)
            z_pad_mask = z_memory_masks.expand(z_batch, tgt_len, z_len)
        else:
            z_pad_mask = z_words.data.eq(padding_idx).unsqueeze(1) \
                .expand(z_batch, tgt_len, z_len)

        if state.cache is None:
            saved_inputs = []

        for i in range(self.num_layers):
            prev_layer_input = None
            if state.cache is None:
                if state.previous_input is not None:
                    prev_layer_input = state.previous_layer_inputs[i]
            output, all_input, attn \
                = self.transformer_layers[i](
                    output, src_memory_bank, z_memory_bank,
                    src_pad_mask, tgt_pad_mask, z_pad_mask,
                    previous_input=prev_layer_input,
                    layer_cache=state.cache["layer_{}".format(i)]
                    if state.cache is not None else None,
                    step=step)
            if state.cache is None:
                saved_inputs.append(all_input)


        output = self.layer_norm(output)

        # Process the result and update the attentions.


        copy_prob = None
        copy_state = None
        if COPY:
            layer_cache = None
            if step is not None:
                prev_layer_input = None
                if state.cache is None:
                    if state.previous_input is not None:
                        prev_layer_input = state.previous_layer_inputs[self.num_layers]
                if prev_layer_input is not None:
                    output = torch.cat((prev_layer_input, output), dim=1)
                if state.cache is None:
                    saved_inputs.append(output)
                layer_cache=state.cache["layer_{}".format(self.num_layers)] if state.cache is not None else None
            copy_state, attn = self.copy_attn(z_memory_bank, z_memory_bank, output,
                                          mask=z_pad_mask,
                                          layer_cache=layer_cache,
                                          type="z_context")
            #attn = torch.mean(attn, 1)
            attn = attn[:, 0, :, :]
            src_onehot = torch.zeros(z_words.size(0), z_words.size(1), self.vocab_size).cuda()
            z = z_words.unsqueeze(2)
            src_onehot.scatter_(2, z, 1)
            copy_prob = torch.bmm(attn, src_onehot)
            #copy_prob[:, :, 101] = 0
            #copy_prob[:, :, 102] = 0
            #copy_prob = copy_prob / torch.sum(copy_prob, -1, keepdim=True)

        if state.cache is None:
            saved_inputs = torch.stack(saved_inputs)

        if state.cache is None:
            state = state.update_state(tgt, saved_inputs)

        return output, state, [copy_prob, emb, copy_state]

    def init_decoder_state(self, src, memory_bank, z, z_memory_bank,
                           with_cache=False):
        """ Init decoder state """
        state = Z_TransformerDecoderState(src, z)
        if with_cache:
            state._init_cache(memory_bank, self.num_layers)
        return state



class Z_TransformerDecoderState(DecoderState):
    """ Transformer Decoder state base class """

    def __init__(self, src, z):
        """
        Args:
            src (FloatTensor): a sequence of source words tensors
                    with optional feature tensors, of size (len x batch).
        """
        self.src = src
        self.z = z
        self.previous_input = None
        self.previous_layer_inputs = None
        self.cache = None

    @property
    def _all(self):
        """
        Contains attributes that need to be updated in self.beam_update().
        """
        if (self.previous_input is not None
                and self.previous_layer_inputs is not None):
            return (self.previous_input,
                    self.previous_layer_inputs,
                    self.src, self.z)
        else:
            return (self.src, self.z)

    def detach(self):
        if self.previous_input is not None:
            self.previous_input = self.previous_input.detach()
        if self.previous_layer_inputs is not None:
            self.previous_layer_inputs = self.previous_layer_inputs.detach()
        self.src = self.src.detach()
        self.z = self.z.detach()

    def update_state(self, new_input, previous_layer_inputs):
        state = Z_TransformerDecoderState(self.src, self.z)
        state.previous_input = new_input
        state.previous_layer_inputs = previous_layer_inputs
        return state

    def _init_cache(self, memory_bank, num_layers):
        self.cache = {}

        for l in range(num_layers):
            layer_cache = {
                "memory_keys": None,
                "memory_values": None,
                "z_memory_keys": None,
                "z_memory_values": None
            }
            layer_cache["self_keys"] = None
            layer_cache["self_values"] = None
            self.cache["layer_{}".format(l)] = layer_cache
        if COPY:
            layer_cache = {
                "memory_keys": None,
                "memory_values": None,
                "z_memory_keys": None,
                "z_memory_values": None
            }
            layer_cache["self_keys"] = None
            layer_cache["self_values"] = None
            self.cache["layer_{}".format(num_layers)] = layer_cache


    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        self.src = self.src.data.repeat(1, beam_size, 1)
        self.z = self.z.data.repeat(1, beam_size, 1)

    def map_batch_fn(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        self.src = fn(self.src, 0)
        self.z = fn(self.z, 0)
        if self.cache is not None:
            _recursive_map(self.cache)
        if self.previous_input is not None:
            self.previous_input = fn(self.previous_input, 0)
        if self.previous_layer_inputs is not None:
            self.previous_layer_inputs = fn(self.previous_layer_inputs, 1)




