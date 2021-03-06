import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from torch.nn import Parameter
import torch.nn.init as init
from fused_ops import softmax_dropout, bias_dropout_add, bias_fast_gelu


class SoftmaxDropout(nn.Module):
    def __init__(self, p = 0.1):
        super(SoftmaxDropout, self).__init__()
        self.p = p
    def forward(self, x):
        return softmax_dropout(x, self.p, True)
    def extra_repr(self):
        return 'p={}'.format(self.p)

class BiasDropoutAdd(nn.Module):
    def __init__(self, p = 0.1):
        super(BiasDropoutAdd, self).__init__()
        self.p = p
    def forward(self, x, b, r):
        return bias_dropout_add(x, b, r, self.p, True)
    def extra_repr(self):
        return 'p={}'.format(self.p)

class BiasGelu(nn.Module):
    def __init__(self):
        super(BiasGelu, self).__init__()
    def forward(self, x, bias):
        return bias_fast_gelu(x, bias)

class BertSelfAttention_fused(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.seq_len = config.max_position_embeddings

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.softmax_dropout = SoftmaxDropout(config.attention_probs_dropout_prob)

        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states) # [batch_size, seq_len, hidden_size]

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states)) # [batch_size, num_heads, seq_len, head_size]
            value_layer = self.transpose_for_scores(self.value(hidden_states)) # [batch_size, num_heads, seq_len, head_size]

        query_layer = self.transpose_for_scores(mixed_query_layer) # [batch_size, num_heads, seq_len, head_size]

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        #attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores /= math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # # Normalize the attention scores to probabilities.
        # attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # # This is actually dropping out entire tokens to attend to, which might
        # # seem a bit unusual, but is taken from the original Transformer paper.
        # attention_probs = self.dropout(attention_probs)

        attention_scores = attention_scores.view(-1, self.seq_len, self.seq_len)
        attention_probs = self.softmax_dropout(attention_scores)
        attention_probs = attention_probs.view(-1, self.num_attention_heads, self.seq_len, self.seq_len)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class BertSelfAttention_fused_sequential(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.seq_len = config.max_position_embeddings

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.softmax_dropout = SoftmaxDropout(config.attention_probs_dropout_prob)

        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_key_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = torch.reshape(x, new_x_shape)
        return x.permute(0, 2, 3, 1)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states) # [batch_size, seq_len, hidden_size]

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_key_for_scores(self.key(hidden_states)) # [batch_size, num_heads, head_size, seq_len]
            value_layer = self.transpose_for_scores(self.value(hidden_states)) # [batch_size, num_heads, seq_len, head_size]

        query_layer = self.transpose_for_scores(mixed_query_layer) # [batch_size, num_heads, seq_len, head_size]

        context_layers=[]

        attention_mask = torch.squeeze(attention_mask, 1)

        for i in range(self.num_attention_heads):
            # Take the dot product between "query" and "key" to get the raw attention scores.
            attention_scores = torch.matmul(query_layer[:,i,:,:], key_layer[:,i,:,:]) # [batch_size, seq_len, seq_len]

            attention_scores /= math.sqrt(self.attention_head_size)
            if attention_mask is not None:
                # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
                attention_scores = attention_scores + attention_mask

            attention_probs = self.softmax_dropout(attention_scores)

            # Mask heads if we want to
            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            context_layers.append(torch.matmul(attention_probs, value_layer[:,i,:,:]).unsqueeze(1)) # [batch_size, seq_len, head_size]

        context_layer = torch.cat(context_layers, dim = 1)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # [batch_size, seq_len, num_heads, head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape) # [batch_size, seq_len, hidden_size]

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class BertSelfOutput_fused(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias = False)
        self.bias = nn.Parameter(torch.Tensor(config.hidden_size))
        self.bias_dropout_add = BiasDropoutAdd(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.dense.weight.size(1))
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.bias_dropout_add(hidden_states, self.bias, input_tensor)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BertOutput_fused(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size, bias = False)
        self.bias = nn.Parameter(torch.Tensor(config.hidden_size))
        self.bias_dropout_add = BiasDropoutAdd(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.dense.weight.size(1))
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.bias_dropout_add(hidden_states, self.bias, input_tensor)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BertIntermediate_fused(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size, bias = False)
        self.bias = nn.Parameter(torch.Tensor(config.intermediate_size))
        self.intermediate_act_fn = BiasGelu()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.dense.weight.size(1))
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states, self.bias)
        return hidden_states

