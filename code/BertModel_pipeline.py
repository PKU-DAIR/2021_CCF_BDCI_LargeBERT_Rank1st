import torch.nn as nn
import torch

def get_extended_attention_mask(attention_mask):
    # the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
    extended_attention_mask = attention_mask[:, None, None, :]
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask

# Use BertEmbeddings defined in bert_model
class BertEmbeddings_(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.embeddings = bert_model.embeddings
    def forward(self, inputs):
        input_ids, token_type_ids, attention_mask = inputs
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )
        return (embedding_output, attention_mask)

# Use BertEncoder defined in bert_model
class BertEncoder_(nn.Module):
    def __init__(self, bert_model, layer_idx_start, layer_idx_end):
        super().__init__()
        self.layer = bert_model.encoder.layer[layer_idx_start:layer_idx_end]
    def forward(self, inputs):
        hidden_states, attention_mask = inputs
        extended_attention_mask = get_extended_attention_mask(attention_mask)
        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=extended_attention_mask,
            )
            hidden_states = layer_outputs[0]
        return (hidden_states, attention_mask)

# Use BertPooler defined in bert_model
class BertPooler_(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.pooler = bert_model.pooler
    def forward(self, inputs):
        hidden_states, _ = inputs
        return (hidden_states, self.pooler(hidden_states))