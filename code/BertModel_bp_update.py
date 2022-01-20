import torch.nn as nn
import torch
from transformers import AdamW, get_scheduler
from torch.autograd import Variable
import torch.autograd as autograd
from torch.optim import SGD


def get_extended_attention_mask(attention_mask):
    # the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
    extended_attention_mask = attention_mask[:, None, None, :]
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask

# Use BertEmbeddings defined in bert_model
class BertEmbeddings_(nn.Module):
    def __init__(self, bert_model, args):
        super().__init__()
        self.embeddings = bert_model.embeddings
        self.optimizer = AdamW(self.embeddings.parameters(), lr=args.lr,
                      weight_decay=args.adam_weight_decay)
        self.lr_scheduler = get_scheduler(
                            "polynomial",
                            optimizer=self.optimizer,
                            num_warmup_steps=0,
                            num_training_steps=300
                        )

    def optimizer_step(self):
        self.optimizer.step()
        self.lr_scheduler.step()

    def forward(self, inputs):
        input_ids, token_type_ids, attention_mask = inputs
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )
        self.output = embedding_output
        return (embedding_output, attention_mask)

    def grad_update(self, grad):
        # backward to compute tensor.grad
        autograd.backward(self.output, grad)
        del grad
        self.optimizer_step() # optimizer step
        # delete grad & activations & empty cache
        for val in self.parameters():
            del val.grad
        del self.output
        torch.cuda.empty_cache()

# Use BertEncoder defined in bert_model
class BertEncoder_(nn.Module):
    def __init__(self, bert_model, layer_idx_start, layer_idx_end, args):
        super().__init__()
        self.layer = bert_model.encoder.layer[layer_idx_start:layer_idx_end]
        self.optimizers = [AdamW(self.layer[i].parameters(), lr=args.lr,
                      weight_decay=args.adam_weight_decay) for i in range(layer_idx_end-layer_idx_start)]
        self.lr_schedulers = [get_scheduler(
                            "polynomial",
                            optimizer=self.optimizers[i],
                            num_warmup_steps=0,
                            num_training_steps=300
                        ) for i in range(layer_idx_end-layer_idx_start)]

    def optimizer_step(self, layer_id):
        self.optimizers[layer_id].step()
        self.lr_schedulers[layer_id].step()

    def forward(self, inputs):
        self.input, self.output = [], []
        hidden_states, attention_mask = inputs
        self.input_node = hidden_states
        extended_attention_mask = get_extended_attention_mask(attention_mask)

        for i, layer_module in enumerate(self.layer):
            hidden_states = Variable(hidden_states.data, requires_grad=True)
            self.input.append(hidden_states)
            if True:
                self_attention_outputs = layer_module.attention(
                    hidden_states=hidden_states,
                    attention_mask=extended_attention_mask,
                )
                attention_output = self_attention_outputs[0]
                mini_seq = 64
                hidden_states_list = []
                for i in range(int(512/mini_seq)):
                    intermediate_output = layer_module.intermediate(attention_output[:,i*mini_seq:(i+1)*mini_seq,:]) # [bsz, mini_seq, 4*hidden]
                    hidden_state = layer_module.output(intermediate_output, attention_output[:,i*mini_seq:(i+1)*mini_seq,:].contiguous()) # [bsz, mini_seq, hidden]
                    hidden_states_list.append(hidden_state) # [bsz, mini_seq, hidden]
                hidden_states = torch.cat(hidden_states_list, dim = 1) # [bsz, seq_len, hidden]
            else:
                layer_outputs = layer_module(
                    hidden_states=hidden_states,
                    attention_mask=extended_attention_mask,
                )
                hidden_states = layer_outputs[0]
            self.output.append(hidden_states)
        return (hidden_states, attention_mask)

    def grad_update(self, grad):
        for i, output in reversed(list(enumerate(self.output))):
            if i == (len(self.output) - 1):
                # backward to compute tensor.grad
                autograd.backward(output, grad)
                del grad
            else:
                autograd.backward(output, self.input[i+1].grad.data)
                del self.input[i+1].grad
                del self.input[i+1]
            self.optimizer_step(i) # optimizer step
            del output # delete forward activation
            # delete grad & activations & empty cache
            for val in self.layer[i].parameters():
                del val.grad
            torch.cuda.empty_cache()
        
        # pass grad of input node to previous layer
        return self.input[0].grad

# Use BertPooler defined in bert_model
class BertPooler_(nn.Module):
    def __init__(self, bert_model, args):
        super().__init__()
        self.pooler = bert_model.pooler
        self.optimizer = AdamW(self.pooler.parameters(), lr=args.lr,
                      weight_decay=args.adam_weight_decay)
        self.lr_scheduler = get_scheduler(
                            "polynomial",
                            optimizer=self.optimizer,
                            num_warmup_steps=0,
                            num_training_steps=300
                        )

    def optimizer_step(self):
        self.optimizer.step()
        self.lr_scheduler.step()

    def forward(self, inputs):
        hidden_states, _ = inputs
        # Split computing graph here, so loss.backward will stop here!
        hidden_states = Variable(hidden_states.data, requires_grad=True)
        self.input = hidden_states
        return (hidden_states, self.pooler(hidden_states))


def BertModel_grad_update(model, cls, loss, in_dev = None):
    loss.backward() # backward to Encoder output
    model[3].optimizer_step()
    for val in cls.parameters():
        del val.grad
    for val in model[3].parameters():
        del val.grad
    torch.cuda.empty_cache()
    grad = model[3].input.grad
    grad = model[2].grad_update(grad)
    if in_dev is not None:
        grad = grad.to(in_dev)
    grad = model[1].grad_update(grad)
    model[0].grad_update(grad)