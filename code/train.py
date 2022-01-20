from abc import ABC

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel, BertConfig, AdamW, get_scheduler
import argparse
from tqdm import tqdm
import numpy as np
import fairscale
import random

# Read and Concatenate all data (2*1024=2048 sequences in total)
def read_npy_data(args):
    npy = open("./code/data.npy", "rb")
    input_ids, token_type_ids, attention_mask, label, masked_lm_ids, masked_lm_positions, masked_lm_weights = [], [], [], [], [], [], []
    for i in range(args.dataset_size):
        data = np.load(npy, allow_pickle=True)
        input_ids.append(data[0]["input_ids"])
        token_type_ids.append(data[0]["token_type_ids"])
        attention_mask.append(data[0]["attention_mask"])
        label.append(data[1]) 
        masked_lm_ids.append(data[2]) 
        masked_lm_positions.append(data[3])
        masked_lm_weights.append(data[4])
    def concat(ls):
        return np.concatenate(np.array(ls),axis=0)
    input_ids, token_type_ids, attention_mask, label, masked_lm_ids, masked_lm_positions, masked_lm_weights = \
        concat(input_ids), concat(token_type_ids), concat(attention_mask), concat(label), \
            concat(masked_lm_ids), concat(masked_lm_positions), concat(masked_lm_weights)
    return input_ids, token_type_ids, attention_mask, label, masked_lm_ids, masked_lm_positions, masked_lm_weights

class DataNumpyDataset(Dataset):
    def __init__(self, args):
        # __getitem__ return two sequences once, batch_size == 2
        self.dataset_size = args.dataset_size
        self.input_ids, self.token_type_ids, self.attention_mask, self.label, \
            self.masked_lm_ids, self.masked_lm_positions, self.masked_lm_weights = read_npy_data(args)

        if args.sparse_embedding:
            shape = self.input_ids.shape
            s = set()
            for i in range(shape[0]):
                for j in range(shape[1]):
                    s.add(self.input_ids[i][j])
            used_id = np.array(list(s))
            new_vocab_size = len(used_id)
            ids_map = np.zeros((args.vocab_size,),dtype = int)
            for i, id in enumerate(used_id):
                ids_map[id] = i
            for i in range(shape[0]):
                for j in range(shape[1]):
                    self.input_ids[i][j] = ids_map[self.input_ids[i][j]]
            args.new_vocab_size = new_vocab_size
            args.used_id = used_id

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if idx >= self.dataset_size:
            raise IndexError
        pt_data = dict()
        pt_data["input_ids"] = torch.tensor(self.input_ids[idx*2:idx*2+2]).long().to(device_in)
        pt_data["token_type_ids"] = torch.tensor(self.token_type_ids[idx*2:idx*2+2]).long().to(device_in)
        pt_data["attention_mask"] = torch.tensor(self.attention_mask[idx*2:idx*2+2]).long().to(device_in)
        label = torch.tensor(self.label[idx*2:idx*2+2]).long().to(device_out)
        masked_lm_ids = torch.tensor(self.masked_lm_ids[idx*2:idx*2+2]).long().to(device_out)
        masked_lm_positions = torch.tensor(self.masked_lm_positions[idx*2:idx*2+2]).long().to(device_out)
        masked_lm_weights = torch.tensor(self.masked_lm_weights[idx*2:idx*2+2]).long().to(device_out)
        return pt_data, label, masked_lm_ids, masked_lm_positions, masked_lm_weights

class BertPreTrainingHeads(nn.Module):
    def __init__(self, hidden_size, vocab_size, hidden_act=nn.GELU()):
        super().__init__()
        self.predictions = BertLMPredictionHead(
            hidden_size, vocab_size, hidden_act)
        self.seq_relationship = nn.Linear(hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_scores = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_scores


class BertLMPredictionHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, hidden_act=nn.GELU()):
        super().__init__()
        self.transform = BertPredictionHeadTransform(hidden_size, hidden_act)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)

        self.output_bias = nn.Parameter(torch.zeros(vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.output_bias

    def forward(self, sequence_output):
        sequence_output = self.transform(sequence_output)
        sequence_output = self.decoder(sequence_output)
        return sequence_output

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, hidden_size, hidden_act=nn.GELU()):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = hidden_act
        self.LayerNorm = nn.LayerNorm(hidden_size)

    def forward(self, sequence_output):
        sequence_output = self.dense(sequence_output)
        sequence_output = self.transform_act_fn(sequence_output)
        sequence_output = self.LayerNorm(sequence_output)
        return sequence_output


class PreTrainer(object):
    def __init__(self, max_predictions_per_seq, *args, **kwargs):
        mlm_criterion = nn.CrossEntropyLoss(reduction="none")
        self.max_predictions_per_seq = max_predictions_per_seq

        def get_masked_lm_loss(
                logit_blob,
                masked_lm_positions,
                masked_lm_labels,
                label_weights,
                max_predictions_per_seq,
                return_intermediate = False,
        ):
            # gather valid position indices
            logit_blob = torch.gather(
                logit_blob,
                index=masked_lm_positions.unsqueeze(2).to(
                    dtype=torch.int64).repeat(1, 1, 30522),
                dim=1,
            )
            logit_blob = torch.reshape(logit_blob, [-1, 30522])
            label_id_blob = torch.reshape(masked_lm_labels, [-1])

            # The `positions` tensor might be zero-padded (if the sequence is too
            # short to have the maximum number of predictions). The `label_weights`
            # tensor has a value of 1.0 for every real prediction and 0.0 for the
            # padding predictions.
            pre_example_loss = mlm_criterion(logit_blob, label_id_blob.long())
            pre_example_loss = torch.reshape(
                pre_example_loss, [-1, max_predictions_per_seq])
            sum_label_weight = torch.sum(label_weights, dim=-1)
            # sum_label_weight = sum_label_weight / label_weights.shape[0]
            sum_label_weight = torch.true_divide(sum_label_weight, label_weights.shape[0])
            numerator = torch.sum(pre_example_loss * label_weights)
            denominator = torch.sum(label_weights) + 1e-5
            loss = numerator / denominator
            if return_intermediate:
                return numerator, denominator
            return loss

        self.ns_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.masked_lm_criterion = get_masked_lm_loss

    def compute_loss(self, model, cls, labels, id, pos, weight, inputs, return_outputs=False, return_intermediate=False):
        inputs = (inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask'])
        outputs = model(inputs)
        hidden_states, pooler_output = outputs
        prediction_scores, seq_relationship_scores = cls(hidden_states, pooler_output)
        next_sentence_loss = self.ns_criterion(
            seq_relationship_scores.view(-1, 2), labels.long().view(-1)
        )

        # return numerator, denominator instead of masked_lm_loss
        if return_intermediate:
            numerator, denominator = self.masked_lm_criterion(
                prediction_scores, pos, id, weight, max_predictions_per_seq=self.max_predictions_per_seq,
                return_intermediate=return_intermediate)
            return next_sentence_loss, numerator, denominator

        masked_lm_loss = self.masked_lm_criterion(
            prediction_scores, pos, id, weight, max_predictions_per_seq=self.max_predictions_per_seq
        )

        total_loss = next_sentence_loss + masked_lm_loss

        return (total_loss, outputs) if return_outputs else total_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ofrecord_path",
        type=str,
        default="wiki_ofrecord_seq_len_128_example",
        help="Path to ofrecord dataset",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=768, help="Hidden size of transformer model",
    )
    parser.add_argument(
        "--dataset_size", type=int, default=1024, help="The number of samples in an epoch cycle",
    )
    parser.add_argument(
        "--num_hidden_layers", type=int, default=12, help="Number of layers"
    )
    parser.add_argument(
        "-a",
        "--num_attention_heads",
        type=int,
        default=12,
        help="Number of attention heads",
    )
    parser.add_argument(
        "-s", "--seq_length", type=int, default=128, help="Maximum sequence len"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=30522, help="Total number of vocab"
    )
    parser.add_argument("--max_predictions_per_seq", type=int, default=20)
    parser.add_argument("-e", "--epochs", type=int,
                        default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate of adam")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=0.01, help="Weight_decay of adam"
    )

    # Add arguments to turn on or off pipeline
    parser.add_argument(
        "--pipeline_on", type=int, default=1, help="Pipeline on or not."
    )
    parser.add_argument(
        "--check_correctness", type=int, default=0, help="Whether to check model correctness."
    )
    parser.add_argument(
        "--fuse_ops", type=int, default=0, help="Whether to fuse ops."
    )
    parser.add_argument(
        "--load_params", type=int, default=0, help="Whether to load saved init params."
    )
    parser.add_argument(
        "--accum_gradient", type=int, default=0, help="Whether to accumulate gradients."
    )
    parser.add_argument(
        "--profile", type=int, default=0, help="Whether to profile model GPU memory."
    )
    parser.add_argument(
        "--sparse_embedding", type=int, default=0, help="Whether to use sparse word embeddings."
    )
    parser.add_argument(
        "--grad_update", type=int, default=0, help="Whether to fuse computing grad and update."
    )

    args = parser.parse_args()

    if args.grad_update and args.accum_gradient:
        assert(False)

    if args.check_correctness:
        seed = 123
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    print("Creating Dataloader")
    dataset = DataNumpyDataset(args)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir = './cached_tokenizer/')
    
    configuration = BertConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               num_attention_heads=args.num_attention_heads, intermediate_size=4*args.hidden_size)

    bert_model = BertModel(configuration)

    # Sparse Embedding table
    if args.sparse_embedding:
        setattr(bert_model.embeddings, 'word_embeddings', nn.Embedding(args.new_vocab_size, configuration.hidden_size))

    if args.load_params:
        params = torch.load("init_params.file")

    if args.fuse_ops:
        # fuse softmax & dropout
        from BertModel_fused import BertSelfAttention_fused, BertSelfAttention_fused_sequential
        for i in range(args.num_hidden_layers):
            setattr(bert_model.encoder.layer[i].attention, "self", BertSelfAttention_fused_sequential(configuration))
            # setattr(bert_model.encoder.layer[i].attention, "self", BertSelfAttention_fused(configuration))

        # fuse bias dropout add
        from BertModel_fused import BertSelfOutput_fused, BertOutput_fused
        for i in range(args.num_hidden_layers):
            setattr(bert_model.encoder.layer[i].attention, "output", BertSelfOutput_fused(configuration))
            setattr(bert_model.encoder.layer[i], "output", BertOutput_fused(configuration))
            if args.load_params:
                params['encoder.layer.%d.attention.output.bias'%i] = params['encoder.layer.%d.attention.output.dense.bias'%i]
                del params['encoder.layer.%d.attention.output.dense.bias'%i]
                params['encoder.layer.%d.output.bias'%i] = params['encoder.layer.%d.output.dense.bias'%i]
                del params['encoder.layer.%d.output.dense.bias'%i]


    if args.load_params:
        if args.sparse_embedding:
            params['embeddings.word_embeddings.weight'] = params['embeddings.word_embeddings.weight'][args.used_id]
        bert_model.load_state_dict(params)

    split_idx = int(args.num_hidden_layers/2)

    if args.grad_update:
        from BertModel_bp_update import BertEmbeddings_, BertEncoder_, BertPooler_, BertModel_grad_update
        model = nn.Sequential(
                        BertEmbeddings_(bert_model, args),
                        BertEncoder_(bert_model, 0, split_idx, args),
                        BertEncoder_(bert_model, split_idx, args.num_hidden_layers, args),
                        BertPooler_(bert_model, args)
                    )
    else:
        from BertModel_pipeline import BertEmbeddings_, BertEncoder_, BertPooler_, BertEncoder_front_half, BertEncoder_back_half
        model = nn.Sequential(
                        BertEmbeddings_(bert_model),
                        BertEncoder_(bert_model, 0, split_idx),
                        BertEncoder_(bert_model, split_idx, args.num_hidden_layers),
                        BertPooler_(bert_model)
                    )

    if args.pipeline_on:
        device_in = torch.device('cuda:0')
        device_out = torch.device('cuda:1')

        chunks = 2
        if args.accum_gradient or args.grad_update:
            chunks = 1
        model = fairscale.nn.Pipe(model, balance = [2, 2], chunks = chunks, devices = [device_in, device_out], checkpoint = 'never')

    else:
        device_in = torch.device('cuda:0')
        device_out = torch.device('cuda:0')
        model = model.to(device_in)

    if not args.grad_update:
        optimizer = AdamW(model.parameters(), lr=args.lr,
                        weight_decay=args.adam_weight_decay)
        lr_scheduler = get_scheduler(
            "polynomial",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=300
        )
    cls = BertPreTrainingHeads(args.hidden_size, args.vocab_size)
    if args.load_params:
        cls.load_state_dict(torch.load("init_params_cls.file"))

    cls.to(device_out)
    
    trainloader = dataset

    trainer = PreTrainer(
        max_predictions_per_seq=args.max_predictions_per_seq, model=model)
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='pt')

    if not args.accum_gradient:
        for i in range(args.epochs):
            if not args.check_correctness:
                trainloader = tqdm(trainloader)
            for iter, batch in enumerate(trainloader):
                encoded_input.data, label, id, pos, weight = batch
                loss = trainer.compute_loss(
                    model, cls, label, id, pos, weight, encoded_input)
                if args.grad_update:
                    BertModel_grad_update(model, cls, loss, in_dev = device_in)
                else:
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if args.check_correctness:
                    print("[Epoch %d] (Iteration %d): Loss = %.3f"%(i, iter, loss.item()))
    else:
        def index(batch, idx): # select one sequence from a batch
            b = []
            for i in range(len(batch)):
                if isinstance(batch[i], dict):
                    d = dict()
                    for key, val in batch[i].items():
                        d[key] = torch.reshape(val[idx],(1,-1))
                    b.append(d)
                else:
                    b.append(torch.reshape(batch[i][idx],(1,-1)))
            return b

        for i in range(args.epochs):
            if not args.check_correctness and not args.profile:
                trainloader = tqdm(trainloader)
            for iter, batch in enumerate(trainloader):
                torch.cuda.empty_cache()
                # Calculate denominator first to scale gradient!
                # mlm_loss = (numerator_0+numerator_1) / denominator, where denominator = denominator_0 + denominator_1 .
                denominator = np.sum(batch[-1].cpu().numpy()) + 1e-5

                # seq1 forward
                encoded_input.data, label, id, pos, weight = index(batch, 0)
                next_sentence_loss_0, numerator_0, denominator_0 = trainer.compute_loss(
                    model, cls, label, id, pos, weight, encoded_input, return_intermediate = True)

                # seq1 backward
                loss_0 = numerator_0 / denominator + next_sentence_loss_0 / 2
                loss_0.backward()
                
                torch.cuda.empty_cache()

                # seq2 forward
                encoded_input.data, label, id, pos, weight = index(batch, 1)
                next_sentence_loss_1, numerator_1, denominator_1 = trainer.compute_loss(
                    model, cls, label, id, pos, weight, encoded_input, return_intermediate = True)

                # seq2 backward
                loss_1 = numerator_1 / denominator + next_sentence_loss_1 / 2
                loss_1.backward()

                torch.cuda.empty_cache()

                loss = loss_0 + loss_1
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                if args.check_correctness or args.profile:
                    print("[Epoch %d] (Iteration %d): Loss = %.3f"%(i, iter, loss.item()))