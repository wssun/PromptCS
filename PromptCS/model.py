import math
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, RobertaConfig, RobertaModel


class PromptCS(nn.Module):
    def __init__(self, args, device, template):
        super(PromptCS, self).__init__()
        self.args = args
        self.mode = args.mode
        self.device = device
        self.use_lm_finetune = False

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

        if args.mode == 'finetune':
            self.use_lm_finetune = True
            template = [0, 0]
        self.template = template

        # set allowed vocab set
        self.vocab = self.tokenizer.get_vocab()
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['[PROMPT]']})
        self.pseudo_token_id = self.tokenizer.get_vocab()['[PROMPT]']
        self.pad_token_id, self.sep_token_id, self.eos_token_id, self.unk_token_id = self.get_special_token_id()

        self.prompt_tokens = [self.pseudo_token_id]
        self.sep_tokens = [self.sep_token_id]
        self.eos_tokens = [self.eos_token_id]

        # load pre-trained model
        self.model = create_model(self.args, self.use_lm_finetune)
        self.model = self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = self.use_lm_finetune
        self.embeddings = get_embedding_layer(self.args, self.model)

        # load prompt encoder
        self.hidden_size = self.embeddings.embedding_dim
        self.spell_length = sum(self.template)

        if args.mode == 'PromptCS':
            self.prompt_agent = PromptAgent(self.template, self.hidden_size, self.tokenizer, self.device, args)
            self.prompt_agent = self.prompt_agent.to(self.device)

        self.max_target_length = args.max_target_length
        self.max_code_length = args.max_code_length
        self.lsm = nn.LogSoftmax(dim=-1)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_token_id, reduction='sum')

    def get_special_token_id(self):
        pad_token_id, sep_token_id, eos_token_id, unk_token_id = None, None, None, None
        model_name = self.args.model_name_or_path.lower()
        if 'starcoder' in model_name:
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.unk_token_id
            sep_token_id = self.vocab['<fim_middle>']
            eos_token_id = self.tokenizer.eos_token_id
            unk_token_id = self.tokenizer.unk_token_id
        elif 'polycoder' in model_name:
            pad_token_id = self.vocab['<|padding|>']
            sep_token_id = self.vocab['<|separator|>']
            eos_token_id = self.vocab['<|endoftext|>']
            unk_token_id = self.vocab['<|padding|>']
        elif 'codegen' in model_name:
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.unk_token_id
            sep_token_id = self.vocab['//']
            eos_token_id = self.tokenizer.eos_token_id
            unk_token_id = self.tokenizer.unk_token_id

        return pad_token_id, sep_token_id, eos_token_id, unk_token_id

    def embed_input(self, queries):
        if self.mode == 'PromptCS':
            return self.cstuning_embed_input(queries)
        else:
            return self.finetune_embed_input(queries)

    def finetune_embed_input(self, queries):
        return self.embeddings(queries)

    def cstuning_embed_input(self, queries):
        bz = queries.shape[0]
        queries_for_embedding = queries.clone()
        queries_for_embedding[(queries == self.pseudo_token_id)] = self.unk_token_id
        raw_embeds = self.embeddings(queries_for_embedding)

        blocked_indices = (queries == self.pseudo_token_id).nonzero().reshape((bz, self.spell_length, 2))[:, :, 1]  # bz
        replace_embeds = self.prompt_agent()
        for bidx in range(bz):
            for i in range(self.prompt_agent.spell_length):
                raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]
        return raw_embeds

    def get_query(self, x_h, x_t=None):
        left = self.prompt_tokens * self.template[0] + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(x_h)[:self.max_code_length]) + self.prompt_tokens * self.template[1]

        if x_t is not None:
            right = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(x_t)[:self.max_target_length]) + self.eos_tokens
        else:
            right = []

        input_ids = left + self.sep_tokens + right

        return torch.LongTensor(input_ids),  len(left)

    def prepare_inputs(self, inputs):
        inputs = pad_sequence(inputs, True, padding_value=self.pad_token_id).long().to(self.device)

        attention_mask = inputs != self.pad_token_id
        inputs_embeds = self.embed_input(inputs)

        inputs_embeds = inputs_embeds.to(self.device)
        attention_mask = attention_mask.to(self.device)

        if self.mode != 'finetune':
            inputs_embeds = inputs_embeds.half()
            attention_mask = attention_mask.half()

        return inputs, inputs_embeds, attention_mask


    def forward(self, x_hs=None, x_ts=None):
        bz = len(x_hs)

        if x_ts is not None:
            inputs, sum_idx, ext_inputs = [], [], []
            for i in range(bz):
                input, idx = self.get_query(x_hs[i], x_ts[i])
                inputs.append(input)
                sum_idx.append(idx)

            inputs, inputs_embeds, attention_mask = self.prepare_inputs(inputs)

            output = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

            logits = output.logits
            loss = None

            for i in range(bz):
                idx = sum_idx[i]
                shift_logits = logits[i][idx:-1, :].contiguous()
                shift_labels = inputs[i][idx+1:].contiguous()

                if loss is None:
                    loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                else:
                    loss += self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            loss = loss / bz

            return loss
        else:
            queries, sum_idx, tmp_idx = [], [], []
            for i in range(bz):
                query, idx = self.get_query(x_h=x_hs[i])
                queries.append(query)
                sum_idx.append(idx)
                tmp_idx.append(idx)

            for _ in range(self.max_target_length):
                inputs, inputs_embeds, attention_mask = self.prepare_inputs(queries)

                output = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

                logits = output.logits
                for i in range(bz):
                    idx = tmp_idx[i]
                    tmp_idx[i] += 1
                    next_token_logits = logits[i, idx:idx+1, :]
                    _, next_token = torch.max(next_token_logits, dim=1)

                    queries[i] = torch.cat([queries[i].to(self.device), next_token], dim=0)

            answer = []
            for i in range(bz):
                idx = sum_idx[i]
                t = queries[i][idx+1:]
                t=t.tolist()
                if self.eos_token_id in t:
                    t = t[:t.index(self.eos_token_id)]
                words = self.tokenizer.decode(t).replace('\n','')
                answer.append(words)

            return answer



class PromptAgent(torch.nn.Module):
    def __init__(self, template, hidden_size, tokenizer, device, args):
        super().__init__()
        self.device = device
        self.spell_length = sum(template)
        self.hidden_size = hidden_size
        self.embed_size = hidden_size
        self.tokenizer = tokenizer
        self.args = args
        # ent embedding
        self.cloze_length = template
        self.cloze_mask = [
            [1] * self.cloze_length[0]  # first cloze
            + [1] * self.cloze_length[1]  # second cloze
        ]
        self.cloze_mask = torch.LongTensor(self.cloze_mask).bool().to(self.device)
        self.seq_indices = torch.LongTensor(list(range(len(self.cloze_mask[0])))).to(self.device)

        if args.prompt_encoder_type == "lstm":
            self.prompt_encoder = Encoder_BiLSTM(input_size=self.hidden_size,
                                           hidden_size=self.hidden_size // 2,
                                           num_layers=2,
                                           dropout=0.0,
                                           bidirectional=True,
                                           batch_first=True)
        elif args.prompt_encoder_type == "transformer":
            self.prompt_encoder = Encoder_Transformer(d_model=self.hidden_size,
                                                  nhead=8,
                                                  num_layers=6,
                                                  max_len=len(self.cloze_mask[0]))


        self.embedding = torch.nn.Embedding(len(self.cloze_mask[0]), self.embed_size).to(self.device)
        self.mlp_head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.hidden_size))
        print("init prompt encoder...")

    def forward(self):
        input_embeds = self.embedding(self.seq_indices).unsqueeze(0)
        output_embeds = self.prompt_encoder(input_embeds)
        output_embeds = self.mlp_head(output_embeds).squeeze()
        return output_embeds


class Encoder_BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional, batch_first):
        super(Encoder_BiLSTM, self).__init__()
        self.lstm_head = torch.nn.LSTM(input_size=input_size,
                                           hidden_size=hidden_size,
                                           num_layers=num_layers,
                                           dropout=dropout,
                                           bidirectional=bidirectional,
                                           batch_first=batch_first)

    def forward(self, inputs):
        outputs = self.lstm_head(inputs)[0]

        return outputs


class Encoder_Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, max_len):
        super(Encoder_Transformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.pos_embedding = PositionalEncoding(d_model, 0.1, max_len)

    def forward(self, inputs):
        input_embedding = self.pos_embedding(inputs)
        input_embedding = input_embedding.permute(1, 0, 2)

        outputs = self.encoder(input_embedding).permute([1, 0, 2])

        return outputs

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2501):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [[0],[1],...[4999]] 5000 * 1
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(
            10000.0) / d_model))  # e ^([0, 2,...,198] * -ln(10000)(-9.210340371976184) / 200) [1,0.912,...,(1.0965e-04)]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe

        return self.dropout(x)


def create_model(args, use_lm_finetune):
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    if not use_lm_finetune:
        model = model.half()
    return model


def get_embedding_layer(args, model):
    return model.base_model.get_input_embeddings()
        


