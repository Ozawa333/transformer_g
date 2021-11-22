from transformer import Transformer
import pytorch_lightning as pl
import torch
import json
import os
import torchtext
from tokenize import tokenize
from io import BytesIO

MAX_LEN = 512


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.transformer = Transformer()

    def forward(self, enc_inputs, dec_inputs, dec_outputs):
        # outputs: [batch_size * tgt_len, tgt_vocab_size]
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = self.transformer(enc_inputs, dec_inputs)
        return outputs, enc_self_attns, dec_self_attns, dec_enc_attns

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
        return optimizer

    def training_step(self, batch, batch_idx):
        enc_inputs, dec_inputs, dec_outputs = batch
        # print(enc_inputs, dec_inputs, dec_outputs) 
        outputs, _, _, _ = self.transformer(enc_inputs, dec_inputs)
        loss = torch.nn.CrossEntropyLoss(ignore_index=0)(outputs, dec_outputs.view(-1))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


def APPS5000():
    filedir = "../apps_dataset/train"
    enc_inputs = []
    dec_inputs = []
    cnt = 0

    for folder in os.listdir(filedir):
        with open(filedir+"/"+folder+"/question.txt","r", encoding="utf8") as f_ques:
            question = "".join(f_ques.readlines())
        with open(filedir+"/"+folder+"/solutions.json","r", encoding="utf8") as f_solu:
            solutions = json.load(f_solu)
            for solution in solutions:
                try:
                    g = tokenize(BytesIO(solution.encode('utf-8')).readline)  # tokenize the string
                    for _, tokval, _, _, _ in g:
                        continue
                    enc_inputs.append(question)
                    dec_inputs.append(solution)
                    cnt += 1
                except:
                    continue

    print("total data pairs: ", cnt)
    return enc_inputs, dec_inputs
inputs_dataset, outputs_dataset = APPS5000()



class python_tokenize():
    # referance: https://docs.python.org/3/library/tokenize.html
    # referance: https://zhuanlan.zhihu.com/p/357021687
    def __call__(self, s):
        result = []
        g = tokenize(BytesIO(s.encode('utf-8')).readline)  # tokenize the string
        for _, tokval, _, _, _ in g:
            result.append(tokval)
        return result


inputs_tokenizer = torchtext.data.utils.get_tokenizer('spacy', language='en_core_web_sm')
outputs_tokenizer = python_tokenize()


PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<pad>', '<unk>', '<bos>', '<eos>']


def yield_tokens_i(dataset):
    for data_sample in dataset:
        yield inputs_tokenizer(data_sample)


def yield_tokens_o(dataset):
    for data_sample in dataset:
        yield outputs_tokenizer(data_sample)


inputs_vocab = torchtext.vocab.build_vocab_from_iterator(yield_tokens_i(inputs_dataset),
                                        min_freq=2,
                                        specials=special_symbols,
                                        special_first=True)

outputs_vocab = torchtext.vocab.build_vocab_from_iterator(yield_tokens_o(outputs_dataset),
                                        min_freq=2,
                                        specials=special_symbols,
                                        special_first=True)


src_vocab_size = len(inputs_vocab)
tgt_vocab_size = len(outputs_vocab)

inputs_vocab.set_default_index(UNK_IDX)
outputs_vocab.set_default_index(UNK_IDX)

# enc_inputs = []
dec_inputs = []
dec_outputs = []


def input_process(inputs_dataset):
    processed = []
    for i, sample in enumerate(inputs_dataset):
        sample = inputs_tokenizer(sample)
        # print(sample)
        sample = inputs_vocab(sample)
        # print(sample)
        sample.insert(0, BOS_IDX)
        sample.append(EOS_IDX)
        sample = torch.tensor(sample)
        # print(sample)
        processed.append(sample[:MAX_LEN])
    return processed

# print(inputs_dataset)
enc_inputs = input_process(inputs_dataset)

for i, sample in enumerate(outputs_dataset):
    sample = outputs_tokenizer(sample)
    sample = sample[1:MAX_LEN-1]
    # print(sample)
    sample = outputs_vocab(sample)
    # print(sample)
    sample.insert(0, BOS_IDX)
    sample = torch.tensor(sample)
    # print(sample)
    dec_inputs.append(sample)

for i, sample in enumerate(outputs_dataset):
    sample = outputs_tokenizer(sample)
    sample = sample[1:MAX_LEN-1]
    # print(sample)
    sample = outputs_vocab(sample)
    # print(sample)
    sample.append(EOS_IDX)
    sample = torch.tensor(sample)
    # print(sample)
    dec_outputs.append(sample)

enc_inputs = torch.nn.utils.rnn.pad_sequence(enc_inputs, padding_value=PAD_IDX).transpose(0, 1)
dec_inputs = torch.nn.utils.rnn.pad_sequence(dec_inputs, padding_value=PAD_IDX).transpose(0, 1)
dec_outputs = torch.nn.utils.rnn.pad_sequence(dec_outputs, padding_value=PAD_IDX).transpose(0, 1)

class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]


loader = torch.utils.data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs),batch_size = 8, num_workers = 16)
trainer = pl.Trainer(gpus=2, accelerator='ddp', default_root_dir="./pl_checkpoint", max_epochs=20)
model = Model()

trainer.fit(model, loader)


