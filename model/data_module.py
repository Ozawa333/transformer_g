import os
import json
import torch
from io import BytesIO
from tokenize import tokenize
import pytorch_lightning as pl
from tokenizers import Tokenizer, ByteLevelBPETokenizer
os.environ['TOKENIZERS_PARALLELISM']="true"

"""dataset"""
def aoj28():
    enc_inputs = []
    dec_inputs = []
    cnt = 0
    with open("./datasets/aoj28/data.json", "r") as load_f:
        load_dict = json.load(load_f)
        # print(load_dict["questions"][1]["answers"][0]["answer"])
        for question in load_dict["questions"]:
            for answer in question["answers"]:
                enc_inputs.append(question["description"])
                dec_inputs.append(answer["answer"])
                cnt += 1
    print("total data pairs: ", cnt)
    return enc_inputs, dec_inputs


def apps5000():
    filedir = "./datasets/apps5000/train"
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


def make_tokeinzer(name:str, dataset:list):
    with open("model/"+name+".txt", 'w') as f:
        for enc_input in dataset:
            f.write(enc_input+'\n')

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(["model/"+name+".txt"], min_frequency=1, special_tokens=['<pad>', '<unk>', '<bos>', '<eos>'])
    tokenizer.save("model/"+name+"_tokenizer.json")
    return tokenizer


class data_process:
    def __init__(self, dataset_name, max_len, remake_vocab):
        # special tokens
        self.PAD_IDX, self.UNK_IDX, self.BOS_IDX, self.EOS_IDX = 0, 1, 2, 3
        special_symbols = ['<pad>', '<unk>', '<bos>', '<eos>']
        self.dataset_name = dataset_name
        self.max_len = max_len
        self.remake_vocab = remake_vocab

        if(True == remake_vocab):
            if(self.dataset_name == "aoj28"):
                inputs_dataset, outputs_dataset = aoj28()
            elif(self.dataset_name == "apps5000"):
                inputs_dataset, outputs_dataset = apps5000()
            else:
                raise ValueError("support dataset: aoj28, apps5000")

            inputs_vocab = make_tokeinzer("inputs", inputs_dataset)
            outputs_vocab = make_tokeinzer("outputs", outputs_dataset)

            self.inputs_dataset = inputs_dataset
            self.outputs_dataset = outputs_dataset


        self.inputs_vocab = Tokenizer.from_file("model/inputs_tokenizer.json")
        self.outputs_vocab = Tokenizer.from_file("model/outputs_tokenizer.json")


    def make_enc_inputs(self, inputs_dataset):
        enc_inputs = []
        for i, sample in enumerate(inputs_dataset):
            sample = self.inputs_vocab.encode('<bos>'+sample)
            sample = sample.ids
            sample = sample[:(self.max_len-1)]
            sample.append(self.EOS_IDX)
            #print(self.inputs_vocab.decode(sample))
            #print(sample)
            #print("---------------")
            sample = torch.tensor(sample)
            enc_inputs.append(sample)
        #print(enc_inputs)
        return enc_inputs


    def make_dec_inputs(self, outputs_dataset):
        dec_inputs = []
        for i, sample in enumerate(outputs_dataset):
            sample = self.outputs_vocab.encode('<bos>'+sample)
            sample = sample.ids
            sample = sample[:self.max_len]

            sample = torch.tensor(sample)
            dec_inputs.append(sample)

        return dec_inputs


    def make_dec_outputs(self, outputs_dataset):
        dec_outputs = []
        for i, sample in enumerate(outputs_dataset):
            sample = self.outputs_vocab.encode(sample)
            sample = sample.ids
            sample = sample[:(self.max_len-1)]
            sample.append(self.EOS_IDX)

            sample = torch.tensor(sample)
            dec_outputs.append(sample)
        #print(dec_outputs)
        return dec_outputs


    def make_train_dataset(self):
        enc_inputs = torch.nn.utils.rnn.pad_sequence(self.make_enc_inputs(self.inputs_dataset), padding_value=self.PAD_IDX).transpose(0, 1)
        dec_inputs = torch.nn.utils.rnn.pad_sequence(self.make_dec_inputs(self.outputs_dataset), padding_value=self.PAD_IDX).transpose(0, 1)
        dec_outputs = torch.nn.utils.rnn.pad_sequence(self.make_dec_outputs(self.outputs_dataset), padding_value=self.PAD_IDX).transpose(0, 1)
        #print(enc_inputs)
        return enc_inputs, dec_inputs, dec_outputs
#, padding_value=self.PAD_IDX

    def output_decoder(self, output):
        #print(self.outputs_vocab.decode(output))
        return self.outputs_vocab.decode(output)

    def input_decoder(self, output):
        #print(self.intputs_vocab.decode(output))
        return self.inputs_vocab.decode(output)


class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return len(self.enc_inputs)

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]


class data_module(pl.LightningDataModule):
    def __init__(self, batch_size:int = 32,
                 num_workers:int = 16,
                 dataset_name:str = "aoj28",
                 max_len:int = 1024,
                 rebuild_vocab = True):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.num_workers = num_workers
        self.max_len = max_len
        self.rebuild_vocab = rebuild_vocab

        self.data_process = data_process(self.dataset_name, self.max_len, self.rebuild_vocab)
        self.enc_inputs, self.dec_inputs, self.dec_outputs = self.data_process.make_train_dataset()

    # def prepare_data(self):
        # self.enc_inputs, self.dec_inputs, self.dec_outputs = data_process(self.dataset_name,self.max_len)

    # def setup(self):
        # self.enc_inputs, self.dec_inputs, self.dec_outputs = data_process(self.dataset_name,self.max_len)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(MyDataSet(self.enc_inputs, self.dec_inputs, self.dec_outputs), 
                                           batch_size = self.batch_size, 
                                           num_workers = self.num_workers,
                                           shuffle=True)





