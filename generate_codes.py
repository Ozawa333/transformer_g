import os
import json
import torch
from train_step import Model
from model.data_module import data_process

test_dir = "./datasets/apps5000/test/"
result_dir = "./codes.json"

question_dir = os.listdir(test_dir)
question_dir = sorted(question_dir)
codes = {}

model = Model.load_from_checkpoint("lightning_logs/1024_single/checkpoints/epoch=19-step=390259.ckpt")
data_process = data_process("apps5000",1024, remake_vocab = False) 

def generate_one_completion(sentences):
    with torch.no_grad():
        sentences = data_process.make_enc_inputs([sentences])
        dec_output = model(sentences)
        #print(dec_output)
        dec_output = data_process.output_decoder(dec_output)
        #print(dec_output)
    return (dec_output)

for index, folder in enumerate(question_dir):
    with open(test_dir+folder+"/question.txt", "r",) as f_ques:
        question = "".join(f_ques.readlines())
        #print(question)
        codes[str(index)] = generate_one_completion(question)
        print("--------------", index, "--------------")
        print(codes[str(index)])
        print()
        #break

with open(result_dir, "w") as f:
    json.dump(codes, f)








