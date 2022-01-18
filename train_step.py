import torch
import pytorch_lightning as pl
from model.data_module import data_module
from model.data_module import data_process
from pytorch_lightning.callbacks import ModelCheckpoint
from model.transformer_model import Transformer, greedy_decoder


MAX_LEN = 1024
train = False
#data_process = data_process("apps5000",1024, remake_vocab = False)



class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.transformer = Transformer()

    def forward(self, sentences):
        # for i, enc_input in enumerate(sentences):
        # print(type(enc_input), "-----------------------")
        # print(type(sentences[0]))
        sentences = sentences[0].cuda()
        _, nexts = greedy_decoder(self.transformer, sentences.view(1, -1), start_symbol = 2)
            # print(enc_input)
            # greedy_dec_input, nexts = greedy_decoder(self.transformer, enc_input.view(1, -1), start_symbol = 2)
            # predict, _, _, _ = self.transformer(enc_input[i].view(1, -1), greedy_dec_input)
            # predict = predict.data.max(1, keepdim=True)[1]
        return nexts

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
        return optimizer

    def training_step(self, batch, batch_idx):
        enc_inputs, dec_inputs, dec_outputs = batch
        #print(data_process.input_decoder(enc_inputs.tolist()[0]), 
              #data_process.output_decoder(dec_inputs.tolist()[0]), 
              #data_process.output_decoder(dec_outputs.tolist()[0]))
        #print(enc_inputs, dec_inputs, dec_outputs)
        outputs, _, _, _ = self.transformer(enc_inputs, dec_inputs)
        loss = torch.nn.CrossEntropyLoss(ignore_index=0)(outputs, dec_outputs.view(-1))
        # print(outputs.size(), dec_outputs.view(-1).size())
        print("--------------------------")
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


if(train == True):
    checkpoint_callback = ModelCheckpoint(save_weights_only = True,
                                            every_n_epochs = 5,
                                            monitor = "train_loss",
                                            #save_top_k = 3,
                                            )

    model = Model()

    data_loader = data_module(batch_size  = 2,
                             num_workers  = 16,
                             dataset_name = "apps5000",
                             max_len      = MAX_LEN,
                             rebuild_vocab = True)

    trainer = pl.Trainer(devices=1, accelerator="gpu", #strategy="ddp",
                         max_epochs=20,
                         # default_root_dir="./pl_checkpoint_2GPU",
                         callbacks=[checkpoint_callback]
                         )

    trainer.fit(model, datamodule=data_loader)


