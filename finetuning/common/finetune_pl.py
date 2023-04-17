import numpy as np
import pandas as pd

import logging
import timeit
from datetime import datetime, timedelta

import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import AutoTokenizer, AutoModel
from rouge import Rouge

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import NeptuneLogger

neptune_logger = NeptuneLogger(
    project="thetang/BART-Finetune",
    api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vbmV3LXVpLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyMWIzODIzZi05ZmZmLTRiNWYtYmM3Mi04MTI4NTcyYzhmN2UifQ==",
    log_model_checkpoints=False,
)

PARAMS = {
    "batch_size": 8,
    "lr": 3e-5,
    "max_epochs": 3,
    "max_": 3,
}

neptune_logger.log_hyperparams(params=PARAMS)


class MyDataset(Dataset):
    def __init__(self, args, data, tokenizer):
        self.data = data
        self.paper_id= list(data['paper_id'])
        self.input = list(data['input_seq'])
        self.target = list(data['target_seq'])
        self.tokenizer = tokenizer
        self.max_input_length = args.max_input_length
        self.max_output_length = args.max_output_length

    def __getitem__(self, index):
        input_tokens = self.tokenizer.encode(self.input[index], padding='max_length', max_length=self.max_input_length, truncation=True, return_tensors='pt')
        target_tokens = self.tokenizer.encode(self.target[index], padding='max_length', max_length=self.max_output_length, truncation=True, return_tensors='pt')
        
        paper_id = self.paper_id[index]
        input_ids = input_tokens.squeeze()
        input_attention_mask = torch.ones(input_ids.shape)
        target_ids = target_tokens.squeeze()
        # target_attention_mask = torch.ones(target_ids.shape)
                
        return paper_id, input_ids, input_attention_mask, target_ids

    def __len__(self):
        return len(self.data)

class BARTFineTuner(LightningModule):
    def __init__(self, model_name_or_path, args):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.args = args

        self.tokenizer = BartTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name_or_path)
        self.metric = Rouge()

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask)

    def training_step(self, batch, batch_idx):
        paper_id, input_ids, attention_mask, target_ids = batch
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        target_ids = target_ids.to(self.device)

        # Forward pass
        outputs = self(input_ids, attention_mask=attention_mask)
        loss = outputs.loss
        self.log("metrics/batch/loss", loss, prog_bar=False)

        return loss    

    def training_epoch_end(self, outputs):
        loss = np.array([])
        for results_dict in outputs:
            loss = np.append(loss, results_dict["loss"])
        self.log("metrics/epoch/loss", loss.mean())

    def validation_step(self, batch, batch_idx):
        paper_id, input_ids, attention_mask, target_ids = batch
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        target_ids = target_ids.to(self.device)

        # Forward pass
        outputs = self(input_ids, attention_mask=attention_mask, target_ids=target_ids)
        loss = outputs.loss

        # Log validation loss
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Calculate rouge score
        hyps = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs]
        refs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in target_ids]
        scores = Rouge().get_scores(hyps, refs)

        return {
            "loss": loss, 
            "rouge-1": scores['rouge-1']['f'],
            "rouge-2": scores['rouge-2']['f'],
            "rouge-l": scores['rouge-l']['f'],
            }
    
    def validation_epoch_end(self, outputs):
        loss = np.array([])
        rouge1 = np.array([])
        rouge2 = np.array([])
        rougel = np.array([])
        for results_dict in outputs:
            loss = np.append(loss, results_dict["loss"])
            rouge1 = np.append(rouge1, results_dict["rouge-1"])
            rouge2 = np.append(rouge2, results_dict["rouge-2"])
            rougel = np.append(rougel, results_dict["rouge-l"])
        self.logger.experiment["val/loss"] = loss.mean()
        self.logger.experiment["val/rouge/r1"] = rouge1
        self.logger.experiment["val/rouge/r2"] = rouge2
        self.logger.experiment["val/rouge/rl"] = rougel
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate)
        return optimizer

    def train_dataloader(self):
        train_dataset = MyDataset(self.args, self.train_data, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=self.args.train_batch_size)
        return train_loader
    
    def val_dataloader(self):
        val_dataset = MyDataset(self.args, self.val_data, self.tokenizer)
        val_loader = DataLoader(val_dataset, batch_size=self.args.val_batch_size)
        return val_loader
