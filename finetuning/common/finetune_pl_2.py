import numpy as np
import pandas as pd
import argparse

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

from pytorch_lightning import LightningModule, LightningDataModule, Trainer

   
MODELS = {
    "bart-large": "facebook/bart-large",
    "bart-large-cnn": "facebook/bart-large-cnn",
    "led-base": "allenai/longformer-base-4096"
}

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
        input_tokens = self.tokenizer.encode(
            self.input[index], 
            padding='max_length', 
            max_length=self.max_input_length, 
            truncation=True, 
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        target_tokens = self.tokenizer.encode(
            self.target[index], 
            padding='max_length', 
            max_length=self.max_input_length, 
            truncation=True, 
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )        
        paper_id = self.paper_id[index]
        input_ids = input_tokens.squeeze()
        input_attention_mask = torch.ones(input_ids.shape)
        target_ids = target_tokens.squeeze()
        # target_ids[target_ids==0] = -100
        target_attention_mask = torch.ones(target_ids.shape)
                
        return {
            'paper_id': paper_id, 
            'input_ids': input_ids, 
            'input_attention_mask': input_attention_mask, 
            'target_ids': target_ids, 
            'target_attention_mask': target_attention_mask
        }

    def __len__(self):
        return len(self.data)

class DataModule(LightningDataModule):
    def __init__(self, train_df, val_df, args):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.args = args
        self.tokenizer = BartTokenizer.from_pretrained(MODELS[args.model])
        
    def setup(self, stage=None):
        self.train_dataset = MyDataset(
            self.args,
            self.train_df,
            self.tokenizer
        )
        self.val_dataset = MyDataset(
            self.args,
            self.val_df,
            self.tokenizer
        )
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True
        )
            
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False
        )

class BartModel(LightningModule):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained(MODELS[args.model])
        self.tokenizer = BartTokenizer.from_pretrained(MODELS[args.model])
        self.args = args
        self.model.resize_token_embeddings(len(self.tokenizer))

        
    def forward(self, input_ids, attention_mask, target_ids, target_attention_mask):
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=target_ids,
            decoder_attention_mask=target_attention_mask
        )
        return output.loss, output.logits
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['input_attention_mask']
        target_ids = batch['target_ids']
        target_attention_mask = batch['target_attention_mask']
        
        loss, outputs = self(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            target_ids=target_ids, 
            target_attention_mask=target_attention_mask
        )
        
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['input_attention_mask']
        target_ids = batch['target_ids']
        
        generated_ids = self.model.generate(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            max_length=self.tokenizer.model_max_length, 
            num_beams=self.args.num_beams
        )
        
        hyps = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]
        refs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in target_ids]
        scores = Rouge().get_scores(hyps, refs, avg=True)
        
        self.log("rouge_1", scores['rouge-1']['f'], prog_bar=True, logger=True)
        self.log("rouge_2", scores['rouge-2']['f'], prog_bar=True, logger=True)
        self.log("rouge_l", scores['rouge-l']['f'], prog_bar=True, logger=True)
        return scores
    
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)