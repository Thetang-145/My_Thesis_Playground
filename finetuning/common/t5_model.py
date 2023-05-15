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
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from lion_pytorch import Lion

# from rouge import Rouge
from rouge_score import rouge_scorer

from pytorch_lightning import LightningModule, LightningDataModule, Trainer

from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, PrefixTuningConfig, TaskType

model_name_or_path = "t5-large"
TOKENIZER = AutoTokenizer.from_pretrained(model_name_or_path)
MODEL = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)


def get_rouge(refs, hyps):
    metrics = ['rouge1', 'rouge2', 'rougeL']
    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
    
    if len(refs)!=len(hyps):
        raise Exception(f"No. of Refs and Hyps are not equal")
    results = {}
    for metric in metrics: results[metric]=[] 
    for idx in range(len(refs)):
        scores = scorer.score(refs[idx].strip(), hyps[idx].strip())
        for metric in metrics: results[metric].append(scores[metric].fmeasure)
    results = {rouge_metric: np.average(rouge_metric_scores) for (rouge_metric, rouge_metric_scores) in results.items()}
    return results

class MyDataset(Dataset):
    def __init__(self, args, data, max_input):
        self.data = data
        self.paper_id= list(data['paper_id'])
        self.input = list(data['input_seq'])
        self.target = list(data['target_seq'])
        self.tokenizer = TOKENIZER
        self.max_input = max_input
        self.max_output = args.max_output

    def __getitem__(self, index):
        input_tokens = self.tokenizer.encode(
            self.input[index], 
            padding='max_length', 
            max_length=self.max_input, 
            truncation=True, 
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        target_tokens = self.tokenizer.encode(
            self.target[index], 
            padding='max_length', 
            max_length=self.max_input, 
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
        self.tokenizer = TOKENIZER
        self.save_hyperparameters()
        
    def setup(self, stage=None):
        self.train_dataset = MyDataset(
            args = self.args,
            data = self.train_df,
            tokenizer = self.tokenizer,
            max_input = self.args.max_input
        )
        self.val_dataset = MyDataset(
            args = self.args,
            data = self.val_df,
            tokenizer = self.tokenizer,
            max_input = self.args.max_input
        )
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.bs,
            shuffle=True
        )
            
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.args.bs,
            shuffle=False
        )
    
class T5Model(LightningModule):
    def __init__(self, args: argparse.Namespace, spacial_token=None, peft_config=None):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = TOKENIZER
        # print(len(self.tokenizer))
        self.args = args
        self.tokenizer.add_tokens(spacial_token)
        # print(len(self.tokenizer))
        self.model = MODEL
        self.model.resize_token_embeddings(len(self.tokenizer))
        if peft_config!=None:
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
        # print(self.model.get_input_embeddings())

        
    def forward(self, input_ids, attention_mask, target_ids, target_attention_mask):
        # self.model.resize_token_embeddings(len(self.tokenizer))
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=target_ids,
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
            labels=target_ids, 
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
        scores = get_rouge(refs, hyps) 
        for metric, score in scores.items():
            self.log(metric, score, prog_bar=True, logger=True)
        return scores
    
    def predict_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['input_attention_mask']
        generated_ids = self.model.generate(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            max_length=self.tokenizer.model_max_length, 
            num_beams=self.args.num_beams
        )
        generated_seq = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]

        return batch['paper_id'], generated_seq
    
    
    def configure_optimizers(self):
        OPTIMIZERS = {
            "adam": torch.optim.AdamW(self.model.parameters(), lr=self.args.lr),
            "lion": Lion(self.model.parameters(), lr=self.args.lr)
        }
        return OPTIMIZERS[self.args.opt]