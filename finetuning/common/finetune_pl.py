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

from datetime import datetime
from typing import Optional
import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything


def print_log(msg):
    print(msg)
    logging.info(msg)    

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
    def __init__(
        self,
        model_name_or_path: str,
        args,
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.args = args

        self.tokenizer = BartTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name_or_path)
        self.metric = Rouge()

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss
