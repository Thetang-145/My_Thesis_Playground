import os
import sys
import json
import math
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from common.data_importer import prepro_KGData, prepro_textData
from common.finetune_pl_2 import DataModule, BartModel, MyDataset
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import BartForConditionalGeneration, BartTokenizer

import wandb
from pytorch_lightning.loggers import WandbLogger

import torch
from tqdm import tqdm
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, default_data_collator, get_linear_schedule_with_warmup
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, PrefixTuningConfig, TaskType

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = "cuda:0"
# model_name_or_path = "facebook/bart-large"
# tokenizer_name_or_path = "facebook/bart-large"
model_name_or_path = "t5-large"
tokenizer_name_or_path = "t5-large"

text_column = "sentence"
label_column = "text_label"
# max_length = 128
lr = 1e-2
batch_size = 8
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


# def preprocess_function(examples):
#     inputs = examples[text_column]
#     targets = examples[label_column]
#     model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
#     labels = tokenizer(targets, max_length=2, padding="max_length", truncation=True, return_tensors="pt")
#     labels = labels["input_ids"]
#     labels[labels == tokenizer.pad_token_id] = -100
#     model_inputs["labels"] = labels
#     return model_inputs

def add_t5_prefix(sentence, prefix="summarize: "):
    return prefix+sentence 

from rouge_score import rouge_scorer

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


def main():
    parser = argparse.ArgumentParser()
    # Model & Dataset
    parser.add_argument('--model', default='bart-large' , type=str)
    parser.add_argument('--dataset', default='MuP' , type=str)    
    parser.add_argument('--section', default='abstract' , type=str)
    parser.add_argument('--inputType', default='text' , type=str)
    parser.add_argument('--prototype', type=int)
    parser.add_argument("--genSum", type=str)
    # Hyper-parameter
    parser.add_argument('--skip_null', action='store_true', help="skip null section")
    parser.add_argument('--max_input', type=int, help="max input token")
    parser.add_argument('--max_output', default=512, type=int, help="max output token")
    parser.add_argument('--bs', type=int, help="batch size")    
    parser.add_argument('--num_epoch', default=5, type=int, help="num epoch")    
    parser.add_argument('--num_beams', default=5, type=int, help="num beams")    
    parser.add_argument('--lr', default=1e-5, type=int, help="learning rate")
    parser.add_argument('--fp', default=16, type=int, help="floating point") 
    parser.add_argument('--opt', default='adam', type=str, help="optimizer") 
    # Other setting
    parser.add_argument("--saveLast", action='store_true', help="Save model after last epoch")    
    parser.add_argument("--cont", action='store_true', help="Continue finetuning")
    parser.add_argument("--saveData", action='store_true', help="Save data used in training and evaluating")
    parser.add_argument('--cuda', default=0, type=int, help="cuda number")
    args = parser.parse_args()

    # ========== Load & Prepare data ==========
    temp_section = args.section
    args.section = args.section.split("+")
    train_df, spacial_token = prepro_textData(args, "train", sections=args.section, skip_null=args.skip_null)
    val_df, spacial_token_val = prepro_textData(args, "val", sections=args.section, skip_null=False)
    args.max_input == 512
    args.max_output == 512
    args.bs = 8
    args.section = temp_section
    
    train_df['input_seq'] = train_df['input_seq'].apply(add_t5_prefix)
    val_df['input_seq'] = val_df['input_seq'].apply(add_t5_prefix)
    
    train_dataset = MyDataset(args=args, data=train_df, tokenizer=tokenizer, max_input=args.max_input)
    val_dataset = MyDataset(args=args, data=val_df, tokenizer=tokenizer, max_input=args.max_input)
    
    train_dataloader =DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    eval_dataloader =DataLoader(val_dataset, batch_size=args.bs, shuffle=True)
    
    peft_config = PrefixTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, num_virtual_tokens=20)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    model = get_peft_model(model, peft_config)
    # print(type(model))
    # print(model.get_input_embeddings())
    # print(len(tokenizer))
    # new_emb = len(tokenizer) + 20
    # model.resize_token_embeddings(new_emb)
    # print(model.get_input_embeddings())
    # exit()

    model.print_trainable_parameters()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * args.num_epoch),
    )
    model = model.to(device)


    for epoch in range(args.num_epoch):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            # if step==1:
            #     for k, v in batch.items():
            #         print(k, torch.is_tensor(v))
            #         if torch.is_tensor(v): print("\t", v.size()) 
            outputs = model(
                input_ids=batch['input_ids'], 
                attention_mask=batch['input_attention_mask'], 
                labels=batch['target_ids'], 
            )
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        model.eval()
        eval_loss = 0
        eval_preds = []
        rouge_scores = {}
        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            with torch.no_grad():
                outputs = model(
                    input_ids=batch['input_ids'], 
                    attention_mask=batch['input_attention_mask'], 
                    labels=batch['target_ids'], 
                )
            loss = outputs.loss
            eval_loss += loss.detach().float()
            eval_preds.extend(
                tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
            )
            generated_ids = model.generate(
                input_ids=batch['input_ids'], 
                attention_mask=batch['input_attention_mask'], 
                max_length=tokenizer.model_max_length, 
                num_beams=args.num_beams
            )
            hyps = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]
            refs = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch['target_ids']]
            scores = get_rouge(refs, hyps)
            for k, v in scores.items():
                if k not in rouge_scores.keys():
                    rouge_scores[k]=[v] 
                else:
                    rouge_scores[k].append(v)
                    
        
        for k, v in rouge_scores.items():
            rouge_scores[k] = sum(v)/len(v)

        eval_epoch_loss = eval_loss / len(eval_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        

        
        
        print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=} {rouge_scores=}")
    
    peft_model_id = "Thetang/t5-large_PREFIX_TUNING_SEQ2SEQ"
    model.push_to_hub(peft_model_id, use_auth_token=True)




if __name__ == "__main__":
    main()