import os
import sys
import json
import math
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

from common.data_importer import prepro_KGData, prepro_textData
from common.t5_model import DataModule, T5Model, MyDataset
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning_transformers.plugins.checkpoint import HFSaveCheckpoint

from datasets import load_metric
from torch.utils.data import DataLoader

import wandb
from pytorch_lightning.loggers import WandbLogger

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PrefixTuningConfig, TaskType

print("DOWNLOADING TOKENIZER")
model_name_or_path = "t5-large"
TOKENIZER = AutoTokenizer.from_pretrained(model_name_or_path)

def get_token(sentence, get_len=True):
    tokens = TOKENIZER.encode(
        sentence, 
        padding=False, 
        truncation=True, 
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    )
    if get_len: return tokens.shape[1]
    else: return tokens

def cal_perc_tokens(df, col, perc=.95):
    tokens_df = pd.DataFrame(df[col].apply(get_token))
    return tokens_df.quantile(perc).iloc[0]
    
def change_max_token():
    pass
    
def main():
    parser = argparse.ArgumentParser()
    # Model & Dataset
    parser.add_argument('--model', default='t5-large' , type=str, 
                        help="model used for training")
    parser.add_argument('--dataset', default='MuP' , type=str, help="e.g. Mup, arXiv, etc.")    
    parser.add_argument('--section', default='abstract' , type=str, help="section to gen summary")
    parser.add_argument('--inputType', default='text' , type=str, help="input types: kg, text")
    parser.add_argument('--prototype', type=int, help="number of data for prototype run")
    parser.add_argument("--genSum", type=str, help="Generated summary from the val, test, etc dataset")
    # Hyper-parameter
    parser.add_argument('--skip_null', action='store_true', help="skip null section")
    parser.add_argument('--max_input', type=int, help="max input token")
    parser.add_argument('--max_output', default=512, type=int, help="max output token")
    parser.add_argument('--bs', type=int, help="batch size")    
    parser.add_argument('--num_epoch', default=5, type=int, help="num epoch")    
    parser.add_argument('--num_beams', default=3, type=int, help="num beams")    
    parser.add_argument('--lr', default=1e-5, type=int, help="learning rate")
    parser.add_argument('--fp', default=16, type=int, help="floating point") 
    parser.add_argument('--opt', default='adam', type=str, help="optimizer") 
    # Other setting
    parser.add_argument("--saveLast", action='store_true', help="Save model after last epoch")    
    parser.add_argument("--cont", action='store_true', help="Continue finetuning")
    parser.add_argument("--saveData", action='store_true', help="Save data used in training and evaluating")
    parser.add_argument('--cuda', default=0, type=int, help="cuda number")
    args = parser.parse_args()

    if isinstance(args.prototype, int):
        prototype = '_prototype'
    else:
        prototype = ''

    # ========== Load & Prepare data ==========
    temp_section = args.section
    args.section = args.section.split("+")
    print(args.section)
    if args.genSum!=None:
        print("Loading Test data")
        if args.inputType == 'kg':
            test_df, spacial_token = prepro_KGData(args, args.genSum, sections=args.section, skip_null=False)
        else:
            test_df, spacial_token = prepro_textData(args, args.genSum, sections=args.section, skip_null=False)
        test_df["target_seq"] = "<PAD>"
        if "index" in (test_df.columns): 
            test_df.drop("index", inplace=True, axis=1)
        test_df = test_df.drop_duplicates()
        print(f"Loaded test dataset: {len(test_df)} samples")
        # exit()
        
        # ========== max_input & BS calculation ========== 
        if args.max_input == None:
            print("***** Calculating max input tokens *****")
            TOKENIZER.add_tokens(spacial_token)
            test_tokens = cal_perc_tokens(test_df, "input_seq", perc=.95)
            args.max_input = min(math.ceil(test_tokens), 1024)
            print(f"Test tokens: {test_tokens:.2f}")
            print(f"Calculation: Max input token = {args.max_input}")
        if args.max_output == None:
            print("***** Calculating max output tokens *****")
            TOKENIZER.add_tokens(spacial_token)
            test_tokens = cal_perc_tokens(test_df, "target_seq", perc=.999)
            args.max_output = min(math.ceil(test_tokens), 1024)
            print(f"Test tokens: {test_tokens:.2f}")
            print(f"Calculation: Max output token = {args.max_input}")
        if args.bs==None: args.bs = 8
        print(f"BS = {args.bs}")
        
    else:
        print("Loading Train data")
        if args.inputType == 'kg':
            train_df, spacial_token_train = prepro_KGData(args, "train", sections=args.section, skip_null=args.skip_null)
            val_df, spacial_token_val = prepro_KGData(args, "val", sections=args.section, skip_null=False)
        else:        
            train_df, spacial_token_train = prepro_textData(args, "train", sections=args.section, skip_null=args.skip_null)
            val_df, spacial_token_val = prepro_textData(args, "val", sections=args.section, skip_null=False)

        if args.saveData: 
            train_df.to_csv(f"model/trainDataset_{args.section}_{args.inputType}{prototype}.csv")
            val_df.to_csv(f"model/valDataset_{args.section}_{args.inputType}{prototype}.csv")
            
        if spacial_token_train==None and spacial_token_val==None:
            spacial_token = None
        else:
            spacial_token = list(set(spacial_token_train).union(set(spacial_token_val)))
        print("added spacial tokens: ", spacial_token)
    
        # ========== max_input & BS calculation ========== 
        if args.max_input == None:
            print("***** Calculating max input tokens *****")
            TOKENIZER.add_tokens(spacial_token)
            train_tokens = cal_perc_tokens(train_df, "input_seq", perc=.95)
            val_tokens = cal_perc_tokens(val_df, "input_seq", perc=.95)
            args.max_input = min(math.ceil(max(train_tokens, val_tokens)), 1024)
            print(f"Train tokens: {train_tokens:.2f} | Val tokens: {val_tokens:.2f}")
            print(f"Calculation: Max input token = {args.max_input}")
        args.bs = 4 if args.max_input>950 else 8
        print(f"Calculation: using BS = {args.bs}")
        
    args.section = temp_section


    exp_name = f'{args.model}_{args.section}_{args.inputType}{prototype}'

    peft_config = PrefixTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, num_virtual_tokens=20)
    model = T5Model(args, spacial_token=spacial_token, peft_config=peft_config)
    print(f"Tokenizer length: {len(model.tokenizer)}")
    print(f"Model emb: {(model.model.get_input_embeddings())}")
    
    dirpath="checkpoints" if args.dataset=='MuP' else "pre_finetune"
    
    # ========== Setting before Wandb ==========     
    seed_everything(42, workers=True)

    if args.genSum!=None:
        test_dataset = MyDataset(args, test_df, model.tokenizer, max_input=args.max_input)
        test_dataloader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)
        trainer = Trainer(devices=[args.cuda], accelerator="gpu", precision=args.fp)
        predictions = trainer.predict(model, dataloaders=test_dataloader)
        
        predictions_df = []
        for batch in predictions:
            for i in range(len(batch[0])):
                predictions_df.append({
                    "paper_id": batch[0][i],
                    "summary": batch[1][i],
                })
        predictions_df = pd.DataFrame(predictions_df)
        export_path = f"generated_summary/{args.genSum}/{exp_name}.csv"
        predictions_df.to_csv(export_path, index=False)
        print(f"Succesfully export {len(predictions_df)} summaries to {export_path}")
        
    else:
        project_name = 'PrefixTuning-experiment'
        
        # ========== Start Wandb ==========
        wandb_logger = WandbLogger(
            # project='MuP-project',
            project=project_name,
            name=exp_name,
        )

        # wandb_logger.experiment.config.update({
            # "model": args.model, 
            # "input_section": args.section, 
            # "input_type": args.inputType, 
            # "batch_size": args.bs, 
            # "max_input_length": args.max_input, 
            # "max_output_length": args.max_output, 
            # "num_beams": args.num_beams, 
            # "learning_rate": args.lr,  
            # "fp": args.fp, 
            # "optimizer": args.opt, 
            # "num_cluster": num_cluster,
        # })
        checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,
            # filename="best-checkpoint",
            filename=exp_name,
            save_last=args.saveLast,
            verbose=True,
            # every_n_epochs=1,
            monitor="rouge1",
            mode="max",
            # filename_suffix='.pt',
        )
        checkpoint_callback.CHECKPOINT_NAME_LAST = f"{exp_name}-last"
        # checkpoint_callback.FILE_EXTENSION = ".pt"

        if args.prototype == None:
            trainer = Trainer(
                logger=wandb_logger,
                callbacks=[checkpoint_callback],
                plugins=HFSaveCheckpoint(model=model),
                max_epochs=args.num_epoch,
                # val_check_interval=1,
                devices=[0], 
                accelerator="gpu",
                precision=args.fp,
            )
        else:
            trainer = Trainer(
                logger=wandb_logger,
                max_epochs=args.num_epoch,
                # val_check_interval=1,
                devices=[0], 
                accelerator="gpu",
                precision=args.fp,
            )

        data_module = DataModule(train_df, val_df, args)
        if args.cont: 
            trainer.fit(model, data_module, ckpt_path)
        else: 
            trainer.fit(model, data_module)

        wandb.finish()        
        
if __name__ == "__main__":
    main()
