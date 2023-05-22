import os
import sys
import json
import math
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

from common.data_importer import prepro_KGData, prepro_textData
from common.finetune_pl_2 import DataModule, BartModel, MyDataset
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

import wandb
from pytorch_lightning.loggers import WandbLogger

import torch
from transformers import BartTokenizer

print("DOWNLOADING BART TOKENIZER")
TOKENIZER = BartTokenizer.from_pretrained("facebook/bart-large")

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
    
def main():
    parser = argparse.ArgumentParser()
    # Model & Dataset
    parser.add_argument('--model', default='bart-large' , type=str, 
                        help="model used for training: bart-large, bart-large-cnn, etc.")
    parser.add_argument('--dataset', default='MuP' , type=str, help="e.g. Mup, arXiv, etc.")    
    parser.add_argument('--section', default='abstract' , type=str, help="section to gen summary")
    parser.add_argument('--inputType', default='kg' , type=str, help="input types: kg, text")
    parser.add_argument('--prototype', type=int, help="number of data for prototype run")
    parser.add_argument("--test", action='store_true', help="Get generated summary from the trained model")    
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

    
    # checkpoint = torch.load("pre_finetune/bart-large_abstract-kg.ckpt", map_location=lambda storage, loc: storage)
    # print(type(checkpoint))
    # print(checkpoint.keys())
    # print((checkpoint["state_dict"][0]))
    # exit()

    if isinstance(args.prototype, int):
        prototype = '_prototype'
    else:
        prototype = ''

    # ========== Load & Prepare data ==========
    temp_section = args.section
    args.section = args.section.split("+")
    print(args.section)
    if args.test:
        print("Loading Test data")
        if args.inputType == 'kg':
            test_df, spacial_token = prepro_KGData(args, "test", sections=args.section, skip_null=False)
        else:
            test_df, spacial_token = prepro_textData(args, "test", sections=args.section, skip_null=False)
        test_df = test_df.drop_duplicates()
        
        # ========== max_input & BS calculation ========== 
        if args.max_input == None:
            print("***** Calculating max input tokens *****")
            TOKENIZER.add_tokens(spacial_token)
            test_tokens = cal_perc_tokens(test_df, "input_seq", perc=.95)
            args.max_input = min(math.ceil(test_tokens), 1024)
            print(f"Test tokens: {test_tokens:.2f}")
            print(f"Calculation: Max input token = {args.max_input}")
        if args.bs==None:
            args.bs = 4 if args.max_input>950 else 8
        elif args.bs > 4 and args.max_input>950:
            args.bs = 4
        print(f"Calculation: using BS = {args.bs}")
        
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


    exp_name = f'{args.model}_{args.section}-{args.inputType}{prototype}'
    if args.test:
        exp_name = f'{args.model}_{args.section}-{args.inputType}'
        if args.model=='bart-large-kg':
            spacial_token = ["[Metric]","[Method]","[/Material]","[/Task]","[PART-OF]","[USED-FOR]","[/OtherScientificTerm]","[/Generic]","[EVALUATE-FOR]","[CONJUNCTION]","[Material]","[FEATURE-OF]","[Task]","[COMPARE]","[Generic]","[OtherScientificTerm]","[HYPONYM-OF]","[/Method]","[/Metric]"]
        ckpt_path = f"checkpoints/{exp_name}.ckpt"        
        model = BartModel.load_from_checkpoint(ckpt_path, args=args, spacial_token=spacial_token)
        print("LOAD MODEL FOR TESTING")
        
    elif args.model=='bart-large-kg':
        ckpt_path = "pre_finetune/bart-large_abstract-kg-v1.ckpt"
        spacial_token = ["[Metric]","[Method]","[/Material]","[/Task]","[PART-OF]","[USED-FOR]","[/OtherScientificTerm]","[/Generic]","[EVALUATE-FOR]","[CONJUNCTION]","[Material]","[FEATURE-OF]","[Task]","[COMPARE]","[Generic]","[OtherScientificTerm]","[HYPONYM-OF]","[/Method]","[/Metric]"]
        model = BartModel.load_from_checkpoint(ckpt_path, train_df=train_df, val_df=val_df, args=args, spacial_token=spacial_token)
    else:
        model = BartModel(args, spacial_token=spacial_token)
    print(f"Tokenizer length: {len(model.tokenizer)}")
    print(f"Model emb: {(model.model.get_input_embeddings())}")
    
    dirpath="checkpoints" if args.dataset=='MuP' else "pre_finetune"
    
    # ========== Setting before Wandb ==========     
    seed_everything(42, workers=True)

    if args.test:
        test_dataset = MyDataset(args, test_df, model.tokenizer)
        test_dataloader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)
        trainer = Trainer(devices=[0], accelerator="gpu", precision=args.fp)
        predictions = trainer.predict(model, dataloaders=test_dataloader)
        
        predictions_df = []
        for batch in predictions:
            for i in range(args.bs):
                predictions_df.append({
                    "paper_id": batch[0][i],
                    "summary": batch[1][i],
                })
        predictions_df = pd.DataFrame(predictions_df)
        predictions_df.to_csv(f"{exp_name}.csv", index=False)

        
    else:
        project_name = 'MuP-experiment' if args.dataset=='MuP' else 'arXiv-PreFinetune'
        if args.dataset=='arXiv': args.saveLast=True
        # ========== Start Wandb ==========
        wandb_logger = WandbLogger(
            # project='MuP-project',
            project=project_name,
            name=exp_name,
        )

        wandb_logger.experiment.config.update({
            "model": args.model, 
            "input_section": args.section, 
            "input_type": args.inputType, 
            "batch_size": args.bs, 
            "max_input_length": args.max_input, 
            "max_output_length": args.max_output, 
            "num_beams": args.num_beams, 
            "learning_rate": args.lr,  
            "fp": args.fp, 
            "optimizer": args.opt, 
        })
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
