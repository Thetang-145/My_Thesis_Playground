import os
import sys
import json
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

import common.data_importer as data_importer
from common.finetune_pl_2 import DataModule, BartModel
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

# from pytorch_lightning.loggers import TensorBoardLogger
# from tensorboard import program

# from pytorch_lightning.loggers import NeptuneLogger

import wandb
from pytorch_lightning.loggers import WandbLogger
    
def main():
    parser = argparse.ArgumentParser()
    # Model & Dataset
    parser.add_argument('--model', default='bart-large' , type=str, 
                        help="model used for training: bart-large, bart-large-cnn, etc.")
    parser.add_argument('--dataset', default='MuP' , type=str, help="e.g. Mup, arXiv, etc.")    
    parser.add_argument('--section', default='abstract' , type=str, help="section to gen summary")
    parser.add_argument('--inputType', default='kg' , type=str, help="input types: kg, text")
    parser.add_argument('--prototype', type=int, help="number of data for prototype run")
    # Hyper-parameter
    parser.add_argument('--max_input_length', default=512, type=int, help="x")
    parser.add_argument('--max_output_length', default=512, type=int, help="x")
    parser.add_argument('--batch_size', default=8, type=int, help="")    
    # parser.add_argument('--val_batch_size', default=8, type=int, help="")    
    # parser.add_argument('--eval_batch_size', default=8, type=int, help="")    
    # parser.add_argument('--eval_frequency', default=1000, type=int, help="")    
    parser.add_argument('--num_epoch', default=5, type=int, help="")    
    parser.add_argument('--num_beams', default=4, type=int, help="")    
    parser.add_argument('--learning_rate', default=1e-5, type=int, help="")    
    # Other setting
    parser.add_argument("--saveData", action='store_true', help="Save data used in training and evaluating")
    parser.add_argument('--cuda', default=0, type=int, help="cuda")
    args = parser.parse_args()
    
    if isinstance(args.prototype, int):
        prototype = '_prototype'
    else:
        prototype = ''
        
    temp_section = args.section
    # == Load & Prepare data ==
    print("Loading Train data")
    if args.inputType == 'kg':
        train_df = data_importer.prepro_KGData(args, "train", section=args.section)
        val_df = data_importer.prepro_KGData(args, "val", section=args.section)
    else:
        args.section = args.section.split("+")
        train_df = data_importer.prepro_textData(args, "train", section=args.section, skip_null=True)
        val_df = data_importer.prepro_textData(args, "val", section=args.section, skip_null=False)
        args.section = temp_section
    if args.saveData: 
        train_df.to_csv(f"model/trainDataset_{args.section}_{args.inputType}{prototype}.csv")
        val_df.to_csv(f"model/valDataset_{args.section}_{args.inputType}{prototype}.csv")
        
    seed_everything(42, workers=True)
    exp_name = f'{args.model}_{args.section}-{args.inputType}{prototype}'
    
    wandb_logger = WandbLogger(
        project='MuP-project',
        name=exp_name,
    )
    
    wandb_logger.experiment.config.update({
        "model": args.model, 
        "input_section": args.section, 
        "input_type": args.inputType, 
        "batch_size": args.batch_size, 
        "max_input_length": args.max_input_length, 
        "max_output_length": args.max_output_length, 
        "num_beams": args.num_beams, 
        "learning_rate": args.learning_rate, 
    })

    # wandb_logger.experiment.config["input_section"] = args.section
    # wandb_logger.experiment.config["input_type"] = args.inputType
    # wandb_logger.experiment.config["batch_size"] = args.train_batch_size


    data_module = DataModule(train_df, val_df, args)
    model = BartModel(args)
    
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        # filename="best-checkpoint",
        filename=exp_name,
        # save_last=True,
        verbose=True,
        # every_n_epochs=1,
        monitor="rouge_1",
        mode="max",
        # filename_suffix='.pt',
    )
    
    trainer = Trainer(
        logger=wandb_logger,
        # logger=logger,
        callbacks=[checkpoint_callback],        
        max_epochs=args.num_epoch,
        # val_check_interval=1,
        devices=[0], 
        accelerator="gpu"
    )
    
    trainer.fit(model, data_module)
    
    
    wandb.finish()


        
        
if __name__ == "__main__":
    main()
