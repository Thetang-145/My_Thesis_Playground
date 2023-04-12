import common.finetune as ft
import common.data_importer as dt_imp

import os
import sys
import json
import logging
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

    
def print_log(msg):
    print(msg)
    logging.info(msg)    
    
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
    parser.add_argument('--train_batch_size', default=8, type=int, help="")    
    parser.add_argument('--val_batch_size', default=8, type=int, help="")    
    parser.add_argument('--eval_batch_size', default=8, type=int, help="")    
    parser.add_argument('--eval_frequency', default=1000, type=int, help="")    
    parser.add_argument('--num_epoch', default=5, type=int, help="")    
    parser.add_argument('--num_beams', default=4, type=int, help="")    
    parser.add_argument('--learning_rate', default=1e-5, type=int, help="")    
    # Other setting
    parser.add_argument("--saveData", action='store_true', help="Save data used in training and evaluating")
    parser.add_argument('--cuda', default=0, type=int, help="cuda")
    args = parser.parse_args()
    
    now = datetime.now()
    dt_string = now.strftime(f"%y%m%d_%H%M%S")
    log_dir = 'log'
    if not(Path(log_dir).exists()): os.system(f"mkdir -p {log_dir}")
    logging.basicConfig(
        filename=f'{log_dir}/finetune_{dt_string}.log',
        level=logging.INFO,
        format='%(asctime)s:%(name)s:%(levelname)s: %(message)s',
    )
    
    if isinstance(args.prototype, int):
        prototype = '_prototype'
    else:
        prototype = ''

    print("Loading Train data")
    logging.info(f"Loading {args.section} {args.inputType} from {args.dataset} dataset")
    if args.inputType == 'kg':
        train_data = dt_imp.prepro_KGData(args, "train")
        val_data = dt_imp.prepro_KGData(args, "val")
    else:
        train_data = dt_imp.prepro_textData(args, "train")
        val_data = dt_imp.prepro_textData(args, "val")
    if args.saveData: 
        train_data.to_csv(f"model/trainDataset_{args.section}_{args.inputType}{prototype}.csv")
        val_data.to_csv(f"model/valDataset_{args.section}_{args.inputType}{prototype}.csv")

#     logging.info(f"Start training on {DEVICE} ({device_name})")
    logging.info(f"All parameters: {args}")
    modelSave_dir = f"model/{args.model}"
    if not(Path(modelSave_dir).exists()): os.system(f"mkdir -p {modelSave_dir}")
    trainRec = ft.train(
        args, train_data, val_data,
        model_filename=f'{args.model}/{args.section}_{args.inputType}{prototype}'
    )
    trainRec.to_csv(f"record_result/train_record/{args.model}_{args.section}_{args.inputType}{prototype}.csv")


        
        
if __name__ == "__main__":
    main()
    print_log("FINISH ALL PROCESSES")