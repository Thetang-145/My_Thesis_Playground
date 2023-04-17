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
    parser.add_argument('--modelSection', default='abstract' , type=str, 
                        help="section used for finetuning: abstract, etc.")
    parser.add_argument('--modelinputType', default='kg' , type=str, 
                        help="input type used for finetuning: kg, text")
    parser.add_argument('--evalSection', default=None , type=str, help="")
    parser.add_argument('--evalInputType', default=None, type=str, help="")
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
    parser.add_argument('--prototype', type=int, help="number of data for prototype run")
    parser.add_argument('--cuda', default=0 , type=int, help="cuda")
    args = parser.parse_args()
    
    now = datetime.now()
    dt_string = now.strftime(f"%y%m%d_%H%M%S")
    log_dir = 'log'
    if not(Path(log_dir).exists()): os.system(f"mkdir -p {log_dir}")
    logging.basicConfig(
        filename=f'{log_dir}/generate_{dt_string}.log',
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s: %(message)s',
    )
    
    if args.evalSection==None: args.evalSection=args.modelSection
    if args.evalInputType==None: args.evalInputType=args.modelinputType
            


    for data_split in ["val", "test"]:
        if args.evalSection=='summary' and data_split=='test': continue
        
        if args.evalInputType == 'kg':
            eval_data = dt_imp.prepro_KGData(args, data_split, section=args.evalSection)
        else:
            eval_data = dt_imp.prepro_textData(args, data_split, section=args.evalSection)
            
        if data_split=='test': eval_data['target_seq'] = "<PAD>"            

        print_log(f"Start generate summary from {data_split} dataset")
        model_filename = f'{args.model}/{args.modelSection}_{args.modelinputType}'
        result_df = ft.generateSum(args, 
                                   eval_data.drop_duplicates(),
                                   model_filename=model_filename)
        result_df = result_df.drop_duplicates()
        csv_dir = f"generated_summary/{data_split}/{args.model}/"
        csv_file = f"MODEL-{args.modelSection}-{args.modelinputType}_EVAL-{args.evalSection}-{args.evalInputType}.csv"
        if not(Path(csv_dir).exists()): os.system(f"mkdir -p {csv_dir}")
        result_df.to_csv(csv_dir+csv_file)
        print_log(f"Saved {len(result_df)} summaries of {data_split} dataset to {csv_file}")

        
        
if __name__ == "__main__":
    main()
    print_log("FINISH ALL PROCESSES")
    
    