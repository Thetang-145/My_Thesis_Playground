import os
import sys
import json
import pandas as pd
from pathlib import Path
import spacy
import argparse

def print_progress(curr, full, desc='', bar_size=50):    
    bar = int((curr+1)/full*bar_size)
    sys.stdout.write(f"\r{desc}[{'='*bar}{' '*(bar_size-bar)}] {curr+1}/{full}")
    sys.stdout.flush()
    
def load_data(filepath, column):

    print(f"\nSuccessfully import {data_len-cant_load}/{data_len} samples")
    return dataset

def convert_data(data):
    num_docs = len(data)
    nlp = spacy.load("en_core_web_sm")
    conv_data = []
    for idx, doc in enumerate(data):
        conv_data.append({})
        doc_text = nlp(doc["sentences"])
        sents, ners, rels = [], [], []
        for sent in doc_text.sents:
            token_sent = [str(tok) for tok in sent]
            sents.append(token_sent)
            ners.append([])
            rels.append([])
        print_progress(idx, num_docs,  desc='Converting data ')
        conv_data[idx]["doc_key"] = doc["doc_key"]
        conv_data[idx]["sentences"] = sents
        conv_data[idx]["ner"] = ners
        conv_data[idx]["relations"] = rels
    print()
    return conv_data

def export_data(data, output_dir, filename, append=False):
    print("Writing prepared data to jsonl file")
    mode = 'a+' if append else 'w'
    print(output_dir)
    if not(Path(output_dir).exists()): 
        os.system(f"mkdir -p {output_dir}")
        # os.mkdir(output_dir, 0755)
    with open(output_dir+filename, mode, encoding='utf-8') as f:
        for line in data:
            json_record = json.dumps(line, ensure_ascii=False)
            f.write(json_record + '\n')
    print('Wrote {} records to {}'.format(len(data), output_dir+filename))
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", default='abstract', type=str,
                        help="which data to be prepated: abstract, section_1")
    parser.add_argument("--output_dir", default='_prepared_data/arXiv', type=str,
                        help="Directory of prepared data")
    
    parser.add_argument("--train", action='store_true', help="operate on train data")
    parser.add_argument("--val", action='store_true', help="operate on validate data")
    parser.add_argument("--test", action='store_true', help="operate on test data")

    parser.add_argument("--split", action='store_true', help="split training set to smaller subfiles")
    
    args = parser.parse_args()
    
    if args.split:
        output_dir = f"{args.output_dir}/{args.section}/"
        
        exit()

    if not (args.train or args.val or args.test):
        args.train = args.val = args.test = True
        
    runPrepare = {
        "train": args.train,
        "val": args.val,
        "test": args.test,
    }
        
    
    main_path = str((Path().absolute()).parents[0])
    dataset_path = main_path+"/dataset_arXiv/"
    
    
    for key, val in runPrepare.items():
        if val:
            print(f"{'*'*25} Processing {key} data {'*'*25}")
            data = load_data(f"{dataset_path}{key}.txt", args.section)
            data = convert_data(data)
            export_data(
                data, 
                output_dir = f"{args.output_dir}/{args.section}/", 
                filename   = f"{key}.jsonl"
            )
        
if __name__ == "__main__":
    main()