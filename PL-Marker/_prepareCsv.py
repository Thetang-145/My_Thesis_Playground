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
    
def load_data(filepath, doc_col, sentence_col):
    df = pd.read_csv(filepath, index_col=0)
    doc_key = list(df[doc_col])
    sentences = list(df[sentence_col])
    dataset= [{
        "doc_key": doc_key[i],
        "sentences": sentences[i],
    } for i in range(len(doc_key))]
    print(f"\nSuccessfully import {len(dataset)} samples")
    return dataset

def convert_data(data):
    num_docs = len(data)
    nlp = spacy.load("en_core_web_sm")
    for idx, doc in enumerate(data):
        doc_text = nlp(doc["sentences"])
        sents, ners, rels = [], [], []
        for sent in doc_text.sents:
            token_sent = [str(tok) for tok in sent]
            sents.append(token_sent)
            ners.append([])
            rels.append([])
        print_progress(idx, num_docs,  desc='Converting data ')
        doc["sentences"] = sents
        doc["ner"] = ners
        doc["relations"] = rels
    print()
    return data

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
    parser.add_argument("--file_path",  required=True, type=str,
                        help="File path of input data")
    parser.add_argument("--doc_col",  required=True, type=str, help="")
    parser.add_argument("--sentence_col",  required=True, type=str, help="")
    
    parser.add_argument("--output_filename", type=str, help="Directory of prepared data")   
    parser.add_argument("--output_dir", default='_prepared_data/csv', type=str, help="Directory of prepared data")    
    args = parser.parse_args()
    
    
    main_path = str((Path().absolute()).parents[0])
    dataset_path = f"{main_path}/{args.file_path}"
    # dataset_path = args.file_path
    
    if args.output_filename is None:
        args.output_filename = args.file_path.split("/")[-1][:-4]
        print(args.output_filename)
    
    
    print(f"{'*'*25} Processing csv data {'*'*25}")
    print(f"from {dataset_path}")
    data = load_data(dataset_path, args.doc_col, args.sentence_col)
    data = convert_data(data)
    export_data(
        data, 
        output_dir = f"{args.output_dir}/", 
        filename   = f"{args.output_filename}.jsonl"
    )
    
        
if __name__ == "__main__":
    main()