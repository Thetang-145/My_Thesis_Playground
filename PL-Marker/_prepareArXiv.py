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
    
def load_data(filepath, section):
    with open(filepath, 'r') as json_file:
        json_list = list(json_file)
    dataset = []
    data_len = len(json_list)
    cant_load = 0
    for i, json_str in enumerate(json_list):
        try:
            line = json.loads(json_str)
            if section=="abstract":
                dataset.append({
                    "doc_key": line["article_id"],
                    "sentences": line["abstract_text"],
                })
            elif section[:7]=="section":
                sec_no = int(section[7:])-1
                dataset.append({
                    "doc_key": line["article_id"],
                    "sentences": line["sections"][sec_no],
                })
        except: 
            cant_load += 1
        print_progress(i, data_len, desc=f'Loading {section} ')
    print(f"\nSuccessfully import {data_len-cant_load}/{data_len} samples")
    return dataset

def drop_outlier(data, limit=200):
    len_data = len(data)
    data = pd.DataFrame(data)
    data['num_sentences'] = [len(sent) for sent in (data['sentences'])]
    data = data[data['num_sentences']<limit]
    data = data.drop('num_sentences', axis=1)
    data = data.to_dict('records')
    print(f"Droped from {len_data} to {len(data)} ({len_data-len(data)} samples)")
    return data

def convert_data(data):
    data = drop_outlier(data)
    num_docs = len(data)
    nlp = spacy.load("en_core_web_sm")
    conv_data = []
    for idx, doc in enumerate(data):
        # doc_text = nlp(doc["sentences"])
        sents, ners, rels = [], [], []
        for sent in doc['sentences']:
            sent = sent.replace("<S> ", "").replace(" </S>", "")
            sent = nlp(sent)
            token_sent = [token.text for token in sent]
            sents.append(token_sent)
            ners.append([])
            rels.append([])
        conv_data.append({
            "doc_key": doc["doc_key"],
            "sentences": sents,
            "ner": ners,
            "relations": rels,
        })
        print_progress(idx, num_docs,  desc='Converting data ')
    print()
    return conv_data

def export_data(data, output_dir, filename, file_size_limit=float('inf')):
    print("Writing prepared data to jsonl file")
    print(output_dir)
    if not(Path(output_dir).exists()): os.system(f"mkdir -p {output_dir}")
    
    # Convert MB to bytes
    if file_size_limit!=float('inf'):
        file_size_limit = int(file_size_limit*pow(1024,2)*0.95)
    
    file_size = 0
    file_number = 1
    recored = 0
    with open(f"{output_dir}{filename}.jsonl", "w") as f:
        for idx, line in enumerate(data):
            json_str = json.dumps(line, ensure_ascii=False)
            file_size += len(json_str.encode('utf-8')) + 1 
            if file_size > file_size_limit:
                f.close()
                if file_number==1:
                    os.rename(f"{output_dir}{filename}.jsonl", f"{output_dir}{filename}_{file_number}.jsonl")    
                print(f'\tWrote {idx-recored} records (({file_size/pow(1024,2):.3f}MB)) to {output_dir}{filename}_{file_number}.')
                recored = idx
                file_number += 1
                file_size = 0
                f = open(f"{output_dir}{filename}_{file_number}.jsonl", "w")
            f.write(json_str + '\n')
    if file_number==1:
        print(f'Successfully wrote {len(data)} records ({file_size/pow(1024,2):.3f}MB) to {output_dir}{filename}')
    else:
        print(f'\tWrote {idx-recored} records (({file_size/pow(1024,2):.3f}MB)) to {output_dir}{filename}_{file_number}.')
        print(f'Successfully wrote {len(data)} records to {file_number} output files')
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", default='abstract', type=str,
                        help="which data to be prepated: abstract, section_1")
    parser.add_argument("--output_dir", default='_prepared_data/arXiv', type=str,
                        help="Directory of prepared data")
    
    parser.add_argument("--train", action='store_true', help="operate on train data")
    parser.add_argument("--val", action='store_true', help="operate on validate data")
    parser.add_argument("--test", action='store_true', help="operate on test data")

    parser.add_argument("--max_filesize", default=50, type=int, help="Maximum output file size (MB)")

    args = parser.parse_args()
    
#     if args.split:
#         output_dir = f"{args.output_dir}/{args.section}/"
        
#         exit()

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
                filename   = f"{key}",
                file_size_limit = args.max_filesize 
            )
        
if __name__ == "__main__":
    main()