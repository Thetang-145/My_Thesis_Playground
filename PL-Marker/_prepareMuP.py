import os
import sys
import json
from pathlib import Path
import spacy
import argparse

DATAFILES = {
    "train": "training_complete.jsonl",
    "val": "validation_complete.jsonl",
    "test": "testing_with_paper_release.jsonl"
}

def print_progress(curr, full, bar_size=50):    
    bar = int((curr+1)/full*bar_size)
    sys.stdout.write(f"\r[{'='*bar}{' '*(bar_size-bar)}] {curr+1}/{full}")
    sys.stdout.flush()
    

def load_data(filepath, section):
    print(f"Loading data")
    with open(filepath, 'r') as json_file:
        json_list = list(json_file)
    dataset = []
    data_len = len(json_list)
    if section == "summary":
        for i, json_str in enumerate(json_list):
            result = json.loads(json_str)
            dataset.append({
                "doc_key": result["paper_id"], 
                "sentences": result["summary"]
            })            
            print_progress(i, data_len)
    elif section == "abstract":
        for i, json_str in enumerate(json_list):
            result = json.loads(json_str)
            dataset.append({
                "doc_key": result["paper_id"], 
                "sentences": result["paper"]["abstractText"]
            })            
            print_progress(i, data_len)
    elif section[:8] == "section_":
        heading = section[8:]
        cannot_import = 0
        try:
            heading = int(heading)
            for i, json_str in enumerate(json_list):
                result = json.loads(json_str)
                try:
                    dataset.append({
                        "doc_key": result["paper_id"], 
                        "sentences": result["paper"]["section"][heading]['text']
                    })            
                except:                
                    cannot_import += 1
                print_progress(i, data_len)
        except:
            for i, json_str in enumerate(json_list):
                result = json.loads(json_str)
                try:
                    for dict_section in (result["paper"]["sections"]):
                        if dict_section['heading'].lower().find(heading) != -1: 
                            dataset.append({
                                "doc_key": result["paper_id"], 
                                "sentences": dict_section['text']
                            })
                            break
                except:                
                    cannot_import += 1
                print_progress(i, data_len)
    else:
        raise Exception("section is not correct")
        print(f"Successfully import {data_len-cannot_import}/{data_len} samples")
    return dataset

def convert_data(data):
    print("Converting data")
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
        print_progress(idx, num_docs)
        doc["sentences"] = sents
        doc["ner"] = ners
        doc["relations"] = rels
    print()
    return data

def export_data(data, output_dir, filename, append=False):
    print("Writing prepared data to jsonl file")
    mode = 'a+' if append else 'w'
    if not(Path(output_dir).exists()): os.mkdir(output_dir)
    with open(output_dir+filename, mode, encoding='utf-8') as f:
        for line in data:
            json_record = json.dumps(line, ensure_ascii=False)
            f.write(json_record + '\n')
    print('Wrote {} records to {}'.format(len(data), output_dir+filename))
    

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--section", default='summary', type=str, required=True,
                        help="which data to be prepated: summary, abstract, 1st_sec")
    parser.add_argument("--output_dir", default='/_prepared_data/', type=str, required=True,
                        help="Directory of prepared data")
    args = parser.parse_args()

    
    
    main_path = str((Path().absolute()).parents[0])
    dataset_path = main_path+"/MuP_sum/dataset/"
    for key, val in DATAFILES.items():
        if key=="test" and args.section=='summary':
             continue
        print(f"{'*'*25} Processing {key} data {'*'*25}")
        data = load_data(dataset_path+val, args.section)
        data = convert_data(data)
        export_data(
            data, 
            output_dir = f"{args.output_dir}/{args.section}/", 
            filename   = f"{key}.jsonl"
        )
        
if __name__ == "__main__":
    main()