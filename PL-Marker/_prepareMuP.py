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

def print_progress(curr, full, desc='', bar_size=50):    
    bar = int((curr+1)/full*bar_size)
    sys.stdout.write(f"\r{desc}[{'='*bar}{' '*(bar_size-bar)}] {curr+1}/{full}")
    sys.stdout.flush()
    

def load_data(filepath, section):
    with open(filepath, 'r') as json_file:
        json_list = list(json_file)
    dataset = []
    data_len = len(json_list)
    cannot_import = 0
    if section == "summary":
        for i, json_str in enumerate(json_list):
            result = json.loads(json_str)
            try:
                dataset.append({
                    "doc_key": result["paper_id"], 
                    "sentences": result["summary"]
                })   
            except:                
                cannot_import += 1          
            print_progress(i, data_len, desc=f'Loading {section} ')
    elif section == "abstract":
        paper_id_list = []
        for i, json_str in enumerate(json_list):
            result = json.loads(json_str)
            try:
                if result["paper_id"] in paper_id_list: continue
                dataset.append({
                    "doc_key": result["paper_id"], 
                    "sentences": result["paper"]["abstractText"]
                })
                paper_id_list.append(result["paper_id"])
            except:                
                cannot_import += 1
            print_progress(i, data_len, desc=f'Loading {section} ')
    elif section[:8] == "section_":
        heading = section[8:]
        paper_id_list = []
        try:
            heading = int(heading)
            for i, json_str in enumerate(json_list):
                result = json.loads(json_str)
                try:
                    if result["paper_id"] in paper_id_list: continue
                    sentences = result["paper"]["sections"][heading-1]['text']
                    n=0
                    while sentences=="":
                        n+=1
                        sentences = result["paper"]["sections"][heading-1+n]['text']
                    dataset.append({
                        "doc_key": result["paper_id"], 
                        # "heading": result["paper"]["sections"][heading-1+n]['heading'],
                        "sentences": sentences
                    })   
                    paper_id_list.append(result["paper_id"])         
                except:                
                    cannot_import += 1
                print_progress(i, data_len, desc=f'Loading {section[:7]} {heading} ')
        except:
            for i, json_str in enumerate(json_list):
                result = json.loads(json_str)
                try:
                    if result["paper_id"] in paper_id_list: continue
                    for dict_section in (result["paper"]["sections"]):
                        if dict_section['heading'].lower().find(heading) != -1: 
                            sentences = dict_section['text']
                            if sentences=="":
                                cannot_import += 1
                                continue
                            dataset.append({
                                "doc_key": result["paper_id"], 
                                # "heading": dict_section['heading'],
                                "sentences": sentences
                            })
                            break
                    paper_id_list.append(result["paper_id"])   
                except:                
                    cannot_import += 1
                print_progress(i, data_len, desc=f'Loading {heading} ')
    else:
        raise Exception("section is not correct")
    print(f"\nSuccessfully import {data_len-cannot_import}/{data_len} samples")
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
    if not(Path(output_dir).exists()): os.mkdir(output_dir)
    with open(output_dir+filename, mode, encoding='utf-8') as f:
        for line in data:
            json_record = json.dumps(line, ensure_ascii=False)
            f.write(json_record + '\n')
    print('Wrote {} records to {}'.format(len(data), output_dir+filename))
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", default='summary', type=str,
                        help="which data to be prepated: summary, abstract, 1st_sec")
    parser.add_argument("--output_dir", default='_prepared_data/', type=str,
                        help="Directory of prepared data")
    
    parser.add_argument("--train", action='store_true', help="operate on train data")
    parser.add_argument("--val", action='store_true', help="operate on validate data")
    parser.add_argument("--test", action='store_true', help="operate on test data")
    
    args = parser.parse_args()

    if not (args.train or args.val or args.test):
        args.train = args.val = args.test = True
    if args.section == 'summary': args.test = True
    
    
    main_path = str((Path().absolute()).parents[0])
    dataset_path = main_path+"/MuP_sum/dataset/"
    
    if args.section == 'summary': args.test = False
    
    for key, val in DATAFILES.items():
        if key=="test" and not(args.test): continue
        if key=="train" and not(args.train): continue
        if key=="val" and not(args.val): continue
        
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