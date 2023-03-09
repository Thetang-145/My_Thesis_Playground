import os
import sys
import json
from pathlib import Path
import spacy

DATAFILES = {
    "train": "training_complete.jsonl",
    "val": "validation_complete.jsonl",
    "test": "testing_with_paper_release.jsonl"
}

def print_progress(curr, full, bar_size=50):    
    bar = int((curr+1)/full*bar_size)
    sys.stdout.write(f"\r[{'='*bar}{' '*(bar_size-bar)}] {curr+1}/{full}")
    sys.stdout.flush()
    

def load_data(filepath):
    print("Loading data")
    with open(filepath, 'r') as json_file:
        json_list = list(json_file)
    dataset = []
    data_len = len(json_list)
    for i, json_str in enumerate(json_list):
        result = json.loads(json_str)
        dataset.append({
            "doc_key": result["paper_id"], 
            "sentences": result["summary"]
        })
        print_progress(i, data_len)
        # break
    print()
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

def write_jsonl(data, output_path, filename,append=False):
    print("Writing data to jsonl file")
    mode = 'a+' if append else 'w'
    if not(Path(output_path).exists()): os.mkdir(output_path)
    with open(output_path+filename, mode, encoding='utf-8') as f:
        for line in data:
            json_record = json.dumps(line, ensure_ascii=False)
            f.write(json_record + '\n')
    print('Wrote {} records to {}'.format(len(data), output_path))
    

def main():
    main_path = str((Path().absolute()).parents[0])
    dataset_path = main_path+"/MuP_sum/dataset/"
    for key, val in DATAFILES.items():
        if key == "test":
             continue
        print(f"{'*'*25} Processing {key} data {'*'*25}")
        data = load_data(dataset_path+val)
        data = convert_data(data)
        write_jsonl(data, f"{main_path}/PL-Marker/_prepared_data/", f"{key}.jsonl")
        
    


if __name__ == "__main__":
    main()