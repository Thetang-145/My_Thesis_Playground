import os
import sys
import json
from pathlib import Path
import pandas as pd
import argparse
import timeit

from tqdm import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer, AdamW
import torch
from torch.utils.data import DataLoader, Dataset


ENT_TYPES = ['Task', 'Method', 'Material', 'Metric', 'OtherScientificTerm', 'Generic']
REL_TYPES = {
    'FEATURE-OF': 'Asym', 
    'PART-OF': 'Asym', 
    'USED-FOR': 'Asym', 
    'EVALUATE-FOR': 'Asym',
    'HYPONYM-OF': 'Asym', 
    'COMPARE': 'Sym',
    'CONJUNCTION': 'Sym',
}
RAWDATAFILES = {
    "train": "training_complete.jsonl",
    "val": "validation_complete.jsonl",
    "test": "testing_with_paper_release.jsonl"
}
# Define hyperparameters
MAX_INPUT_LENGTH = 512
MAX_OUPUT_LENGTH = 512
TRIAN_BATCH_SIZE = 8
EVAL_BATCH_SIZE  = 8
NUM_EPOCH  = 5
LEARNING_RATE = 1e-5
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def print_progress(curr, full, desc='', bar_size=30):    
    bar = int((curr+1)/full*bar_size)
    sys.stdout.write(f"\r{desc}[{'='*bar}{' '*(bar_size-bar)}] {curr+1}/{full}")
    sys.stdout.flush()
    if curr+1==full: print()
    

def getInputDataset(data_split, section):
    main_path = str((Path().absolute()).parents[0])
    filepath = f"{main_path}/PL-Marker/_scire_models/{section}/{data_split}_re.json"
    with open(filepath, 'r') as json_file:
        json_list = list(json_file)
    return [json.loads(json_str) for json_str in json_list]

def addTripEnt(allEnt, tripEnt, ent):
    try:
        tripEnt[ent] = allEnt[ent]
    except:
        tripEnt[ent] = None
    return tripEnt

def getTripSeq(data, sym=True):
    all_sentences = [j for i in data["sentences"] for j in i]
    flatten_ner = [j for i in data["predicted_ner"] for j in i]
    flatten_re  = [j for i in data["predicted_re"] for j in i[1]]
    tripSeq = ""
    tripEnt = {}
    allEnt = {}

    for ner in flatten_ner:
        ent = " ".join(all_sentences[ner[0]:ner[1]+1])
        allEnt[ent] = ner[2]

    for rel in flatten_re:
        s = " ".join(all_sentences[rel[0][0]:(rel[0][1]+1)])
        o = " ".join(all_sentences[rel[1][0]:(rel[1][1]+1)])
        p = rel[2]
        tripSeq += f"{s} {p} {o}. "
        if sym and REL_TYPES[p]=='Sym':
            tripSeq += f"{o} {p} {s}. "
        tripEnt = addTripEnt(allEnt, tripEnt, s)
        tripEnt = addTripEnt(allEnt, tripEnt, o)
        
    freeEnt = allEnt.copy()
    for key, val in tripEnt.items(): freeEnt.pop(key)

    return tripEnt, freeEnt, tripSeq

def getFreeEntSeq(freeEnt):
    freeEntList = {}
    for k, v in freeEnt.items():
        if v not in list(freeEntList.keys()): 
            freeEntList[v] = [k]
        else:
            freeEntList[v].append(k)
    freeEntSeq = ""
    for entType, ents in freeEntList.items():
        if len(ents)==1:
            freeEntSeq += f"{entType} is {ents[0]}. "
        else:
            freeEntSeq += f"{entType} are {', '.join(ents[:-1])}"
            freeEntSeq += f", and {ents[-1]}. "
    return freeEntSeq

def getInputDF(data_split, section, prototype):
    dataset = getInputDataset(data_split, section)
    dataset_list = []
    for i, data in enumerate(dataset):
        tripEnt, freeEnt, tripSeq = getTripSeq(data)
        # for entType in ENT_TYPES:
        freeEntSeq = getFreeEntSeq(freeEnt)
        row = {
            "paper_id": data['doc_key'], 
            "input_seq": tripSeq+freeEntSeq
        }
        dataset_list.append(row)
        if isinstance(prototype, int):
            if i>=prototype-1: break
    return pd.DataFrame(dataset_list)


def getTargetDF(data_split):
    main_path = str((Path().absolute()).parents[0])    
    filepath = f"{main_path}/MuP_sum/dataset/{RAWDATAFILES[data_split]}"
    with open(filepath, 'r') as json_file:
        json_list = list(json_file)
    dataset_list = []
    data_len = len(json_list)
    for i, json_str in enumerate(json_list):
        data = json.loads(json_str)
        dataset_list.append({
            "paper_id": data["paper_id"], 
            "target_seq": data["summary"]
        })   
        print_progress(i, data_len, desc=f'Loading summary ({data_split})')
    return pd.DataFrame(dataset_list)
    
    

def prepro_data(data_split, section, prototype):
    input_df = getInputDF(data_split, section, prototype)
    target_df = getTargetDF(data_split)
    merged_df = pd.merge(input_df, target_df, on='paper_id')
    
    # remove paper with issue
    issue_doc = pd.read_csv("issue_data.csv")
    remove_index = []
    for paper_id in list(issue_doc['paper_id']):
        remove_index += list(merged_df[merged_df['paper_id']==paper_id].index)
    remove_index
    merged_df_drop = merged_df.drop(index=remove_index)
    return merged_df_drop

class MyDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.paper_id= list(data['paper_id'])
        self.input = list(data['input_seq'])
        self.target = list(data['target_seq'])
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        input_tokens = self.tokenizer.encode(self.input[index], padding='max_length', max_length=MAX_INPUT_LENGTH, truncation=True, return_tensors='pt')
        target_tokens = self.tokenizer.encode(self.target[index], padding='max_length', max_length=MAX_OUPUT_LENGTH, truncation=True, return_tensors='pt')
        
        paper_id = self.paper_id[index]
        input_ids = input_tokens.squeeze()
        input_attention_mask = torch.ones(input_ids.shape)
        target_ids = target_tokens.squeeze()
        target_attention_mask = torch.ones(target_ids.shape)
                
        return paper_id, input_ids, input_attention_mask, target_ids

    def __len__(self):
        return len(self.data)

def train(model, tokenizer, train_data):
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    train_dataset = MyDataset(train_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=TRIAN_BATCH_SIZE)
    
    model.train()
    for epoch in range(NUM_EPOCH):
        for batch in tqdm(train_loader, desc=f"Fine-tuning epoch {epoch+1}"):
            paper_id, input_ids, attention_mask, target_ids = batch
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            target_ids = target_ids.to(DEVICE)

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=target_ids)
            loss = outputs.loss

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    torch.save(model.state_dict(), f'model/finetune_full.pt')
    return

def generateSummary(model, tokenizer, test_data):
    model.load_state_dict(torch.load('model/finetune_full.pt'))

    model.to(DEVICE)
    model.eval()
    total_loss = 0.0
    
    test_dataset = MyDataset(test_data, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE)
    
    start_time = timeit.default_timer() 
    with torch.no_grad():
        result = []
        for batch in tqdm(test_loader, desc=f"Summarizing"):
            paper_id_list, input_ids, attention_mask, target_ids = batch
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            target_ids = target_ids.to(DEVICE)

            generated_ids = model.generate(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                max_length=tokenizer.model_max_length, 
                num_beams=4
            )
            for idx, paper_id in enumerate(paper_id_list):
                result.append({
                    'paper_id': paper_id,
                    'gen_sum': tokenizer.decode(generated_ids[idx], skip_special_tokens=True)
                })
    result_df = pd.DataFrame(result)            
    return result_df



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true', help="train model")
    parser.add_argument("--getSum", action='store_true', help="get generated summary from fine-tuned model")
    parser.add_argument("--eval", action='store_true', help="evaluate generated summary")
    parser.add_argument('--prototype', type=int, help="number of data for prototype run")
    args = parser.parse_args()
    
            
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    
    if args.train:
        train_data = prepro_data("train", section="abstract", prototype=args.prototype)
        train(model, tokenizer, train_data)
    
    if args.getSum:
        eval_data = prepro_data("val", section="abstract", prototype=args.prototype)
        result_df = generateSummary(model, tokenizer, eval_data)
        result_df.to_csv("model/result.csv")
        
    if args.eval:
        
        
if __name__ == "__main__":
    main()