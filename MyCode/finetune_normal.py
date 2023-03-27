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



def getDataset(data_split):
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
            "input_seq": data["paper"]["abstractText"], 
            "target_seq": data["summary"]
        })   
        print_progress(i, data_len, desc=f'Loading summary ({data_split})')
    return pd.DataFrame(dataset_list)
    
    

def prepro_data(data_split):
    dataset_df = getDataset(data_split)
    
    # remove paper with issue
    issue_doc = pd.read_csv("issue_data.csv")
    remove_index = []
    for paper_id in list(issue_doc['paper_id']):
        remove_index += list(dataset_df[dataset_df['paper_id']==paper_id].index)
    dataset_df_drop = dataset_df.drop(index=remove_index)
    return dataset_df_drop

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
    args = parser.parse_args()
    
            
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    
    if args.train:
        train_data = prepro_data("train")
        train(model, tokenizer, train_data)
    
    if args.getSum:
        eval_data = prepro_data("val")
        result_df = generateSummary(model, tokenizer, eval_data)
        result_df.to_csv("model/result_fullAbstract.csv")
        
    if args.eval:
        pass
        
if __name__ == "__main__":
    main()