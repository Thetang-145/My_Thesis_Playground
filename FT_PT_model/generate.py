import os
import sys
import json
import argparse
import pandas as pd
from pathlib import Path

import logging
import timeit
from datetime import datetime, timedelta

import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import LEDForConditionalGeneration, LEDTokenizer
from transformers import AdamW
from rouge import Rouge

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
}
MODELS = {
    "bart-large": "facebook/bart-large",
    "bart-large-cnn": "facebook/bart-large-cnn",
    "led-base": "allenai/longformer-base-4096"
}
MAX_INPUT_LENGTH = 512
MAX_OUPUT_LENGTH = 512
EVAL_BATCH_SIZE  = 8
NUM_BEAMS = 4


def print_progress(curr, full, desc='', bar_size=30):    
    bar = int((curr+1)/full*bar_size)
    sys.stdout.write(f"\r{desc}[{'='*bar}{' '*(bar_size-bar)}] {curr+1}/{full}")
    sys.stdout.flush()
    if curr+1==full: print()
    
def print_log(msg):
    print(msg)
    logging.info(msg)    
            
def getInputDataset(dataset, data_split, section):
    main_path = str((Path().absolute()).parents[0])
    filepath = f"{main_path}/PL-Marker/_scire_models/{dataset}/{section}/{data_split}_re.json"
    print(f"Loading data from: {filepath}")
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

def getInputDF(dataset, data_split, section, prototype):
    input_dataset = getInputDataset(dataset, data_split, section)
    input_dataset_list = []
    for i, data in enumerate(input_dataset):
        tripEnt, freeEnt, tripSeq = getTripSeq(data)
        # for entType in ENT_TYPES:
        freeEntSeq = getFreeEntSeq(freeEnt)
        row = {
            "paper_id": data['doc_key'], 
            "input_seq": tripSeq+freeEntSeq
        }
        input_dataset_list.append(row)
        if isinstance(prototype, int):
            if i>=prototype-1: break
    return pd.DataFrame(input_dataset_list)


def getTargetDF(dataset, data_split, prototype):
    main_path = str((Path().absolute()).parents[0])    
    filepath = f"{main_path}/dataset_MuP/{RAWDATAFILES[data_split]}"
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
        if isinstance(prototype, int):
            if i>=prototype-1: break
    return pd.DataFrame(dataset_list)
    
def removeIssue(df):
    issue_doc = pd.read_csv("issue_data.csv")
    remove_index = []
    for paper_id in list(issue_doc['paper_id']):
        remove_index += list(df[df['paper_id']==paper_id].index)
    return df.drop(index=remove_index)

def prepro_KGData(dataset, data_split, section, prototype):
    input_df = getInputDF(dataset, data_split, section, prototype)
    target_df = getTargetDF(dataset, data_split, prototype)
    # input_df.set_index('paper_id', inplace=True)
    # target_df.set_index('paper_id', inplace=True)
    merged_df = input_df.merge(target_df, how='inner', on='paper_id')
    return removeIssue(merged_df.reset_index())

def getDataset(dataset, data_split, section, prototype):
    main_path = str((Path().absolute()).parents[0])    
    filepath = f"{main_path}/dataset_MuP/{RAWDATAFILES[data_split]}"
    with open(filepath, 'r') as json_file:
        json_list = list(json_file)
    dataset_list = []
    data_len = len(json_list)
    for i, json_str in enumerate(json_list):
        data = json.loads(json_str)
        if section=='abstract':
            dataset_list.append({
                "paper_id": data["paper_id"], 
                "input_seq": data["paper"]["abstractText"], 
                "target_seq": data["summary"]
            })
        elif isinstance(section, int):
            try:
                dataset_list.append({
                    "paper_id": data["paper_id"], 
                    "input_seq": data["paper"]["section"][section-1], 
                    "target_seq": data["summary"]
                })
            except:
                pass
        print_progress(i, data_len, desc=f'Loading summary ({data_split})')
        if isinstance(prototype, int):
            if i>=prototype-1: break
    return pd.DataFrame(dataset_list)

def prepro_textData(data_split, section, prototype):
    dataset_df = getDataset(data_split, section, prototype)
    return removeIssue(dataset_df)

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

def generateSum(model, tokenizer, test_data, model_filename):

    test_dataset = MyDataset(test_data, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE)
    
    bestScores = ['bestRouge1', 'bestRougeAvg']
    for i, best in enumerate(bestScores):
        chechpoint = f'model/{model_filename}_{best}.pt'
        print_log(f"Start generating summary using chechpoint: {chechpoint}")
        model.load_state_dict(torch.load(chechpoint))
        model.to(device=DEVICE)
        model.eval()
        total_loss = 0.0

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
                    num_beams=NUM_BEAMS
                )
                for idx, paper_id in enumerate(paper_id_list):
                    result.append({
                        'paper_id': paper_id,
                        'input': tokenizer.decode(input_ids[idx], skip_special_tokens=True),
                        f'output_{best}': tokenizer.decode(generated_ids[idx], skip_special_tokens=True)
                    })
            if i==0:
                result_df = pd.DataFrame(result)
            else:
                result_df_ = pd.DataFrame(result)
                result_df = pd.merge(result_df, result_df_, on=['paper_id', 'input'], how='outer')
        print_log(f"Finish generating {len(result)} summaries using {chechpoint}")        
    return result_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='bart-large' , type=str, 
                        help="model used for training: bart-large, bart-large-cnn, etc.")
    parser.add_argument('--modelSection', default='abstract' , type=str, help="section to gen summary")
    parser.add_argument('--modelinputType', default='kg' , type=str, help="input types: kg, text")
    parser.add_argument('--evalSection', default=None , type=str, help="")
    parser.add_argument('--evalInputType', default=None, type=str, help="")    
    
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
    
    global DEVICE 
    DEVICE = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(args.cuda) if torch.cuda.is_available() else 'CPU'
    
    MODEL_NAME = MODELS[args.model]
    MODEL_CAT  = args.model.split("-")[0]
    
    if MODEL_CAT=="bart":
        tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
        model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
    elif MODEL_CAT=="led":
        tokenizer = LEDTokenizer.from_pretrained(MODEL_NAME)
        model = LEDForConditionalGeneration.from_pretrained(MODEL_NAME)
    else:
        print("No this model")
        exit()
        

    dataset_dict = {
        "validation": "val",
        "testing"   : "test"
    }
    for k, v in dataset_dict.items():
        if args.evalSection=='summary' and v==test: continue
        
        if args.evalInputType == 'kg':
            eval_data = prepro_KGData("MuP", v, section=args.evalSection, prototype=args.prototype)
        else:
            eval_data = prepro_textData("MuP", v, section=args.evalSection, prototype=args.prototype)

        print_log(f"Start generate summary from {k} dataset")
        result_df = generateSum(model, tokenizer, eval_data, 
                                model_filename=f'{args.model}/{args.modelSection}_{args.modelinputType}')
        result_df = result_df.drop_duplicates()
        csv_file = f"record_result/generated_summary/{v}_MODEL-{args.modelSection}-{args.modelinputType}_EVALL-{args.evalSection}-{args.evalInputType}.csv"
        result_df.to_csv(csv_file)
        print_log(f"Saved {len(result_df)} summaries of {k} dataset to {csv_file}")

        
        
if __name__ == "__main__":
    main()
    print_log("FINISH ALL PROCESSES")