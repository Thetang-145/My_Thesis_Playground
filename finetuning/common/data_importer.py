import sys
import json
import pandas as pd
from pathlib import Path
import logging

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

def print_log(msg):
    print(msg)
    logging.info(msg)    

def print_progress(curr, full, desc='', bar_size=30):    
    bar = int((curr+1)/full*bar_size)
    sys.stdout.write(f"\r{desc}[{'='*bar}{' '*(bar_size-bar)}] {curr+1}/{full}")
    sys.stdout.flush()
    if curr+1==full: print()

            
def getInputDataset(args, data_split, section):
    main_path = str((Path().absolute()).parents[0])
    filepath = f"{main_path}/PL-Marker/_scire_models/{args.dataset}/{section}/{data_split}_re.json"
    print(f"Loading data from: {filepath}")
    with open(filepath, 'r') as json_file:
        json_list = list(json_file)
    logging.info(f"Loaded {len(json_list)} kg inputs from {filepath}")
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

def getInputDF(args, data_split, section):
    input_dataset = getInputDataset(args, data_split, section)
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
        if isinstance(args.prototype, int):
            if i>=args.prototype-1: break
    logging.info(f"Preprocessed {len(input_dataset_list)} inputs")
    return pd.DataFrame(input_dataset_list)


def getTargetDF(args, data_split):
    main_path = str((Path().absolute()).parents[0])    
    filepath = f"{main_path}/dataset_MuP/{RAWDATAFILES[data_split]}"
    with open(filepath, 'r') as json_file:
        json_list = list(json_file)
    dataset_list = []
    if isinstance(args.prototype, int): data_len = args.prototype 
    else: data_len = len(json_list)
    for i, json_str in enumerate(json_list):
        data = json.loads(json_str)
        dataset_list.append({
            "paper_id": data["paper_id"], 
            "target_seq": data["summary"]
        })
        print_progress(i, data_len, desc=f'Loading summary ({data_split})')
        if isinstance(args.prototype, int):
            if i>=args.prototype-1: break
    logging.info(f"Loaded {len(dataset_list)} targets from {filepath}")
    return pd.DataFrame(dataset_list)
    
def removeIssue(df):
    issue_doc = pd.read_csv("issue_data.csv")
    remove_index = []
    for paper_id in list(issue_doc['paper_id']):
        remove_index += list(df[df['paper_id']==paper_id].index)
    logging.info(f"Removed {len(remove_index)} issue data")
    return df.drop(index=remove_index)

def prepro_KGData(args, data_split, section):
    input_df = getInputDF(args, data_split, section)
    target_df = getTargetDF(args, data_split)
    # input_df.set_index('paper_id', inplace=True)
    # target_df.set_index('paper_id', inplace=True)
    merged_df = input_df.merge(target_df, how='inner', on='paper_id')
    logging.info(f"Merge input and target to {len(merged_df)} samples")
    return removeIssue(merged_df.reset_index())

def getDataset(args, data_split, section):
    main_path = str((Path().absolute()).parents[0])    
    filepath = f"{main_path}/dataset_MuP/{RAWDATAFILES[data_split]}"
    with open(filepath, 'r') as json_file:
        json_list = list(json_file)
    dataset_list = []
    if isinstance(args.prototype, int): data_len = args.prototype 
    else: data_len = len(json_list)
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
        if isinstance(args.prototype, int):
            if i>=args.prototype-1: break
    return pd.DataFrame(dataset_list)

def prepro_textData(args, data_split, section):
    dataset_df = getDataset(args, data_split, section)
    logging.info(f"Loaded and Finished preprocessing {len(dataset_df)} text data")
    return removeIssue(dataset_df)