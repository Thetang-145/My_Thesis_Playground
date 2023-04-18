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

def getTripSeq(data, sym=False):
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


def main():
    input_df = getInputDF(args, data_split, section)
    pass

if __name__ == "__main__":
    main()