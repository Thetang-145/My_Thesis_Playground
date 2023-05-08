import os
import sys
import json
import pandas as pd
from pathlib import Path

RAWDATAFILES = {
    "train": "training_complete.jsonl",
    "val": "validation_complete.jsonl",
    "test": "testing_with_paper_release.jsonl"
}
ENT_TYPES = ['Task', 'Method', 'Metric', 'Material', 'OtherScientificTerm', 'Generic']
REL_TYPES = {
    'FEATURE-OF':   'Asym', 
    'PART-OF':      'Asym', 
    'USED-FOR':     'Asym', 
    'EVALUATE-FOR': 'Asym',
    'HYPONYM-OF':   'Asym', 
    'COMPARE':      'Sym',
    'CONJUNCTION':  'Sym',
}
ENT_TYPE_TOKENS = [f"[{prefix}{ent_type}]" for ent_type in ENT_TYPES for prefix in ["", "/"]]
REL_TYPE_TOKENS = [f"[{rel_type}]" for rel_type in REL_TYPES.keys()]

# Setting
FREE_ENT             = True
REVERSE_SYM_RELATION = False

def print_progress(curr, full, desc='', bar_size=30):    
    bar = int((curr+1)/full*bar_size)
    sys.stdout.write(f"\r{desc}[{'='*bar}{' '*(bar_size-bar)}] {curr+1}/{full}")
    # sys.stdout.flush()
    if curr+1==full: print()

# =================================== GET TARGET DATA & COMMON FN ===================================
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
        if data_split=='test': summary = "<PAD>" 
        else: summary = data["summary"]
        dataset_list.append({
            "paper_id": data["paper_id"], 
            "target_seq": summary
        })
        print_progress(i, data_len, desc=f'Loading summary ({data_split})')
        if isinstance(args.prototype, int):
            if i>=args.prototype-1: break
    return pd.DataFrame(dataset_list)
    
def removeIssue(df):
    issue_doc = pd.read_csv("issue_data.csv")
    remove_index = []
    for paper_id in list(issue_doc['paper_id']):
        remove_index += list(df[df['paper_id']==paper_id].index)
    return df.drop(index=remove_index)    
    
# =================================== GET KG DATA ===================================
def get_multifile(prepared_dir, data_split, suffix='', revised=True):
    files = sorted(os.listdir(prepared_dir))
    run_files = []
    if revised:
        for file in files:
            filename = file.split(".")[0]
            if filename[:-(2+len(suffix))]==f"revised_{data_split}_": 
                run_files.append(file)
    else:
        for file in files:
            filename = file.split(".")[0]    
            if filename[:-(2+len(suffix))]==f"{data_split}_" and filename[:-(2+len(suffix))]!=f"revised_{data_split}_": 
                run_files.append(file)
    return run_files

def get_KGdata(args, data_split, section):
    main_path = str((Path().absolute()).parents[0])
    data_dir = f"{main_path}/PL-Marker/_scire_models/{args.dataset}/{section}/"
    if args.dataset=='arXiv' and data_split=='train':
        run_files = get_multifile(data_dir, data_split, suffix="_re",revised=True)
    elif args.dataset=='MuP' and section in ['introduction', 'conclusion']:
        run_files = [f"revised_{data_split}_re.json"]
    else:
        run_files = [f"{data_split}_re.json"]
        
    json_list = []
    num_files = len(run_files)
    for i, filename in enumerate(run_files):
        with open(data_dir+filename, 'r') as json_file:
            json_list += list(json_file)
        if isinstance(args.prototype, int): 
            if len(json_list)>=args.prototype: break
    return [json.loads(json_str) for json_str in json_list]

# ========== KG PROCESSES ==========
def manage_confilt_ent(ent_list):
    count_ent = {}
    for ent_type in ENT_TYPES: count_ent[ent_type] = 0
    for ent in ent_list: count_ent[ent] += 1
    return max(count_ent, key=count_ent.get)

def count_list(input_list):
    count_dict = {}
    for ele in input_list:
        if ele in count_dict.keys(): count_dict[ele] += 1
        else: count_dict[ele] = 1
    return count_dict
    
def build_graph(data, sections):
    ent_type_dict = {}
    triple_list = []
    for sec in sections:
        if not isinstance(data[f"{sec}_sent"], list): break
        flatten_sent = [j for i in data[f"{sec}_sent"] for j in i]
        flatten_ner  = [j for i in data[f"{sec}_ner"] for j in i]
        flatten_re   = [j for i in data[f"{sec}_rel"] for j in i[1]]
        # print(" ".join(flatten_sent))

        for ner in flatten_ner:
            ent = " ".join(flatten_sent[ner[0]:ner[1]+1])
            if ent not in ent_type_dict.keys():
                ent_type_dict[ent] = [ner[2]]
            else:
                ent_type_dict[ent].append(ner[2])
        
        for rel in flatten_re:
            sub = " ".join(flatten_sent[rel[0][0]:rel[0][1]+1])
            obj = " ".join(flatten_sent[rel[1][0]:rel[1][1]+1])
            triple_list.append({
                'subject':  " ".join(flatten_sent[rel[0][0]:rel[0][1]+1]),
                'relation': rel[2],
                'object':   " ".join(flatten_sent[rel[1][0]:rel[1][1]+1]),
            })
            
    # Choose the most frequence type (If there is any conflict on ent types)
    for ent, ent_type in ent_type_dict.items():
        if len(set(ent_type))>1:
            ent_type_dict[ent] = manage_confilt_ent(ent_type)
        else:
            ent_type_dict[ent] = ent_type[0]
    
    return ent_type_dict, pd.DataFrame(triple_list).drop_duplicates()

def add_entType(ent, ent_type):
    return f"[{ent_type}] {ent} [/{ent_type}]"

def graph2seq(ent_type_dict, triple_df, is_add_entType=True):
    seq = ""
    triple_ent = []
    
    while len(triple_df)!=0:
        count = count_list(list(triple_df['subject']))
        sub = max(count, key=count.get)
        sub_seq = add_entType(sub, ent_type_dict[sub]) if is_add_entType else sub
        seq += f"{sub_seq} "            
            
        rel_obj = {}
        for _, row in triple_df[triple_df['subject']==sub].iterrows():
            triple_ent += [row['subject'], row['object']]
            rel = row['relation']
            obj_seq = add_entType(row['object'], ent_type_dict[row['object']]) if is_add_entType else row['object']
            if rel not in rel_obj.keys():
                rel_obj[rel] = [obj_seq]
            else:
                rel_obj[rel].append(obj_seq)
            triple_df.drop(_, inplace=True)
        seq += ', '.join([f"[{rel}] {','.join(objs)}" for rel, objs in rel_obj.items()])
        seq += '. '
    free_ent = list(set(ent_type_dict)-set(triple_ent))
    if is_add_entType: free_ent = [add_entType(ent, ent_type_dict[ent]) for ent in free_ent]
    if len(free_ent)>0:
        seq += '/n'
        seq += ', '.join(free_ent)
        seq += '.'

    return seq


def getKGInputDF(args, data_split, sections, skip_null):
    # input_dataset = get_KGdata(args, data_split, sections)
    how = 'inner' if skip_null else 'outer'
    for i, sec in enumerate(sections):
        sec_df = pd.DataFrame(get_KGdata(args, data_split, sec))
        sec_df.drop(["ner", "relations"], axis=1, inplace=True)
        sec_df.rename(columns={
            "sentences": f"{sec}_sent",
            "predicted_ner": f"{sec}_ner",
            "predicted_re": f"{sec}_rel"
        }, inplace=True)
        if i==0: input_dataset = sec_df.copy()
        else: input_dataset = pd.merge(input_dataset, sec_df, on="doc_key", how=how)
    input_dataset_list = []
    data_len=args.prototype if isinstance(args.prototype, int) else len(input_dataset)
    # for i, data in enumerate(input_dataset):
    for i, data in input_dataset.iterrows():
        print_progress(i, data_len, desc=f'Processing arXiv KG ({data_split})', bar_size=30)
        ent_type_dict, triple_df = build_graph(data, sections)
        seq = graph2seq(ent_type_dict, triple_df)
        input_dataset_list.append({
            "paper_id": data['doc_key'], 
            "input_seq": seq
        })
        if isinstance(args.prototype, int): 
            if i>=args.prototype-1: break
    return pd.DataFrame(input_dataset_list)


def prepro_KGData(args, data_split, sections, skip_null):
    if args.dataset == 'MuP':
        input_df = getKGInputDF(args, data_split, sections, skip_null)
        target_df = getTargetDF(args, data_split)
        if sections == 'summary':
            merged_df = pd.concat([input_df,target_df['target_seq']], axis=1)
        else:
            merged_df = input_df.merge(target_df, how='inner', on='paper_id')
        return removeIssue(merged_df.reset_index()), ENT_TYPE_TOKENS+REL_TYPE_TOKENS
    elif args.dataset == 'arXiv':
        dataset = get_KGdata(args, data_split, sections)
        prepro_data = []
        data_len=args.prototype if isinstance(args.prototype, int) else len(dataset)
        for i, data in enumerate(dataset):
            print_progress(i, data_len, desc=f'Processing arXiv KG ({data_split})', bar_size=30)
            ent_type_dict, triple_list = build_graph(data)
            seq = graph2seq(ent_type_dict, triple_list)
            flatten_sent = [j for i in data["sentences"] for j in i]
            prepro_data.append({
                'paper_id': data['doc_key'],
                'input_seq': seq,
                'target_seq': " ".join(flatten_sent),
            })
            if isinstance(args.prototype, int): 
                if i>=args.prototype-1: break
            
        return pd.DataFrame(prepro_data), ENT_TYPE_TOKENS+REL_TYPE_TOKENS


# =================================== GET TEXT DATA ===================================
def getText_abstract(args, data_split):
    main_path = str((Path().absolute()).parents[0])
    filepath = f"{main_path}/dataset_MuP/{RAWDATAFILES[data_split]}"
    with open(filepath, 'r') as json_file:
        json_list = list(json_file)
    dataset_list = []
    
    data_len=args.prototype if isinstance(args.prototype, int) else len(json_list)
    for i, json_str in enumerate(json_list):
        data = json.loads(json_str)
        if data_split=='test': summary = "<PAD>" 
        else: summary = data["summary"]
        dataset_list.append({
            "paper_id": data["paper_id"], 
            "input_seq": data["paper"]["abstractText"], 
            "target_seq": summary
        })
        print_progress(i, data_len, desc=f'Loading abstract input ({data_split})')
        if isinstance(args.prototype, int) and i>=args.prototype-1: break
    return pd.DataFrame(dataset_list)

def getText_section(args, data_split, sections, skip_null):
    main_path = str((Path().absolute()).parents[0])
    filepath = f"{main_path}/dataset_MuP/{data_split}_iden.jsonl"
    with open(filepath, 'r') as json_file:
        json_list = list(json_file)
    dataset_list = []
    
    if isinstance(args.prototype, int): data_len = args.prototype
    else: data_len = len(json_list)
    for i, json_str in enumerate(json_list):
        data = json.loads(json_str)
        input_seq = []
        for sec in sections:
            if sec=="introduction-1":
                sec='introduction'
                if isinstance(data[sec], str): 
                    input_seq.append(data[sec].split("\n")[-1])
                elif skip_null:
                    continue
            else:
                if isinstance(data[sec], str): 
                    input_seq.append(data[sec])
                elif skip_null:
                    continue
        input_seq = " [SEP] ".join(input_seq)
        dataset_list.append({
            "paper_id": data["paper_id"], 
            "input_seq": input_seq
        })
        print_progress(i, data_len, desc=f'Loading section input ({data_split})')
        
        
        if isinstance(args.prototype, int) and i>=args.prototype-1: break
    return pd.DataFrame(dataset_list)

def prepro_textData(args, data_split, sections, skip_null):
    if sections==['abstract']:
        dataset_df = getText_abstract(args, data_split)
    else:
        input_df = getText_section(args, data_split, sections, skip_null)
        target_df = getTargetDF(args, data_split)
        dataset_df = input_df.merge(target_df, how='inner', on='paper_id')
    if len(sections)==1: spacial_token = None
    else: spacial_token=['[SEP]']
    return removeIssue(dataset_df), spacial_token