from __future__ import absolute_import, division, print_function

import logging
import os
import sys
from pathlib import Path
from collections import defaultdict


import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, Dataset
import json
import pickle
import numpy as np
import unicodedata
import itertools
import math
from tqdm import tqdm
import re
import timeit
import logging
logger = logging.getLogger(__name__)

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertTokenizer,
                                  RobertaConfig,
                                  RobertaTokenizer,
                                  get_linear_schedule_with_warmup,
                                  AdamW,
                                  BertForNER,
                                  BertForSpanNER,
                                  BertForSpanMarkerNER,
                                  BertForSpanMarkerBiNER,
                                  AlbertForNER,
                                  AlbertConfig,
                                  AlbertTokenizer,
                                  BertForLeftLMNER,
                                  RobertaForNER,
                                  RobertaForSpanNER,
                                  RobertaForSpanMarkerNER,
                                  AlbertForSpanNER,
                                  AlbertForSpanMarkerNER,
                                  )

LEARNING_RATE               = 2e-5
NUM_TRAIN_EPOCHS            = 50
PER_GPU_TRAIN_BATCH_SIZE    = 8
PER_GPU_EVAL_BATCH_SIZE     = 16
GRADIENT_ACCUMULATION_STEPS = 1
MAX_SEQ_LENGTH              = 512
SAVE_STEPS                  = 2000
MAX_PAIR_LENGTH             = 256
MAX_MENTION_ORI_LENGTH      = 8
NUM_LABELS                  = 7

MODEL_TYPE  = "bertspanmarker"
OUTPUT_DIR  = "sciner_models/sciner-scibert"

SHUFFLE     = False
GROUP_SORT  = False
GROUP_EDGE  = False
GROUP_AXIS  = -1
ALPHA       = 1
ONEDROPOUT  = True
USE_FULL_LAYER = -1
LOCAL_RANK  = -1
NO_CUDA     = False
OUTPUT_RESULTS = True

MODEL_CLASSES = {
    'bert': (BertConfig, BertForNER, BertTokenizer),
    'bertspan': (BertConfig, BertForSpanNER, BertTokenizer),
    'bertspanmarker': (BertConfig, BertForSpanMarkerNER, BertTokenizer),
    'bertspanmarkerbi': (BertConfig, BertForSpanMarkerBiNER, BertTokenizer),
    'bertleftlm': (BertConfig, BertForLeftLMNER, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForNER, RobertaTokenizer),
    'robertaspan': (RobertaConfig, RobertaForSpanNER, RobertaTokenizer),
    'robertaspanmarker': (RobertaConfig, RobertaForSpanMarkerNER, RobertaTokenizer),
    'albert': (AlbertConfig, AlbertForNER, AlbertTokenizer),
    'albertspan': (AlbertConfig, AlbertForSpanNER, AlbertTokenizer),
    'albertspanmarker': (AlbertConfig, AlbertForSpanMarkerNER, AlbertTokenizer),
}

def print_progress(curr, full, prefix="", bar_size=40):    
    bar = int((curr+1)/full*bar_size)
    sys.stdout.write(f"\r{prefix}[{'='*bar}{' '*(bar_size-bar)}] {curr+1}/{full}")
    sys.stdout.flush()
    
class ACEDatasetNER(Dataset):
    def __init__(self, tokenizer, file_path, evaluate=False, do_test=False):

        assert os.path.isfile(file_path)

        self.file_path = file_path
                
        self.tokenizer = tokenizer
        self.max_seq_length = MAX_SEQ_LENGTH

        self.evaluate = evaluate
        # self.local_rank = local_rank
        self.model_type = MODEL_TYPE
        self.output_dir = OUTPUT_DIR

        self.ner_label_list = ['NIL', 'Method', 'OtherScientificTerm', 'Task', 'Generic', 'Material', 'Metric']

        self.max_pair_length = MAX_PAIR_LENGTH

        self.shuffle = SHUFFLE
        self.group_sort = GROUP_SORT
        self.group_edge = GROUP_EDGE
        self.group_axis = GROUP_AXIS

        self.max_entity_length = MAX_PAIR_LENGTH * 2
        self.initialize()

    def is_punctuation(self, char):
        # obtained from:
        # https://github.com/huggingface/transformers/blob/5f25a5f367497278bf19c9994569db43f96d5278/transformers/tokenization_bert.py#L489
        cp = ord(char)
        if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False 

    def get_original_token(self, token):
        escape_to_original = {
            "-LRB-": "(",
            "-RRB-": ")",
            "-LSB-": "[",
            "-RSB-": "]",
            "-LCB-": "{",
            "-RCB-": "}",
        }
        if token in escape_to_original:
            token = escape_to_original[token]
        return token
        
    
    def initialize(self):
        print("\tACEDatasetNER-Initialize")
        tokenizer = self.tokenizer
        max_num_subwords = self.max_seq_length - 2

        ner_label_map = {label: i for i, label in enumerate(self.ner_label_list)}

        def tokenize_word(text):
            if (
                isinstance(tokenizer, RobertaTokenizer)
                and (text[0] != "'")
                and (len(text) != 1 or not self.is_punctuation(text))
            ):
                return tokenizer.tokenize(text, add_prefix_space=True)
            return tokenizer.tokenize(text)

        len_f = sum(1 for line in open(self.file_path))
        f = open(self.file_path, "r", encoding='utf-8')
        self.data = []
        self.tot_recall = 0
        self.ner_golden_labels = set([])
        maxL = 0
        maxR = 0

        for l_idx, line in enumerate(f):
            print_progress(l_idx, len_f, prefix="\t")
            data = json.loads(line)
            # if len(self.data) > 5:
            #     break

            if self.output_dir.find('test')!=-1:
                if len(self.data) > 5:
                    break

            sentences = data['sentences']
            for i in range(len(sentences)):
                for j in range(len(sentences[i])):
                    sentences[i][j] = self.get_original_token(sentences[i][j])
            
            ners = data['ner']

            sentence_boundaries = [0]
            words = []
            L = 0
            for i in range(len(sentences)):
                L += len(sentences[i])
                sentence_boundaries.append(L)
                words += sentences[i]

            tokens = [tokenize_word(w) for w in words]
            subwords = [w for li in tokens for w in li]
            maxL = max(len(tokens), maxL)
            subword2token = list(itertools.chain(*[[i] * len(li) for i, li in enumerate(tokens)]))
            token2subword = [0] + list(itertools.accumulate(len(li) for li in tokens))
            subword_start_positions = frozenset(token2subword)
            subword_sentence_boundaries = [sum(len(li) for li in tokens[:p]) for p in sentence_boundaries]

            for n in range(len(subword_sentence_boundaries) - 1):
                sentence_ners = ners[n]

                self.tot_recall += len(sentence_ners)
                entity_labels = {}
                for start, end, label in sentence_ners:
                    entity_labels[(token2subword[start], token2subword[end+1])] = ner_label_map[label]    
                    self.ner_golden_labels.add( ((l_idx, n), (start, end), label) )

                doc_sent_start, doc_sent_end = subword_sentence_boundaries[n : n + 2]

                left_length = doc_sent_start
                right_length = len(subwords) - doc_sent_end
                sentence_length = doc_sent_end - doc_sent_start
                half_context_length = int((max_num_subwords - sentence_length) / 2)

                if left_length < right_length:
                    left_context_length = min(left_length, half_context_length)
                    right_context_length = min(right_length, max_num_subwords - left_context_length - sentence_length)
                else:
                    right_context_length = min(right_length, half_context_length)
                    left_context_length = min(left_length, max_num_subwords - right_context_length - sentence_length)
                if self.output_dir.find('ctx0')!=-1:
                    left_context_length = right_context_length = 0 # for debug

                doc_offset = doc_sent_start - left_context_length
                target_tokens = subwords[doc_offset : doc_sent_end + right_context_length]
                assert(len(target_tokens)<=max_num_subwords)
                target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
                
                entity_infos = []

                for entity_start in range(left_context_length, left_context_length + sentence_length):
                    doc_entity_start = entity_start + doc_offset
                    if doc_entity_start not in subword_start_positions:
                        continue
                    for entity_end in range(entity_start + 1, left_context_length + sentence_length + 1):
                        doc_entity_end = entity_end + doc_offset
                        if doc_entity_end not in subword_start_positions:
                            continue

                        if subword2token[doc_entity_end - 1] - subword2token[doc_entity_start] + 1 > MAX_MENTION_ORI_LENGTH:
                            continue

                        label = entity_labels.get((doc_entity_start, doc_entity_end), 0)
                        entity_labels.pop((doc_entity_start, doc_entity_end), None)
                        entity_infos.append(((entity_start+1, entity_end), label, (subword2token[doc_entity_start], subword2token[doc_entity_end - 1] )))
                # if len(entity_labels):
                #     print ((entity_labels))
                # assert(len(entity_labels)==0)
                    
                # dL = self.max_pair_length 
                # maxR = max(maxR, len(entity_infos))
                # for i in range(0, len(entity_infos), dL):
                #     examples = entity_infos[i : i + dL]
                #     item = {
                #         'sentence': target_tokens,
                #         'examples': examples,
                #         'example_index': (l_idx, n),
                #         'example_L': len(entity_infos)
                #     }                
                    
                #     self.data.append(item)                    
                maxR = max(maxR, len(entity_infos))
                dL = self.max_pair_length 
                if self.shuffle:
                    random.shuffle(entity_infos)
                if self.group_sort:
                    group_axis = np.random.randint(2)
                    sort_dir = bool(np.random.randint(2))
                    entity_infos.sort(key=lambda x: (x[0][group_axis], x[0][1-group_axis]), reverse=sort_dir)

                if not self.group_edge:
                    for i in range(0, len(entity_infos), dL):

                        examples = entity_infos[i : i + dL]
                        item = {
                            'sentence': target_tokens,
                            'examples': examples,
                            'example_index': (l_idx, n),
                            'example_L': len(entity_infos)
                        }        
                        self.data.append(item)
                else:
                    if self.group_axis==-1:
                        group_axis = np.random.randint(2)
                    else:
                        group_axis = self.group_axis
                    sort_dir = bool(np.random.randint(2))
                    entity_infos.sort(key=lambda x: (x[0][group_axis], x[0][1-group_axis]), reverse=sort_dir)
                    _start = 0 
                    while _start < len(entity_infos):
                        _end = _start+dL
                        if _end >= len(entity_infos):
                            _end = len(entity_infos)
                        else:
                            while  entity_infos[_end-1][0][group_axis]==entity_infos[_end][0][group_axis] and _end > _start:
                                _end -= 1
                            if _start == _end:
                                _end = _start+dL

                        examples = entity_infos[_start: _end]

                        item = {
                            'sentence': target_tokens,
                            'examples': examples,
                            'example_index': (l_idx, n),
                            'example_L': len(entity_infos)
                        }  
                                       

                        self.data.append(item)   
                        _start = _end                 



        logger.info('maxL: %d', maxL) # 334
        logger.info('maxR: %d', maxR) 

        # exit() 
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]

        input_ids = self.tokenizer.convert_tokens_to_ids(entry['sentence'])
        L = len(input_ids)

        input_ids += [0] * (self.max_seq_length - len(input_ids))
        position_plus_pad = int(self.model_type.find('roberta')!=-1) * 2

        if self.model_type not in ['bertspan', 'robertaspan', 'albertspan']:

            if self.model_type.startswith('albert'):
                input_ids = input_ids + [30000] * (len(entry['examples'])) + [0] * (self.max_pair_length - len(entry['examples']))   
                input_ids = input_ids + [30001] * (len(entry['examples'])) + [0] * (self.max_pair_length - len(entry['examples']))
            elif self.model_type.startswith('roberta'):
                input_ids = input_ids + [50261] * (len(entry['examples'])) + [0] * (self.max_pair_length - len(entry['examples']))   
                input_ids = input_ids + [50262] * (len(entry['examples'])) + [0] * (self.max_pair_length - len(entry['examples']))
            else:
                input_ids = input_ids + [1] * (len(entry['examples'])) + [0] * (self.max_pair_length - len(entry['examples']))   
                input_ids = input_ids + [2] * (len(entry['examples'])) + [0] * (self.max_pair_length - len(entry['examples']))

            attention_mask = torch.zeros((self.max_entity_length + self.max_seq_length, self.max_entity_length + self.max_seq_length), dtype=torch.int64)
            attention_mask[:L, :L] = 1
            position_ids = list(range(position_plus_pad, position_plus_pad+self.max_seq_length)) + [0] * self.max_entity_length 

        else:
            attention_mask = [1] * L + [0] * (self.max_seq_length - L)
            attention_mask = torch.tensor(attention_mask, dtype=torch.int64)
            position_ids = list(range(position_plus_pad, position_plus_pad+self.max_seq_length)) + [0] * self.max_entity_length 



        labels = []
        mentions = []
        mention_pos = []
        num_pair = self.max_pair_length

        full_attention_mask = [1] * L + [0] * (self.max_seq_length - L) + [0] * (self.max_pair_length)*2

        for x_idx, x in enumerate(entry['examples']):
            m1 = x[0]
            label = x[1]
            mentions.append(x[2])
            mention_pos.append((m1[0], m1[1]))
            labels.append(label)

            if self.model_type in ['bertspan', 'robertaspan', 'albertspan']:
                continue

            w1 = x_idx  
            w2 = w1 + num_pair

            w1 += self.max_seq_length
            w2 += self.max_seq_length
            position_ids[w1] = m1[0]
            position_ids[w2] = m1[1]

            for xx in [w1, w2]:
                full_attention_mask[xx] = 1
                for yy in [w1, w2]:
                    attention_mask[xx, yy] = 1
                attention_mask[xx, :L] = 1

        labels += [-1] * (num_pair - len(labels))
        mention_pos += [(0, 0)] * (num_pair - len(mention_pos))



        item = [torch.tensor(input_ids),
                attention_mask,
                torch.tensor(position_ids),
                torch.tensor(labels, dtype=torch.int64),
                torch.tensor(mention_pos),
                torch.tensor(full_attention_mask)
        ]       

        if self.evaluate:
            item.append(entry['example_index'])
            item.append(mentions)


        return item

    @staticmethod
    def collate_fn(batch):
        fields = [x for x in zip(*batch)]

        num_metadata_fields = 2
        stacked_fields = [torch.stack(field) for field in fields[:-num_metadata_fields]]  # don't stack metadata fields
        stacked_fields.extend(fields[-num_metadata_fields:])  # add them as lists not torch tensors

        return stacked_fields


def evaluate(model, tokenizer, file_path, prefix="", do_test=False):

    eval_output_dir = OUTPUT_DIR

    results = {}
    
    print("Before ACEDatasetNER")
    eval_dataset = ACEDatasetNER(tokenizer=tokenizer, file_path=file_path, evaluate=True, do_test=do_test)    
    print("After ACEDatasetNER")

    ner_golden_labels = set(eval_dataset.ner_golden_labels)
    ner_tot_recall = eval_dataset.tot_recall

    if not os.path.exists(eval_output_dir) and LOCAL_RANK in [-1, 0]:
        os.makedirs(eval_output_dir)

    EVAL_BATCH_SIZE = PER_GPU_EVAL_BATCH_SIZE * max(1, N_GPU)

    eval_sampler = SequentialSampler(eval_dataset) 

    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=EVAL_BATCH_SIZE,  collate_fn=ACEDatasetNER.collate_fn, num_workers=4*int(OUTPUT_DIR.find('test')==-1))

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", EVAL_BATCH_SIZE)

    scores = defaultdict(dict)
    predict_ners = defaultdict(list)

    model.eval()

    start_time = timeit.default_timer() 

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        
        indexs = batch[-2]
        batch_m2s = batch[-1]

        batch = tuple(t.to(DEVICE) for t in batch[:-2])

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'position_ids':   batch[2],
                    #   'labels':         batch[3]
                      }

            if MODEL_TYPE.find('span')!=-1:
                inputs['mention_pos'] = batch[4]
            if USE_FULL_LAYER!=-1:
                inputs['full_attention_mask']= batch[5]

            outputs = model(**inputs)

            ner_logits = outputs[0]
            ner_logits = torch.nn.functional.softmax(ner_logits, dim=-1)
            ner_values, ner_preds = torch.max(ner_logits, dim=-1)
            
            for i in range(len(indexs)):
                index = indexs[i]
                m2s = batch_m2s[i]
                for j in range(len(m2s)):
                    obj = m2s[j]
                    ner_label = eval_dataset.ner_label_list[ner_preds[i,j]]
                    if ner_label!='NIL':
                        scores[(index[0], index[1])][(obj[0], obj[1])] = (float(ner_values[i,j]), ner_label)
            # break
    cor = 0 
    tot_pred = 0
    cor_tot = 0
    tot_pred_tot = 0

    
    print("EXAMPLE")
    print(len(scores))
    for example_index, pair_dict in scores.items():


        sentence_results = []
        for k1, (v2_score, v2_ner_label) in pair_dict.items():
            if v2_ner_label!='NIL':
                sentence_results.append((v2_score, k1, v2_ner_label))

        sentence_results.sort(key=lambda x: -x[0])
        # print(sentence_results)
        no_overlap = []
        def is_overlap(m1, m2):
            if m2[0]<=m1[0] and m1[0]<=m2[1]:
                return True
            if m1[0]<=m2[0] and m2[0]<=m1[1]:
                return True
            return False

        for item in sentence_results:
            m2 = item[1]
            overlap = False
            for x in no_overlap:
                _m2 = x[1]
                if (is_overlap(m2, _m2)):
                    if item[2]==x[2]:
                        overlap = True
                        break

            if not overlap:
                no_overlap.append(item)

            pred_ner_label = item[2]
            tot_pred_tot += 1
            if (example_index, m2, pred_ner_label) in ner_golden_labels:
                cor_tot += 1

        for item in no_overlap:
            m2 = item[1]
            pred_ner_label = item[2]
            tot_pred += 1
            if OUTPUT_RESULTS:
                predict_ners[example_index].append( (m2[0], m2[1], pred_ner_label) )
            if (example_index, m2, pred_ner_label) in ner_golden_labels:
                cor += 1      
    # print(predict_ners)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f example per second)", evalTime,  len(eval_dataset) / evalTime)


#     precision_score = p = cor / tot_pred if tot_pred > 0 else 0 
#     recall_score = r = cor / ner_tot_recall 
#     f1 = 2 * (p * r) / (p + r) if cor > 0 else 0.0

#     p = cor_tot / tot_pred_tot if tot_pred_tot > 0 else 0 
#     r = cor_tot / ner_tot_recall 
#     f1_tot = 2 * (p * r) / (p + r) if cor > 0 else 0.0

#     results = {'f1':  f1, 'f1_overlap': f1_tot, 'precision': precision_score, 'recall': recall_score}



#     logger.info("Result: %s", json.dumps(results))

    if OUTPUT_RESULTS:
        f = open(eval_dataset.file_path)
        if do_test:
            output_w = open(os.path.join(OUTPUT_DIR, 'ent_pred_test.json'), 'w')  
        else:
            output_w = open(os.path.join(OUTPUT_DIR, 'ent_pred_dev.json'), 'w')  
        for l_idx, line in enumerate(f):
            data = json.loads(line)
            num_sents = len(data['sentences'])
            predicted_ner = []
            for n in range(num_sents):
                item = predict_ners.get((l_idx, n), [])
                item.sort()
                predicted_ner.append( item )

            data['predicted_ner'] = predicted_ner
            output_w.write(json.dumps(data)+'\n')

    return results

        
        
def main():
    
    global DEVICE
    global N_GPU
    if LOCAL_RANK == -1 or NO_CUDA: 
        DEVICE = torch.device("cuda" if torch.cuda.is_available() and not NO_CUDA else "cpu")
        N_GPU = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(LOCAL_RANK)
        DEVICE = torch.device("cuda", LOCAL_RANK)
        torch.distributed.init_process_group(backend='nccl')
        N_GPU = 1
        
    DEVICE = torch.device("cuda:0")
    # DEVICE = torch.device("cpu")
    N_GPU  = 1
    print(f"{'='*10} Running on {DEVICE} with {N_GPU} GPU {'='*10}\n")

    print("Import success")
    
    config_class, model_class, tokenizer_class = MODEL_CLASSES[MODEL_TYPE]
    
    main_path = str((Path().absolute()).parents[0])
    
    model_name_or_path = main_path+"/PL-Marker/pretrained_model/sciner-scibert"
    
    config = config_class.from_pretrained(model_name_or_path, num_labels=NUM_LABELS)

    tokenizer = tokenizer_class.from_pretrained(model_name_or_path,  do_lower_case=True)
    print("import tokenizer: DONE")
    
    config.max_seq_length = MAX_SEQ_LENGTH
    config.alpha = ALPHA
    config.onedropout = ONEDROPOUT
    config.use_full_layer= USE_FULL_LAYER
    
    model = model_class.from_pretrained(model_name_or_path, from_tf=bool('.ckpt' in model_name_or_path), config=config)

  
    model.to(DEVICE)
    print("import model: DONE")
    
    file_path = f"{main_path}/PL-Marker/prepared_data/train.jsonl"
    evaluate(model, tokenizer, file_path)


if __name__ == "__main__":
    main()
    
