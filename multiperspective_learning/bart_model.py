import os
import sys
import json

import pandas as pd


import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import AutoTokenizer, AutoModelWithLMHead
from transformers import AdapterConfig, PrefixTuningConfig, AdapterType
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from transformers.adapters import BartAdapterModel
from transformers.adapters.composition import Stack, Fuse

BATCH_SIZE        = 16
MAX_INPUT_LENGTH  = 1024
MAX_TARGET_LENGTH = 2048
PROTOTYPE_SIZE    = 16*8 #samples

MODEL_CHECKPOINT = 'facebook/bart-large'

DATASET_PATHS = {
    "train": "MuP_sum/dataset/training_complete.jsonl",
    "val": "MuP_sum/dataset/validation_complete.jsonl",
    "test": "MuP_sum/dataset/testing_with_paper_release.jsonl"
}



def load_dataset(dataset_type="train"):
    cur_path = os.path.join(os.getcwd(), os.pardir)
    path = os.path.join(os.path.abspath(cur_path), DATASET_PATHS[dataset_type])
    
    with open(path, 'r') as json_file:
        json_list = list(json_file)
        
    input_data = []
    output_data = []
    for i, json_str in enumerate(json_list):
        result = json.loads(json_str)
        
        abstract    = result['paper']['abstractText']
        try: 
            heading_1st = result['paper']['sections'][0]['heading']
        except:
            heading_1st = ''

        text_1st    = result['paper']['sections'][0]['text']
        summary     = result['summary']
        
            
        input_data.append(f'{abstract} {heading_1st} {text_1st}')
        output_data.append(summary)
        
        if i+1 >= PROTOTYPE_SIZE:
            break
            
    # tokenizer    = BartTokenizer.from_pretrained(MODEL_CHECKPOINT)
    # model_inputs = tokenizer(
    #     input_data, 
    #     max_length=MAX_INPUT_LENGTH, 
    #     truncation=True, 
    #     # return_tensors="pt"
    # )
    # labels = tokenizer(
    #     output_data, 
    #     max_length=MAX_TARGET_LENGTH, 
    #     truncation=True, 
    #     # return_tensors="pt"
    # )
    # model_inputs["labels"] = labels["input_ids"]
    # model_inputs["labels_mask"] = labels["attention_mask"]
    # model_inputs_tensor = {}
    # for k,v in model_inputs.items():
    #     print(k, len(v), sep="\t")
    #     for i, sub_v in enumerate(v):
    #         print("", len(sub_v), sub_v[:10], sep="\t")
    #         if i>=5:
    #             break
    #     model_inputs_tensor[k] = [torch.tensor(sub_v) for sub_v in v]
    # print("="*100)
    # for k,v in model_inputs_tensor.items():
    #     print(k, len(v), sep="\t")
    #     for i, sub_v in enumerate(v):
    #         print("", len(sub_v), sub_v[:10], sep="\t")
    #         if i>=5:
                # break
    return input_data, output_data

class SummarizationDataset(Dataset):
    def __init__(self, source_texts, target_texts, tokenizer):
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.source_texts)
    
    def __getitem__(self, index):
        source_text = self.source_texts[index]
        target_text = self.target_texts[index]
        
        # Encode the source and target texts using the tokenizer
        source_inputs = self.tokenizer(
            source_text, padding='max_length', 
            truncation=True, max_length=MAX_INPUT_LENGTH, 
            return_tensors='pt')
        target_inputs = self.tokenizer(
            target_text, padding='max_length', 
            truncation=True, max_length=MAX_TARGET_LENGTH, 
            return_tensors='pt')
        
        # Create the input and target dictionaries
        input_model = {
            'input_ids': source_inputs['input_ids'][0],
            'attention_mask': source_inputs['attention_mask'][0],
            'labels': target_inputs['input_ids'][0][1:],
            'labels_mask': target_inputs['attention_mask'][0][1:]
        }
        
        return input_model


# class Dataset(torch.utils.data.Dataset):
#     def __init__(self, inputs, targets):
#         self.inputs = inputs
#         self.targets = targets
    
#     def __len__(self):
#         return len(self.targets)
    
#     def __getitem__(self, index):
#         input_ids = torch.tensor(self.inputs[index]).squeeze()
#         target_ids = torch.tensor(self.targets[index]).squeeze()
        
#         return {"input_ids": input_ids, "labels": target_ids}

def finetuning(dataset):
    model = BartForConditionalGeneration.from_pretrained(MODEL_CHECKPOINT)
    tokenizer = BartTokenizer.from_pretrained(MODEL_CHECKPOINT)
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        weight_decay=0.01,
        save_total_limit=3,
        predict_with_generate=True,
        fp16=True,
        # push_to_hub=True,
    )
    # dataset = ForTrainDataset(dataset['input_ids'], dataset['labels'])
    # dataset = dataset[0]
    trainer = Seq2SeqTrainer(
        model=model, 
        args=training_args, 
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # trainer = Seq2SeqTrainer(
    #     model,
    #     training_args,
    #     train_dataset=dataset,
    #     eval_dataset=dataset,
    #     data_collator=data_collator,
    #     tokenizer=tokenizer,
    #     compute_metrics=compute_metrics
    # )
    
    for batch in dataset:
        print("training")
        trainer.train(batch)




def main():
    sources, targets = load_dataset()
    print("Load data")
    
    tokenizer = BartTokenizer.from_pretrained(MODEL_CHECKPOINT)
    dataset_class = SummarizationDataset(sources, targets, tokenizer)
    print("Create a data class")

    dataloader = DataLoader(dataset_class, batch_size=BATCH_SIZE, shuffle=True)
    print(len(dataloader))
    for batch in dataloader:
        print(len(batch))
        for k, v in batch.items():
            print(k, type(v), v.shape, sep='\t')
        break
    print("DataLoader")
    finetuning(dataloader)
    
    # dataset_loader = DataLoader(load_dataset(), batch_size=BATCH_SIZE, shuffle=True)
    # trained_model = training(dataset_loader)
    
#     input_text = 'The quick brown fox jumps over the lazy dog.'
#     input_ids = tokenizer.encode(input_text, return_tensors='pt')
#     generated_summary_ids = trained_model.generate(input_ids)
#     generated_summary = tokenizer.decode(generated_summary_ids[0], skip_special_tokens=True)

#     print(generated_summary)
    


if __name__ == '__main__':
    main()