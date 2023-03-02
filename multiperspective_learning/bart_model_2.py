import os
import sys
import json

import pandas as pd


import torch
from torch.utils.data import Dataset
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import AutoTokenizer, AutoModelWithLMHead
# from transformers import AdapterConfig, PrefixTuningConfig, AdapterType
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
# from transformers.adapters import BartAdapterModel
# from transformers.adapters.composition import Stack, Fuse

from datasets import load_dataset, Dataset

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

BATCH_SIZE        = 2
MAX_INPUT_LENGTH  = 512
MAX_TARGET_LENGTH = 512
PROTOTYPE_SIZE    = 16*8 #samples

MODEL_CHECKPOINT = 'facebook/bart-large'

text_column = "text"
summary_column = "summary"
prefix = ""
padding='max_length'

tokenizer = BartTokenizer.from_pretrained(MODEL_CHECKPOINT)

def preprocess_function(examples):
    # remove pairs where at least one record is None

    inputs, targets = [], []
    for i in range(len(examples[text_column])):
        if examples[text_column][i] is not None and examples[summary_column][i] is not None:
            inputs.append(examples[text_column][i])
            targets.append(examples[summary_column][i])
    inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, padding=padding, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=MAX_TARGET_LENGTH, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and True:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def show_samples(dataset, num_samples=1, seed=42):
    sample = dataset.shuffle(seed=seed).select(range(num_samples))
    for example in sample:
        print(f"\n'>> Title: {example['paper_name']}'")
        print(f"'>> Text ({len(example['text'])} characters):  {(example['text'][:200])}'")
        print(f"'>> Summary: {(example['summary'])}'")

        
        
        
        
        
        
def main():
    raw_dataset = load_dataset("allenai/mup")
    train_dataset = Dataset.from_pandas(pd.DataFrame(data=(raw_dataset["train"][:PROTOTYPE_SIZE])))
    
    
    torch.cuda.set_device(0)

    print(f"{bcolors.OKGREEN}DEIVCE: {bcolors.ENDC}{torch.cuda.get_device_name(torch.cuda.current_device())}")
    # for i in range(torch.cuda.device_count()):
    #     print(torch.cuda.device(i))
    #     print(torch.cuda.get_device_name(i))
    
    # return
    # show_samples(train_dataset)

    train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                desc="Running tokenizer on train dataset",
            )
    print(train_dataset)
    
    print(f"{bcolors.OKGREEN}\n===== LOAD MODEL ====={bcolors.ENDC}")
    model = BartForConditionalGeneration.from_pretrained(MODEL_CHECKPOINT)
    
    print(f"{bcolors.OKGREEN}\n===== DATA COLLATOR ====={bcolors.ENDC}")
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        # label_pad_token_id=label_pad_token_id,
        # pad_to_multiple_of=8,
    )
        
    print(f"{bcolors.OKGREEN}\n===== TRAINING ARGS ====={bcolors.ENDC}")
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

    print(f"{bcolors.OKGREEN}\n===== TRAINER ====={bcolors.ENDC}")
    
    train_dataset = train_dataset.remove_columns(
        raw_dataset["train"].column_names
    )
   
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    print(f"{bcolors.OKGREEN}\n===== TRAIN ====={bcolors.ENDC}")
    trainer.train()

    


if __name__ == '__main__':
    main()