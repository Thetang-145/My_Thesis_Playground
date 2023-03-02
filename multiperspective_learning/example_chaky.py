import numpy as np
import pandas as pd
import evaluate

import nltk
from nltk.tokenize import sent_tokenize

import huggingface_hub
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer

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

VERBOSE = False
PADDING = False

huggingface_hub.login("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", True)

spanish_dataset = load_dataset("amazon_reviews_multi", "es")
english_dataset = load_dataset("amazon_reviews_multi", "en")

print(f"{bcolors.OKGREEN}\n===== LOAD DATASETS ====={bcolors.ENDC}")
print(english_dataset)

def show_samples(dataset, num_samples=1, seed=42):
    sample = dataset["train"].shuffle(seed=seed).select(range(num_samples))
    print(f"{bcolors.OKBLUE}\n===== SHOW SAMPLES ====={bcolors.ENDC}")
    for example in sample:
        print(f"'{bcolors.OKBLUE}>> Title:{bcolors.ENDC} {example['review_title']}'")
        print(f"'{bcolors.OKBLUE}>> Review:{bcolors.ENDC} {example['review_body']}'")


show_samples(english_dataset)

english_dataset.set_format("pandas")
english_df = english_dataset["train"][:]
# Show counts for top 20 products

if VERBOSE:
    print(f"{bcolors.OKGREEN}\n===== PRODUCT CATAGORIES ====={bcolors.ENDC}")
    print(english_df["product_category"].value_counts()[:20])

english_dataset.reset_format()

def filter_books(example):
    return (
        example["product_category"] == "book"
        or example["product_category"] == "digital_ebook_purchase"
    )


print(f"{bcolors.OKGREEN}\n===== FILTER DATASETS ====={bcolors.ENDC}")
spanish_books = spanish_dataset.filter(filter_books)
english_books = english_dataset.filter(filter_books)
if VERBOSE:
    show_samples(english_books)

books_dataset = DatasetDict()

print(f"{bcolors.OKGREEN}\n===== SPLITING DATASETS ====={bcolors.ENDC}")
for split in english_books.keys():
    books_dataset[split] = concatenate_datasets(
        [english_books[split], spanish_books[split]]
    )
    books_dataset[split] = books_dataset[split].shuffle(seed=42)

# Peek at a few examples
if VERBOSE:
    show_samples(books_dataset)

books_dataset = books_dataset.filter(lambda x: len(x["review_title"].split()) > 2)


model_checkpoint = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

max_input_length = 512
max_target_length = 30


def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["review_body"],
        max_length=max_input_length,
        truncation=True,
        padding=PADDING,
    )
    labels = tokenizer(
        text_target=examples["review_title"], 
        max_length=max_target_length, 
        truncation=True,
        padding=PADDING,
    )
    model_inputs["labels"] = labels["input_ids"]
    if PADDING:
        model_inputs["labels_mask"] = labels["attention_mask"]
    return model_inputs

if VERBOSE:
    print(books_dataset)
    
print(f"{bcolors.OKGREEN}\n===== PREPROCESSING ====={bcolors.ENDC}")
tokenized_datasets = books_dataset.map(preprocess_function, batched=True)

nltk.download("punkt")

def three_sentence_summary(text):
    return "\n".join(sent_tokenize(text)[:3])


if VERBOSE:
    print(f"{bcolors.OKGREEN}\n===== 3 SENTENCES SUMMARY ====={bcolors.ENDC}")
    print(three_sentence_summary(books_dataset["train"][1]["review_body"]))

def evaluate_baseline(dataset, metric):
    summaries = [three_sentence_summary(text) for text in dataset["review_body"]]
    return metric.compute(predictions=summaries, references=dataset["review_title"])

rouge_score = evaluate.load("rouge")
# score = evaluate_baseline(books_dataset["validation"], rouge_score)
# rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
# rouge_dict = dict((rn, round(score[rn].mid.fmeasure * 100, 2)) for rn in rouge_names)
# print(rouge_dict)


model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

batch_size = 8
num_train_epochs = 8
# Show the training loss with every epoch
logging_steps = len(tokenized_datasets["train"]) // batch_size
model_name = model_checkpoint.split("/")[-1]

args = Seq2SeqTrainingArguments(
    output_dir=f"{model_name}-finetuned-amazon-en-es",
    evaluation_strategy="epoch",
    learning_rate=5.6e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    logging_steps=logging_steps,
    push_to_hub=True,
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Decode generated summaries into text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    # Compute ROUGE scores
    result = rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract the median scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}

data_collator = DataCollatorForSeq2Seq(
    tokenizer, 
    model=model
)


# print(f"{bcolors.OKGREEN}\n===== COLUMN ====={bcolors.ENDC}")

# print(tokenized_datasets["train"].column_names)
tokenized_datasets = tokenized_datasets.remove_columns(
    books_dataset["train"].column_names
)
# print(tokenized_datasets["train"].column_names)


features = [tokenized_datasets["train"][i] for i in range(2)]


print(f"{bcolors.OKGREEN}\n===== FEATURES ====={bcolors.ENDC}")
# print(features)
for k, v in features[0].items():
    print(k, (v))

print(f"{bcolors.OKGREEN}\n===== DATA COLLECTOR ====={bcolors.ENDC}")
print(data_collator(features).keys())

print(f"{bcolors.OKGREEN}\n===== TRAINER ====={bcolors.ENDC}")
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print(f"{bcolors.OKGREEN}\n===== TRAIN ====={bcolors.ENDC}")
trainer.train()
