import pandas as pd

import logging
import timeit
from datetime import datetime, timedelta

import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import LEDForConditionalGeneration, LEDTokenizer
from transformers import AutoTokenizer, AutoModel
from transformers import AdamW
from rouge import Rouge
   
MODELS = {
    "bart-large": "facebook/bart-large",
    "bart-large-cnn": "facebook/bart-large-cnn",
    "led-base": "allenai/longformer-base-4096"
}

def print_log(msg):
    print(msg)
    logging.info(msg)    

class MyDataset(Dataset):
    def __init__(self, args, data, tokenizer):
        self.data = data
        self.paper_id= list(data['paper_id'])
        self.input = list(data['input_seq'])
        self.target = list(data['target_seq'])
        self.tokenizer = tokenizer
        self.max_input_length = args.max_input_length
        self.max_output_length = args.max_output_length

    def __getitem__(self, index):
        input_tokens = self.tokenizer.encode(self.input[index], padding='max_length', max_length=self.max_input_length, truncation=True, return_tensors='pt')
        target_tokens = self.tokenizer.encode(self.target[index], padding='max_length', max_length=self.max_output_length, truncation=True, return_tensors='pt')
        
        paper_id = self.paper_id[index]
        input_ids = input_tokens.squeeze()
        input_attention_mask = torch.ones(input_ids.shape)
        target_ids = target_tokens.squeeze()
        target_attention_mask = torch.ones(target_ids.shape)
                
        return paper_id, input_ids, input_attention_mask, target_ids

    def __len__(self):
        return len(self.data)


def validation(model, tokenizer, val_loader):
    model.eval()
    rouge_f = {
        'rouge-1': [],
        'rouge-2': [],
        'rouge-l': [],
    }
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            # Forward pass through model
            paper_id, input_ids, attention_mask, target_ids = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            target_ids = target_ids.to(device)
            generated_ids = model.generate(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                max_length=tokenizer.model_max_length, 
                num_beams=NUM_BEAMS
            )
            # Compute evaluation metrics
            hyps = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]
            refs = [tokenizer.decode(ids, skip_special_tokens=True) for ids in target_ids]
            # print(len(hyps), len(refs))
            # print(len(hyps[0]), hyps[0])
            # print(len(refs[0]), refs[0])
            scores = Rouge().get_scores(hyps, refs)
            for k in rouge_f.keys(): rouge_f[k]+=[score[k]['f'] for score in scores]
            if (idx+1)>=100: break
        for k in rouge_f.keys(): rouge_f[k]=sum(rouge_f[k])/len(rouge_f[k])
    return rouge_f

def contrastive_loss(logits, labels, margin=1.0):
    """
    Contrastive loss function
    logits: output of the model
    labels: target labels for the input
    margin: margin parameter for the contrastive loss
    """
    # split the logits into anchor and positive pairs
    anchor_logits = logits[::2]
    pos_logits = logits[1::2]

    # compute the distance between the anchor and positive pairs
    distance = F.pairwise_distance(anchor_logits, pos_logits)

    # compute the contrastive loss
    loss_contrastive = torch.mean((1 - labels) * torch.pow(distance, 2) +
                                  labels * torch.pow(torch.clamp(margin - distance, min=0.0), 2))

    return loss_contrastive


def train(args, train_data, val_data, model_filename):
    
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(args.cuda) if torch.cuda.is_available() else 'CPU'
    
    model_path = MODELS[args.model]
    model_cat  = args.model.split("-")[0]
    
    print_log(f"Loading {args.model} from {model_path}")
    if model_cat=='bart':
        tokenizer = BartTokenizer.from_pretrained(model_path)
        model = BartForConditionalGeneration.from_pretrained(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    # model.config.loss_fct = contrastive_loss    
    
    train_dataset = MyDataset(args, train_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size)
    
    val_dataset = MyDataset(args, val_data, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size)

    print(f"Training data on {device} ({device_name})")
    model.to(device=device)    

    eval_record = []
    max_rouge1 = 0
    # max_rougeAvg = 0
    iter = 0
    for epoch in range(args.num_epoch):
        logging.info(f"Training Epoch:{epoch+1}")
        start_time_ep = timeit.default_timer() 
        for batch in tqdm(train_loader, desc=f"Fine-tuning epoch:{epoch+1}"):
            model.train()
            iter += 1
            paper_id, input_ids, attention_mask, target_ids = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            target_ids = target_ids.to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=target_ids)
            loss = outputs.loss

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Validation
            if iter%args.eval_frequency==0:
                start_time_val = timeit.default_timer() 
                rouge_f = validation(model, tokenizer, val_loader)
                valTime = timeit.default_timer() - start_time_val
                valTime = str(timedelta(seconds=valTime))[:8]
                logging.info(f"""Validating 
                Epoch: {epoch+1} Iteration: {iter+1} 
                TimeUsed: {valTime}""")
                rouge_list = [rouge_f[k] for k in rouge_f.keys()]
                rouge_avg = sum(rouge_list)/len(rouge_list)
                eval_record.append({
                    'epoach': epoch,
                    'iteration': iter,
                    'loss': loss.item(),
                    'rouge-1': rouge_f['rouge-1'],
                    'rouge-2': rouge_f['rouge-2'],
                    'rouge-l': rouge_f['rouge-l'],
                    'rouge-avg': rouge_avg,
                })
                # Save model
                if rouge_f['rouge-1']>max_rouge1:
                    torch.save(model.state_dict(), f'model/{model_filename}_best.pt')
                    logging.info(f"Save best rouge1 model")
                    max_rouge1 = rouge_f['rouge-1']
                # if rouge_avg>max_rougeAvg:
                #     torch.save(model.state_dict(), f'model/{model_filename}_bestRougeAvg.pt')
                #     logging.info(f"Save best rouge-avg model")
                #     max_rougeAvg = rouge_avg
            else:
                eval_record.append({
                    'epoach': epoch,
                    'iteration': iter,
                    'loss': loss.item(),
                })
        epTime = timeit.default_timer() - start_time_ep
        epTime = str(timedelta(seconds=epTime))[:8]
        logging.info(f"Finish Epoch: {epoch+1} in {epTime}")

    return pd.DataFrame(eval_record)

def generateSum(model, tokenizer, test_data, model_filename):
    
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(args.cuda) if torch.cuda.is_available() else 'CPU'
    
    model_path = MODELS[args.model]
    model_cat  = args.model.split("-")[0]
    
    print_log(f"Loading {args.model} from {model_path}")
    if model_cat=='bart':
        tokenizer = BartTokenizer.from_pretrained(model_path)
        model = BartForConditionalGeneration.from_pretrained(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        
    test_dataset = MyDataset(args, test_data, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size)
    
      
    # bestScores = ['bestRouge1', 'bestRougeAvg']
    # for i, best in enumerate(bestScores):
    chechpoint = f'model/{model_filename}_{best}.pt'
    print_log(f"Start generating summary using chechpoint: {chechpoint}")
    model.load_state_dict(torch.load(chechpoint))
    model.to(device=device)
    model.eval()
    total_loss = 0.0

    start_time = timeit.default_timer() 
    with torch.no_grad():
        result = []
        for batch in tqdm(test_loader, desc=f"Summarizing"):
            paper_id_list, input_ids, attention_mask, target_ids = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            target_ids = target_ids.to(device)

            generated_ids = model.generate(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                max_length=tokenizer.model_max_length, 
                num_beams=args.num_beams
            )
            for idx, paper_id in enumerate(paper_id_list):
                result.append({
                    'paper_id': paper_id,
                    'input': tokenizer.decode(input_ids[idx], skip_special_tokens=True),
                    f'output_{best}': tokenizer.decode(generated_ids[idx], skip_special_tokens=True)
                })
                
    result_df = pd.DataFrame(result)
        # if i==0:
        #     result_df = pd.DataFrame(result)
        # else:
        #     result_df_ = pd.DataFrame(result)
        #     result_df = pd.merge(result_df, result_df_, on=['paper_id', 'input'], how='outer')
    print_log(f"Finish generating {len(result)} summaries using {chechpoint}")        
    return result_df  
    
