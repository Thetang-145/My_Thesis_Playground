import os
import json
import argparse
import timeit
from datetime import timedelta

def getRE(dataset, section, filename, gpu):
    command = f"CUDA_VISIBLE_DEVICES={gpu} python3 _getRE.py  --model_type bertsub  \
    --model_name_or_path /pretrained_model/scire-scibert  --do_lower_case  \
    --data_dir _sciner_models/{dataset}/{section}  \
    --learning_rate 2e-5  --num_train_epochs 10  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
    --max_seq_length 256  --max_pair_length 16  --save_steps 2500  \
    --do_eval  --eval_all_checkpoints  --eval_logsoftmax  \
    --fp16   \
    --test_file {filename}_ner.json  \
    --use_ner_results \
    --output_dir _scire_models/{dataset}/{section}  --output_file {filename}_re.json"
    # print(f"Run command: {command}")
    return os.system(command)
        
def getNER(dataset, section, filename, gpu, bs=16):
    command = f"CUDA_VISIBLE_DEVICES={gpu} python3  _getNER.py  --model_type bertspanmarker  \
    --model_name_or_path  pretrained_model/sciner-scibert  --do_lower_case  \
    --data_dir _prepared_data/{dataset}/{section}  \
    --learning_rate 2e-5  --num_train_epochs 50  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size {bs}  --gradient_accumulation_steps 1  \
    --max_seq_length 512  --save_steps 2000  --max_pair_length 256  --max_mention_ori_length 8    \
    --do_eval  --evaluate_during_training   --eval_all_checkpoints  \
    --fp16  --seed 42  --onedropout  --lminit  \
    --test_file {filename}.jsonl  \
    --output_dir _sciner_models/{dataset}/{section}  --output_file {filename}_ner.json \
    --overwrite_output_dir  --output_results"
    # print(f"Run command: {command}")
    return os.system(command)
    
def scan_issue(dataset, section, filename, gpu):
    command = f"CUDA_VISIBLE_DEVICES={gpu} python3  _scan_issue.py  --model_type bertspanmarker  \
    --model_name_or_path  pretrained_model/sciner-scibert  --do_lower_case  \
    --data_dir _prepared_data/{dataset}/{section}  \
    --learning_rate 2e-5  --num_train_epochs 50  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
    --max_seq_length 512  --save_steps 2000  --max_pair_length 256  --max_mention_ori_length 8    \
    --do_eval  --evaluate_during_training   --eval_all_checkpoints  \
    --fp16  --seed 42  --onedropout  --lminit  \
    --test_file {filename}.jsonl   \
    --output_dir {dataset}_{section} \
    --overwrite_output_dir  --output_results"
    # print(f"Run command: {command}")
    return os.system(command)

def get_multifile(prepared_dir, data_split, revised=True):
    files = sorted(os.listdir(prepared_dir))
    run_files = []
    if revised:
        for file in files:
            if file[:14]==f"{data_split}_revised_": 
                run_files.append(file.split('.')[0])
    else:
        for file in files:
            if file[:14]!=f"{data_split}_revised_" and file[:6]==f"{data_split}_": 
                run_files.append(file.split('.')[0])
    return run_files

def remove_revised(dataset, section, data_split, rec_sent=False):
    record_docID_path = f"_issue_files/doc_id/{dataset}_{section}_{data_split}.txt"
    with open(record_docID_path, 'r') as f:
        docId_list = (f.read()).split(",") 
    docId_list = [int(docId) for docId in docId_list]
    
    original_path = f"_prepared_data/{dataset}/{section}/{data_split}.jsonl"
    output_path   = f"_prepared_data/{dataset}/{section}/revised_{data_split}.jsonl"
    issue_sent_path = f"_issue_files/issue_sentence/{dataset}_{section}_{data_split}.txt"
    
    original = open(original_path, 'r')
    output_w = open(output_path, 'w')
    wrote = 0
    for l_idx, line in enumerate(original):
        data = json.loads(line)
        if l_idx not in docId_list:
            output_w.write(json.dumps(data)+'\n')
            wrote += 1
        else:
            if rec_sent:
                with open(issue_sent_path, 'a') as issue_w:
                    issue_w.write("="*50+data['doc_key']+"="*50+"\n")
                    for sent in data['sentences']:
                        issue_w.write(" ".join(sent)+"\n")
    output_w.close()
    original.close()
    total = l_idx+1
    print(f"Finish remove issue samples from {original_path}")
    print(f"\tWrote: {wrote}/{total}")
    print(f"\tRemove: {len(docId_list)}/{total} ({len(docId_list)/total*100:.2}%)")
    print(f"\tCheck: remove+wrote={len(docId_list)+wrote}, total={total}")
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='MuP', type=str,
                        help="which dataset to be operate: MuP, arXiv etc.")
    parser.add_argument("--section", default='abstract', type=str,
                        help="which data to be operate: summary, abstract, section_1, section_introduction, etc.")
    
    parser.add_argument("--nopre", action='store_true', help="prepareMuP")
    parser.add_argument("--noner", action='store_true', help="getNER")
    parser.add_argument("--norel", action='store_true', help="getRE")
    
    parser.add_argument("--train", action='store_true', help="operate on train data")
    parser.add_argument("--val", action='store_true', help="operate on validate data")
    parser.add_argument("--test", action='store_true', help="operate on test data")
    
    parser.add_argument("--scan", action='store_true', help="scan issue samples")
    parser.add_argument("--remove", action='store_true', help="remove issue samples")
    parser.add_argument("--run_revised", action='store_true', help="extract KG from revised samples")
    
    parser.add_argument("--gpu", default=0, type=int, 
                        help="CUDA_VISIBLE_DEVICES")
    args = parser.parse_args()

    
    # Data split handling    
    if not (args.train or args.val or args.test):
        args.train = args.val = args.test = True
    if args.section == 'summary': args.test = False    
    data_split_dict = {
        "train": args.train,
        "val": args.val,
        "test": args.test
    }    
    run_data_split = []
    for k, v in data_split_dict.items():
        if v: run_data_split.append(k)
    
    
    # Run issue samples scaning
    if args.scan:
        for data_split in run_data_split:
            if args.dataset=='arXiv' and data_split=='train':
                prepared_dir = f"_prepared_data/{args.dataset}/{args.section}"
                run_files = get_multifile(prepared_dir, data_split, revised=False)
                for file in run_files:
                    start_time = timeit.default_timer() 
                    record_txt_path = f"_issue_files/batch/{args.dataset}_{args.section}_{file}.txt"
                    if not (os.path.isfile(record_txt_path)):
                        print(f"Creating {record_txt_path} file")
                        with open(record_txt_path, 'w') as f: f.write('')
                    while (open(record_txt_path, 'r').read()).split(", ")[-1] != 'FINISH':
                        scan_issue(args.dataset, args.section, file, args.gpu)
                    scanTime = timeit.default_timer() - start_time
                    scanTime = str(timedelta(seconds=scanTime))[:8]
                    print(f'Finish scanning {args.dataset} dataset: {file} in {scanTime}')
                    # break
            else:
                record_txt_path = f"_issue_files/batch/{args.dataset}_{args.section}_{data_split}.txt"
                if not (os.path.isfile(record_txt_path)):
                    print(f"Creating {record_txt_path} file")
                    with open(record_txt_path, 'w') as f: f.write('')
                while (open(record_txt_path, 'r').read()).split(", ")[-1] != 'FINISH':
                    scan_issue(args.dataset, args.section, data_split, args.gpu)
                print(f'Finish scanning {args.dataset} dataset: {data_split}')
        exit()
        
    if args.remove:
        for data_split in run_data_split:
            if args.dataset=='arXiv' and data_split=='train':
                prepared_dir = f"_prepared_data/{args.dataset}/{args.section}"
                run_files = get_multifile(prepared_dir, data_split, revised=False)
                for file in run_files:
                    remove_revised(args.dataset, args.section, file)
            else:
                remove_revised(args.dataset, args.section, data_split, rec_sent=True)
        exit()
                
                    
        # for k, v in data_split.items():
        #     if v:
        #         for file in files:
        #             if file[:len(k)]==k:
        #                 while (open("issue_arXiv.txt", 'r').read()).split(", ")[-1] != 'FINISH':
        #                     check_inputID(args.dataset, args.section, file[:-6], args.gpu)
        #                 print('finish')
        #                 break
        # return
    

    # Run dataset preparing
    if not args.nopre:
        train = val = test = ""
        if args.train: train = "--train "
        if args.val:   val   = "--val "
        if args.test:  test  = "--test "
        if args.dataset=="MuP":
            command = f"python3 _prepareMuP.py --section {args.section} {train}{val}{test}"
        elif args.dataset=="arXiv":
            command = f"python3 _prepareArXiv.py --section {args.section} {train}{val}{test}"
        print(f"Run command: {command}")
        output = os.system(command)
        
        

    revised_str = "revised_" if args.run_revised else ""
            
    # Run NER extraction
    if not args.noner:
        for data_split in run_data_split:
        # for k, v in data_split.items():
            if args.dataset=='arXiv' and data_split=='train':
                prepared_dir = f"_prepared_data/{args.dataset}/{args.section}"
                run_files = get_multifile(prepared_dir, data_split, revised=args.run_revised)
                for file in run_files:
                    print(f"Extract NER from {file}")
                    getNER(args.dataset, args.section, f"{revised_str}{file}", args.gpu)
            else:
                print(f"Extract NER from {revised_str}{data_split}")
                getNER(args.dataset, args.section, f"{revised_str}{data_split}", args.gpu)


        # files = sorted(os.listdir(prepared_dir))
        # for k, v in data_split.items():
        #     if v:
        #         for file in files:
        #             if args.dataset=='arXiv' and k=='train':
        #                 if file[:14]!="train_revised_" and file[:6]=="train_":
        #                 # if file[:14]=="train_revised_":
        #                     print(f"Get dataset from {file[:-6]}")
        #                     getNER(args.dataset, args.section, file[:-6], args.gpu)
        #                     break
        #             elif file[:len(k)]==k: 
        #                 getNER(args.dataset, args.section, file[:-6], args.gpu)
        #                 break


    # Run Relation extraction
    if not args.norel:
        for data_split in run_data_split:
        # for k, v in data_split.items():
            if args.dataset=='arXiv' and data_split=='train':
                prepared_dir = f"_prepared_data/{args.dataset}/{args.section}"
                run_files = get_multifile(prepared_dir, data_split, revised=args.run_revised)
                for file in run_files:
                    print(f"Extract RE from {file}")
                    getNER(args.dataset, args.section, f"{revised_str}{file}", args.gpu)
            else:
                print(f"Extract RE from {revised_str}{data_split}")
                getRE(args.dataset, args.section, f"{revised_str}{data_split}", args.gpu)

                    
        # prepared_dir = f"_sciner_models/{args.dataset}/{args.section}"
        # files = sorted(os.listdir(prepared_dir))
        # for k, v in data_split.items():
        #     if v:
        #         for file in files:
        #             if file[:len(k)]==k: 
        #                 getRE(args.dataset, args.section, file[:-9], args.gpu)
        #                 break
                        
    exit()
                
if __name__ == "__main__":
    main()