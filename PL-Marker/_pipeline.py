import os
import argparse


def getRE(section, data_split, gpu):
    os.system(f"CUDA_VISIBLE_DEVICES={gpu} python3 _getRE.py  --model_type bertsub  \
    --model_name_or_path /pretrained_model/scire-scibert  --do_lower_case  \
    --data_dir _sciner_models/{section}  \
    --learning_rate 2e-5  --num_train_epochs 10  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
    --max_seq_length 256  --max_pair_length 16  --save_steps 2500  \
    --do_eval  --eval_all_checkpoints  --eval_logsoftmax  \
    --fp16   \
    --test_file {data_split}_ner.json  \
    --use_ner_results \
    --output_dir _scire_models/{section}  --output_file {data_split}_re.json")
        
def getNER(section, data_split, gpu):
    os.system(f"CUDA_VISIBLE_DEVICES={gpu} python3  _getNER.py  --model_type bertspanmarker  \
    --model_name_or_path  pretrained_model/sciner-scibert  --do_lower_case  \
    --data_dir _prepared_data/{section}  \
    --learning_rate 2e-5  --num_train_epochs 50  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
    --max_seq_length 512  --save_steps 2000  --max_pair_length 256  --max_mention_ori_length 8    \
    --do_eval  --evaluate_during_training   --eval_all_checkpoints  \
    --fp16  --seed 42  --onedropout  --lminit  \
    --test_file {data_split}.jsonl  \
    --output_dir _sciner_models/{section}  --output_file {data_split}_ner.json \
    --overwrite_output_dir  --output_results")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", default='summary', type=str,
                        help="which data to be operate: summary, abstract, section_1, section_introduction, etc.")
    
    parser.add_argument("--nopre", action='store_true', help="prepareMuP")
    parser.add_argument("--noner", action='store_true', help="getNER")
    parser.add_argument("--norel", action='store_true', help="getRE")
    
    parser.add_argument("--train", action='store_true', help="operate on train data")
    parser.add_argument("--val", action='store_true', help="operate on validate data")
    parser.add_argument("--test", action='store_true', help="operate on test data")
    
    parser.add_argument("--gpu", default=0, type=int, 
                        help="CUDA_VISIBLE_DEVICES")
    args = parser.parse_args()
    
    if not (args.train or args.val or args.test):
        args.train = args.val = args.test = True
    if args.section == 'summary': args.test = False
    data_split = {
        "train": args.train,
        "val": args.val,
        "test": args.test
    }
    train = val = test = ""
    if args.train: train = "--train "
    if args.val:   val   = "--val "
    if args.test:  test  = "--test "
    
    if not args.nopre:
        command = f"python3 _prepareMuP.py --section {args.section} {train}{val}{test}"
        utput = os.system(command)
        
    if not args.noner:
        for k, v in data_split.items():
            if v: getNER(args.section, k, args.gpu)        
        
    if not args.norel:
        for k, v in data_split.items():
            if v: getRE(args.section, k, args.gpu)    


if __name__ == "__main__":
    main()