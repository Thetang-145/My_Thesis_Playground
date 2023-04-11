import os
from pathlib import Path

def prepareCsv(file_path, doc_col, sentence_col, output_filename=None, output_dir=None):
    command = f"python3 _prepareCsv.py --file_path {file_path} \
    --doc_col {doc_col} \
    --sentence_col {sentence_col} "
    if output_filename is not None: command += f"--output_filename {output_filename}  "
    if output_dir is not None: command += f"--output_dir {output_dir}  "
    return os.system(command)
        
def getNER(filename, gpu=0):
    return os.system(f"CUDA_VISIBLE_DEVICES={gpu} python3  _getNER.py  --model_type bertspanmarker  \
    --model_name_or_path  pretrained_model/sciner-scibert  --do_lower_case  \
    --data_dir _prepared_data/csv  \
    --learning_rate 2e-5  --num_train_epochs 50  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
    --max_seq_length 512  --save_steps 2000  --max_pair_length 256  --max_mention_ori_length 8    \
    --do_eval  --evaluate_during_training   --eval_all_checkpoints  \
    --fp16  --seed 42  --onedropout  --lminit  \
    --test_file {filename}.jsonl  \
    --output_dir _sciner_models/csv  --output_file {filename}_ner.json \
    --overwrite_output_dir  --output_results")

def getRE(filename, gpu=0):
    return os.system(f"CUDA_VISIBLE_DEVICES={gpu} python3 _getRE.py  --model_type bertsub  \
    --model_name_or_path /pretrained_model/scire-scibert  --do_lower_case  \
    --data_dir _sciner_models/csv  \
    --learning_rate 2e-5  --num_train_epochs 10  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
    --max_seq_length 256  --max_pair_length 16  --save_steps 2500  \
    --do_eval  --eval_all_checkpoints  --eval_logsoftmax  \
    --fp16   \
    --test_file {filename}_ner.json  \
    --use_ner_results \
    --output_dir _scire_models/csv  --output_file {filename}_re.json")


def main():    
    main_path = str((Path().absolute()))
    models = [
        'bart-large', 
        'bart-large-cnn'
    ]
    experiments = [
        ['abstract-kg', 'abstract-kg'],
        ['summary-kg', 'abstract-kg'],
        ['summary-kg', 'summary-kg'],
    ]
    best_score = ['output_bestRouge1', 'output_bestRougeAvg']
    exclude = [
        'bart-large-cnn_abstract-kg_abstract-kg',
        'bart-large-cnn_summary-kg_summary-kg',
    ]
    GPU = 1
    temp_file = "finish.txt"
    if not os.path.exists(temp_file): open(temp_file, "w").close()
    with open(temp_file, "r") as file: Finished = [line.strip() for line in file]
    # for fin in Finished:
    #     print(fin)
    # exit()

    
    for model in models:
        for exp in experiments:
            for best in best_score:
                if f"{model}_{exp[0]}_{exp[1]}" in exclude: 
                    output_filename=f"[{model}]_[{exp[0]}]_[{exp[1]}]"
                else:
                    output_filename=f"[{model}]_[{exp[0]}]_[{exp[1]}]_{best[11:]}"
                if output_filename in Finished: 
                    print(f"{'*'*10} PASS {output_filename} BECAUSE IT WAS SUCCESSFULLY EXTRACTED {'*'*10}")
                    continue
                prepareCsv(
                    file_path=f"/FT_PT_model/generated_summary/val/{model}/MODEL-{exp[0]}_EVAL-{exp[1]}.csv", 
                    doc_col="paper_id", 
                    sentence_col=best, 
                    output_filename=output_filename,
                )
                getNER(output_filename, GPU)
                getRE(output_filename, GPU)
                # with open("finish.txt", "a") as file: file.write(f"{output_filename}\n")
                if f"{model}_{exp[0]}_{exp[1]}" in exclude: break

    # result_file = f"/FT_PT_model/generated_summary/val/{model}/MODEL-{modelInput}_EVAL-{evalInput}.csv"
    # doc_col = "paper_id"
    # sentence_col = "output_bestRouge1"
    # prepareCsv(file_path, doc_col="paper_id", sentence_col):
    # file_path 
                
if __name__ == "__main__":
    main()