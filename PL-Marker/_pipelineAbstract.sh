python3 _prepareMuP.py  --section abstract  --output_dir _prepared_data

# ****************************** Entity Extraction ******************************

# CUDA_VISIBLE_DEVICES=0  python3  _getNER.py  --model_type bertspanmarker  \
#     --model_name_or_path  pretrained_model/sciner-scibert  --do_lower_case  \
#     --data_dir _prepared_data/abastract  \
#     --learning_rate 2e-5  --num_train_epochs 50  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
#     --max_seq_length 512  --save_steps 2000  --max_pair_length 256  --max_mention_ori_length 8    \
#     --do_eval  --evaluate_during_training   --eval_all_checkpoints  \
#     --fp16  --seed 42  --onedropout  --lminit  \
#     --test_file train.jsonl  \
#     --output_dir _sciner_models/abastract  --output_file train_ner.json \
#     --overwrite_output_dir  --output_results
    
CUDA_VISIBLE_DEVICES=0  python3  _getNER.py  --model_type bertspanmarker  \
    --model_name_or_path  pretrained_model/sciner-scibert  --do_lower_case  \
    --data_dir _prepared_data/abastract  \
    --learning_rate 2e-5  --num_train_epochs 50  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
    --max_seq_length 512  --save_steps 2000  --max_pair_length 256  --max_mention_ori_length 8    \
    --do_eval  --evaluate_during_training   --eval_all_checkpoints  \
    --fp16  --seed 42  --onedropout  --lminit  \
    --test_file val.jsonl  \
    --output_dir _sciner_models/abastract  --output_file val_ner.json \
    --overwrite_output_dir  --output_results
        
# CUDA_VISIBLE_DEVICES=0  python3  _getNER.py  --model_type bertspanmarker  \
#     --model_name_or_path  pretrained_model/sciner-scibert  --do_lower_case  \
#     --data_dir _prepared_data/abastract  \
#     --learning_rate 2e-5  --num_train_epochs 50  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
#     --max_seq_length 512  --save_steps 2000  --max_pair_length 256  --max_mention_ori_length 8    \
#     --do_eval  --evaluate_during_training   --eval_all_checkpoints  \
#     --fp16  --seed 42  --onedropout  --lminit  \
#     --test_file test.jsonl  \
#     --output_dir _sciner_models/abastract  --output_file test_ner.json \
#     --overwrite_output_dir  --output_results

# ****************************** Relation Extraction ******************************

# CUDA_VISIBLE_DEVICES=0  python3 _getRE.py  --model_type bertsub  \
#     --model_name_or_path /pretrained_model/scire-scibert  --do_lower_case  \
#     --data_dir _sciner_models/abastract  \
#     --learning_rate 2e-5  --num_train_epochs 10  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
#     --max_seq_length 256  --max_pair_length 16  --save_steps 2500  \
#     --do_eval  --eval_all_checkpoints  --eval_logsoftmax  \
#     --fp16   \
#     --test_file train_ner.json  \
#     --use_ner_results \
#     --output_dir _scire_models/abastract  --output_file train_re.json

CUDA_VISIBLE_DEVICES=0  python3 _getRE.py  --model_type bertsub  \
    --model_name_or_path /pretrained_model/scire-scibert  --do_lower_case  \
    --data_dir _sciner_models/abastract  \
    --learning_rate 2e-5  --num_train_epochs 10  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
    --max_seq_length 256  --max_pair_length 16  --save_steps 2500  \
    --do_eval  --eval_all_checkpoints  --eval_logsoftmax  \
    --fp16   \
    --test_file val_ner.json  \
    --use_ner_results \
    --output_dir _scire_models/abastract  --output_file val_re.json

# CUDA_VISIBLE_DEVICES=0  python3 _getRE.py  --model_type bertsub  \
#     --model_name_or_path /pretrained_model/scire-scibert  --do_lower_case  \
#     --data_dir _sciner_models/abastract  \
#     --learning_rate 2e-5  --num_train_epochs 10  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
#     --max_seq_length 256  --max_pair_length 16  --save_steps 2500  \
#     --do_eval  --eval_all_checkpoints  --eval_logsoftmax  \
#     --fp16   \
#     --test_file test_ner.json  \
#     --use_ner_results \
#     --output_dir _scire_models/abastract  --output_file test_re.json