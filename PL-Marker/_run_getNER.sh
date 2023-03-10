CUDA_VISIBLE_DEVICES=0  python3  _getNER.py  --model_type bertspanmarker  \
    --model_name_or_path  pretrained_model/sciner-scibert  --do_lower_case  \
    --data_dir _prepared_data/summary  \
    --learning_rate 2e-5  --num_train_epochs 50  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
    --max_seq_length 512  --save_steps 2000  --max_pair_length 256  --max_mention_ori_length 8    \
    --do_eval  --evaluate_during_training   --eval_all_checkpoints  \
    --fp16  --seed 42  --onedropout  --lminit  \
    --test_file val.jsonl  \
    --output_dir _sciner_models/summary  --output_file val_ner.json \
    --overwrite_output_dir  --output_results
    
# CUDA_VISIBLE_DEVICES=0  python3  _getNER.py  --model_type bertspanmarker  \
#     --model_name_or_path  pretrained_model/sciner-scibert  --do_lower_case  \
#     --data_dir _prepared_data/summary  \
#     --learning_rate 2e-5  --num_train_epochs 50  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
#     --max_seq_length 512  --save_steps 2000  --max_pair_length 256  --max_mention_ori_length 8    \
#     --do_eval  --evaluate_during_training   --eval_all_checkpoints  \
#     --fp16  --seed 42  --onedropout  --lminit  \
#     --test_file train.jsonl  \
#     --output_dir _sciner_models/summary  --output_file train_ner.json \
#     --overwrite_output_dir  --output_results