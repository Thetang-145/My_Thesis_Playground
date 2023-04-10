python3 finetune.py --genSum --model bart-large --section summary
python3 finetune.py --genSum --model bart-large --section abstract 

python3 generate.py \
    --model bart-large \
    --modelSection summary \
    --evalSection abstract