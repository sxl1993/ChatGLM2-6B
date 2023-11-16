# !/bin/bash

PRE_SEQ_LEN=128
LR=2e-2

XPU_VISIBLE_DEVICES=7 XACC=1 python3 main.py \
    --do_train \
    --train_file AdvertiseGen/train.json \
    --validation_file AdvertiseGen/dev.json \
    --prompt_column content \
    --response_column summary \
    --preprocessing_num_workers 1 \
    --overwrite_cache \
    --model_name_or_path ./chatglm2-6b \
    --output_dir output/adgen-chatglm2-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 128 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --predict_with_generate \
    --max_steps 100 \
    --logging_steps 1 \
    --save_steps 1000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --save_total_limit 1 \
    --report_to "none"
