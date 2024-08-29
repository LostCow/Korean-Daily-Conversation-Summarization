#!/bin/bash

BASE_MODEL_NAME="/data/yohan/ko-gemma-9b"
TAPT_OUTPUT_DIR="ko-gemma-2-9b-it-tapt"

deepspeed korean_dcs_2024/trainer/train.py \
    --deepspeed korean_dcs_2024/trainer/ds_config_zero_2.json \
    --output_dir=$TAPT_OUTPUT_DIR \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=4 \
    --gradient_accumulation_steps=8 \
    --learning_rate=2e-5 \
    --weight_decay=0.01 \
    --adam_beta2=0.95 \
    --num_train_epochs=5 \
    --lr_scheduler_type=cosine \
    --warmup_ratio=0.01 \
    --logging_steps=1 \
    --bf16 \
    --tf32=True \
    --dataloader_num_workers=8 \
    --optim=adamw_torch_fused \
    --report_to=none \
    --eval_strategy=epoch \
    --save_strategy=epoch \
    --model_name_or_path=$BASE_MODEL_NAME \
    --model_max_length=4096 \
    --cache_dir=/data/.cache \
    --gradient_checkpointing \
    --save_only_model \
    --neftune_noise_alpha 5.0 \
    --predict_with_generate \
    --use_peft \
    --quantization=4bit \
    --do_tapt

python merge_lora.py --peft_model_name_or_path $TAPT_OUTPUT_DIR --base_model_name_or_path $BASE_MODEL_NAME
