#!/bin/bash
# Llama-3.2-1B SFT Launcher for 8 GPUs
# Usage: ./train_sft.sh

set -e

# Configuration
MODEL_NAME="meta-llama/Llama-3.2-1B"
DATASET_PATH="./splits"  # Path to your split dataset
SPLIT_NAME="sft_train"   # Or "sft_train" with --val_split "sft_val"
OUTPUT_DIR="./outputs/llama-3.2-1b-sft"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Launch training with Accelerate
accelerate launch \
    --config_file "deepspeed_config.json" \
    --num_processes 8 \
    --main_process_port 29500 \
    finetune_llama.py \
    --model_name "$MODEL_NAME" \
    --dataset_path "$DATASET_PATH" \
    --train_split "$SPLIT_NAME" \
    --val_split "sft_val" \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --logging_steps 10 \
    --eval_steps 100 \
    --save_steps 500 \
    --warmup_steps 100 \
    --weight_decay 0.01 \
    --lr_scheduler_type "cosine" \
    --bf16 \
    --gradient_checkpointing