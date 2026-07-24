#!/bin/bash
# Qwen-3.5-2B SFT Launcher for 8 GPUs (Accelerate Data Parallel)
# Usage: ./train_sft.sh

set -e

# Change to project root directory
cd /hfcache/harissh/ChessLM

# W&B Configuration
export WANDB_API_KEY="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"
export WANDB_PROJECT="chesslm"

# Configuration
MODEL_NAME="Qwen/Qwen3.5-2B"
DATASET_PATH="./data/splits"
SPLIT_NAME="sft_train"
RUN_NAME="qwen-3.5-2b-sft-chess-lr-1e-5-bs4-epochs3-fixed-prompt"
OUTPUT_DIR="./outputs/${RUN_NAME}"

mkdir -p "$OUTPUT_DIR"

# Launch with Accelerate
accelerate launch \
    --config_file "configs/accelerate_dp_config.json" \
    src/finetune.py \
    --model_name "$MODEL_NAME" \
    --dataset_path "$DATASET_PATH" \
    --train_split "train" \
    --val_split "validation" \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --logging_steps 10 \
    --eval_steps 100 \
    --save_steps 500 \
    --warmup_steps 100 \
    --weight_decay 0.01 \
    --lr_scheduler_type "cosine" \
    --bf16 \
    --gradient_checkpointing \
    --report_to "wandb" \
    --run_name "$RUN_NAME" \
    --wandb_project "chesslm" \
    --wandb_entity ""