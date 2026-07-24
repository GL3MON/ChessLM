#!/bin/bash
# Qwen-3.5-2B RL GRPO Launcher for 8 GPUs
# Usage: ./train_rl.sh

set -e

# Change to project root directory
cd /hfcache/harissh/ChessLM

# Configuration
# Note: SFT training should complete first and produce checkpoint-5940
# SFT script: ./scripts/train_sft.sh
MODEL_NAME="outputs/qwen-3.5-2b-sft-chess-lr-1e-5-bs4-epochs3/checkpoint-5940"
DATASET_PATH="./data/splits/rl"
DATASET_PATH_SFT="./data/splits"
RUN_NAME="qwen-3.5-2b-rl-grpo-beta-0.1-lr-1e-6-chk-5940"
OUTPUT_DIR="./outputs/${RUN_NAME}"

# W&B Configuration
export WANDB_API_KEY="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"
export WANDB_PROJECT="chesslm"
export WANDB_RUN_GROUP="rl-grpo"
export WANDB_NAME="$RUN_NAME"

# Stockfish configuration
STOCKFISH_PATH="./bin/stockfish"
STOCKFISH_DEPTH=10
STOCKFISH_THREADS=1

# GRPO hyperparameters
NUM_SAMPLES_PER_PROMPT=4
BETA=0.1
EPSILON=0.2
LEARNING_RATE=1e-6
NUM_EPOCHS=1

# Training arguments
PER_DEVICE_BATCH_SIZE=2
GRADIENT_ACCUMULATION=2
WARMUP_STEPS=100
LOGGING_STEPS=10
SAVE_STEPS=500

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Launch training with Accelerate
accelerate launch \
    --config_file "configs/accelerate_dp_config.json" \
    src/train_rl_grpo.py \
    --model_name "$MODEL_NAME" \
    --dataset_path "$DATASET_PATH" \
    --train_split "train" \
    --val_split "validation" \
    --output_dir "$OUTPUT_DIR" \
    --stockfish_path "$STOCKFISH_PATH" \
    --stockfish_depth "$STOCKFISH_DEPTH" \
    --stockfish_threads "$STOCKFISH_THREADS" \
    --num_samples_per_prompt "$NUM_SAMPLES_PER_PROMPT" \
    --beta "$BETA" \
    --epsilon "$EPSILON" \
    --learning_rate "$LEARNING_RATE" \
    --num_epochs "$NUM_EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION" \
    --max_seq_length 1024 \
    --logging_steps "$LOGGING_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --warmup_steps "$WARMUP_STEPS" \
    --weight_decay 0.01 \
    --lr_scheduler_type "cosine" \
    --bf16 \
    --wandb_project "chesslm" \
    --wandb_entity "" \
    --run_name "$RUN_NAME"