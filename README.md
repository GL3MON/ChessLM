# ChessLM - Qwen-3.5-2B Chess Assistant

Fine-tuning Qwen-3.5-2B for chess move prediction using Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) with Stockfish as the reward signal.

## Project Structure

```
ChessLM/
├── src/                        # Python source code
│   ├── finetune.py             # SFT training script
│   ├── train_rl_grpo.py        # GRPO RL training script
│   └── stockfish_reward.py     # Stockfish reward wrapper
├── scripts/                    # Launch scripts
│   ├── split_dataset.py        # Data splitting script
│   ├── train_sft.sh            # SFT launch script (Accelerate DP)
│   ├── train_sft_deepspeed.sh  # SFT launch script (DeepSpeed)
│   └── train_rl.sh             # RL launch script
├── configs/                    # Accelerate/Deepspeed configs
│   ├── accelerate_dp_config.json
│   ├── deepspeed_config.json
│   └── grpo_config.json
├── templates/                  # Chat templates
│   └── llama3_chat_template.jinja
├── bin/                        # Binaries (includes Stockfish)
│   └── stockfish               # Stockfish binary
├── data/                       # Dataset splits (created during split)
├── outputs/                    # Training outputs (models, logs)
├── wandb/                      # Weights & Biases artifacts
└── README.md
```

## Setup

### 1. Install Dependencies

```bash
pip install transformers datasets peft accelerate torch stockfish wandb trl
```

### 2. Stockfish

Stockfish binary is included in `./bin/`. Make sure it's executable:
```bash
chmod +x bin/stockfish
```

Or download manually:
```bash
curl -L https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-ubuntu-x86-64-avx2.tar -o stockfish.tar
tar -xf stockfish.tar
mv stockfish-ubuntu-x86-64-avx2 bin/stockfish
```

### 3. Login to Hugging Face

```bash
huggingface-cli login
```

## Training Pipeline

### Step 1: Split Dataset

```bash
python scripts/split_dataset.py \
    --dataset_path Thytu/ChessInstruct \
    --with-validation \
    --output_dir data/splits
```

### Step 2: Supervised Fine-Tuning (SFT)

```bash
./scripts/train_sft.sh
```

### Step 3: Reinforcement Learning (RL) with Stockfish

```bash
./scripts/train_rl.sh
```

## Task Categories

| Category | Description | Reward Signal |
|----------|-------------|---------------|
| `FIND_NEXT_BEST_MOVE` | Predict the next best move | Stockfish best move comparison |
| `FIND_ADVANTAGED_PLAYER` | Identify who has advantage | Stockfish eval sign |
| `FIND_FINAL_SCORE` | Predict game result | Stockfish eval magnitude |
| `MLM_ON_MOVES` | Fill in missing moves | Stockfish move verification |
| `FIND_LAST_MOVE` | Find the final move | Stockfish best move |
| `SORT_FENS` | Sort positions by game order | Stockfish eval trajectory |

## Configuration

- **SFT**: Uses Qwen-3.5-2B with DeepSpeed ZeRO Stage 2 or Accelerate DP
- **RL**: Uses GRPO with Stockfish depth=10-15, 4 samples per prompt
- **Precision**: BF16 for training

## Logging

Training logs are tracked with Weights & Biases (W&B) under the `chesslm` project.

## License

MIT License