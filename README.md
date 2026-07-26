# ChessLM - Qwen-3.5-2B Chess Assistant

Fine-tuning Qwen-3.5-2B for chess move prediction using Supervised Fine-Tuning (SFT) and Reinforcement Learning (GRPO) with Stockfish as the reward signal.

## Pre-trained Models

**Hugging Face Collection**: [ChessLM](https://huggingface.co/collections/GL3MON/chesslm)

| Model | Description | Type |
|-------|-------------|------|
| Qwen3.5-2B-chess-finetuned | SFT checkpoint trained on ChessInstruct | Supervised Fine-Tuning |
| Qwen3.5-2B-chess-rl-grpo | RL fine-tuned using GRPO with Stockfish reward | Reinforcement Learning |

See the [RL RECIPE](./RL_RECIPE.md) for training details.

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

### 3. Login to Hugging Face (for training only)

```bash
hf auth login
```

### 4. Using Pre-trained Models (Inference)

To use the pre-trained model without training:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load from Hugging Face
model_name = "GL3MON/Qwen3.5-2B-chess-finetuned"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Generate chess moves
messages = [
    {"role": "system", "content": "You are a chess assistant."},
    {"role": "user", "content": "What is the best move for white in e2e4?"}
]

input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
outputs = model.generate(input_ids, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
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

## License

MIT License
