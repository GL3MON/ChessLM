# ChessLM RL Training Recipe

This document describes the Reinforcement Learning (RL) training setup for ChessLM using Group Relative Policy Optimization (GRPO) with Stockfish as the reward signal.

## Overview

The RL training pipeline fine-tunes an SFT (Supervised Fine-Tuning) model using:
- **Algorithm:** Group Relative Policy Optimization (GRPO)
- **Reward Signal:** Stockfish chess engine (depth 10-15)
- **Framework:** TRL (Transformers RL)

## Architecture

```
┌─────────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   SFT Model         │────>│   GRPO Trainer   │────>│  Stockfish RL    │
│ (Qwen-3.5-2B)       │     │   (TRL)          │     │  Reward Signal   │
└─────────────────────┘     └──────────────────┘     └──────────────────┘
```

## Key Components

### 1. GRPO Trainer (`src/train_rl_grpo.py`)

The main RL training script uses Hugging Face's TRL library with GRPO.

**Key Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_samples_per_prompt` | 4 | Number of completions per prompt (batch size factor) |
| `beta` | 0.1 | KL divergence control parameter |
| `epsilon` | 0.2 | PPO-style clipping range |
| `learning_rate` | 1e-6 | Optimizer learning rate |
| `num_epochs` | 1 | Training epochs |
| `max_new_tokens` | 256 | Maximum tokens to generate per rollout |

### 2. Stockfish Reward (`src/stockfish_reward.py`)

The reward function computes rewards based on Stockfish's evaluation. Different reward functions for each task category:

| Category | Reward Function | Description |
|----------|-----------------|-------------|
| `FIND_NEXT_BEST_MOVE` | `compute_reward_find_next_best_move` | Evaluates the position after the predicted move |
| `FIND_ADVANTAGED_PLAYER` | `compute_reward_find_advantaged_player` | Checks if correctly identified the advantaged player |
| `FIND_FINAL_SCORE` | `compute_reward_find_final_score` | Validates predicted game result |
| `MLM_ON_MOVES` | `compute_reward_mlm_on_moves` | Verifies predicted missing moves |
| `FIND_LAST_MOVE` | `compute_reward_find_last_move` | Checks if last move is in top Stockfish options |
| `SORT_FENS` | `compute_reward_sort_fens` | Validates FEN ordering by eval |

**Reward Scaling:**
- Scores are clamped to ±5000 centipawns
- Normalized to ±1.0 range using `tanh(score/200)`
- Stronger advantages yield higher reward differentiation

This prevents the model from spending tokens on reasoning blocks, focusing on JSON output.

## Training Configuration

### GRPO Hyperparameters

**File:** `configs/grpo_config.json`

```json
{
  "beta": 0.1,
  "epsilon": 0.2,
  "num_samples_per_prompt": 4,
  "learning_rate": 1e-6,
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 2,
  "max_seq_length": 1024,
  "max_completion_length": 256
}
```

### Launch Script (`scripts/train_rl.sh`)

```bash
# Key environment variables
export WANDB_PROJECT="chesslm"
export WANDB_RUN_GROUP="rl-grpo"

# Model and data paths
MODEL_NAME="outputs/qwen-3.5-2b-sft-chess-lr-1e-5-bs4-epochs3/checkpoint-5940"
DATASET_PATH="./data/splits/rl"

# Stockfish config
STOCKFISH_PATH="./bin/stockfish"
STOCKFISH_DEPTH=10
STOCKFISH_THREADS=1

# Training hyperparameters
NUM_SAMPLES_PER_PROMPT=4
BETA=0.1
EPSILON=0.2
LEARNING_RATE=1e-6
```

## Training Pipeline

### Step 1: SFT Pre-training

```bash
./scripts/train_sft.sh
```

This produces a baseline model at `checkpoint-5940`.

### Step 2: RL Fine-tuning

```bash
./scripts/train_rl.sh
```

**What happens:**
1. Loads SFT checkpoint
2. Initializes Stockfish engine
3. Preprocesses dataset (extracts prompts and metadata)
4. Trains with GRPO using Stockfish reward
5. Saves model checkpoint periodically

## Key Metrics (Logged to W&B)

| Metric | Description |
|--------|-------------|
| `loss` | GRPO loss |
| `kl` | KL divergence from reference policy |
| `reward` | Mean Stockfish reward |
| `reward_std` | Reward standard deviation |
| `clip_ratio/*` | Fraction of samples at policy clip limits |
| `completions/*` | Length and termination stats |
| `entropy` | Response entropy (lower = more focused) |

## Reward Engineering Notes

### Continuous Reward Signals

Instead of binary success/failure, rewards are continuous:

```python
# Example: FIND_NEXT_BEST_MOVE
reward = _score_to_reward_scaled(eval_score, side_to_move_is_white)
# Maps Stockfish cp → [-1.0, 1.0] with smooth gradients
```

### Reasoning Penalty

To discourage excessive thinking tokens:

```python
if reasoning_tokens > 500:
    penalty = 0.1 * (1 - math.exp(-overage / 300))
    reward -= penalty
```

### Reward Normalization

The `compute_reward` function:
1. Extracts predicted moves from JSON (handles thinking blocks)
2. Sets Stockfish position with provided moves
3. Computes Stockfish evaluation
4. Scales to [-1.0, 1.0] using `tanh(score/200)`
5. Returns category-specific reward

## Troubleshooting

### Common Issues

**1. Stockfish connection errors**
```python
# Ensure Stockfish is executable
chmod +x bin/stockfish
# Verify path in config
STOCKFISH_PATH="./bin/stockfish"
```

**2. GPU OOM**
```bash
# Reduce batch size
--per_device_train_batch_size 1
--gradient_accumulation_steps 4
# Or reduce max completion length
--max_new_tokens 128
```

**3. No learning progress**
- Check KL is small (< 0.01) - if high, reduce beta
- Verify reward signal is non-zero
- Ensure learning rate is sufficient (try 5e-6)

### Debug Mode

Add logging to `stockfish_reward.py`:
```python
print(f"=== DEBUG COMPLETION ===")
print(repr(completion[:200]))
print(f"Category: {category}")
```

## Performance Targets

| Metric | Target |
|--------|--------|
| Epoch time | ~2-4 hours (8xA100) |
| Reward improvement | +0.1 per 1000 steps |
| KL divergence | < 0.01 after warmup |
| Completion length | 20-60 tokens |

## Citation

If using this RL setup, please cite:

```bibtex
@software{ChessLM2026,
  title = {ChessLM - Qwen-3.5-2B Chess Assistant},
  author = {GL3MON},
  year = {2026},
  url = {https://github.com/GL3MON/ChessLM}
}
```

## References

- [TRL GRPO Documentation](https://huggingface.co/docs/trl/en/grpo_trainer)
- [Stockfish Engine](https://github.com/official-stockfish/Stockfish)
- [Qwen Model Family](https://huggingface.co/Qwen)