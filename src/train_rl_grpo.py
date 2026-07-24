#!/usr/bin/env python
"""
GRPO Training Script for ChessLM

Group Relative Policy Optimization (GRPO) with Stockfish reward signal.
Uses TRL's GRPOTrainer for proper multi-GPU and W&B support.
"""

import argparse
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stockfish_reward import StockfishReward, compute_reward, reset_stockfish


@dataclass
class ModelArguments:
    model_name: str = field(
        default="meta-llama/Llama-3.2-1B",
        metadata={"help": "Pretrained model name or path"},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Enable remote code execution"},
    )


@dataclass
class DataArguments:
    dataset_path: str = field(
        default="./splits",
        metadata={"help": "Path to dataset or Hugging Face dataset name"},
    )
    train_split: str = field(
        default="train",
        metadata={"help": "Dataset split to use for training"},
    )
    val_split: Optional[str] = field(
        default="validation",
        metadata={"help": "Dataset split to use for validation"},
    )
    max_seq_length: int = field(
        default=1024,
        metadata={"help": "Maximum sequence length"},
    )


def apply_chat_template(example, tokenizer):
    """Apply chat template to the conversation and extract components."""
    messages = example.get("messages", [])

    if not messages:
        task = example.get("task", "")
        kind = example.get("KIND", "")
        input_text = example.get("input", "")
        output_text = example.get("expected_output", example.get("output", ""))


        if input_text:
            system_prompt = f"You are an expert chess assistant. Your task is: {task}\n\nThe move classification is: {kind}"

            # Add format instructions based on task type
            if kind == "FIND_NEXT_BEST_MOVE":
                input_text = f'{input_text} Respond in a json in the format {{"next best move": "<move>"}}'
            elif kind == "MLM_ON_MOVES":
                input_text = f'{input_text} Respond in a json in the format {{"missing moves": ["<move1>", "<move2>", ...]}}'
            elif kind == "FIND_ADVANTAGED_PLAYER":
                input_text = f'{input_text} Respond in a json in the format {{"Most advantaged": "<white/black/draw>"}}'
            elif kind == "FIND_FINAL_SCORE":
                input_text = f'{input_text} Respond in a json in the format {{"score": "<score>"}}'
            elif kind == "FIND_LAST_MOVE":
                input_text = f'{input_text} Respond in a json in the format {{"missing move": "<move>"}}'
            elif kind == "SORT_FENS":
                input_text = f'{input_text} Respond in a json in the format {{"sorted FENs": ["<FEN1>", "<FEN2>", ...]}}'
            else:
                input_text = f'{input_text} Respond in a json'

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": output_text},
            ]

    # Apply template with add_generation_prompt=True for RL
    # enable_thinking=False for faster JSON-only generation (no thinking tokens)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    return {"text": text}


def preprocess_dataset(dataset, tokenizer, max_seq_length):
    """Preprocess the dataset for training.

    For GRPO, we need to:
    1. Extract prompt messages (without assistant response) for training
    2. Keep metadata (category, moves, fens) for reward computation

    Note: TRL's GRPOTrainer handles tokenization internally using apply_chat_template.
    We return the messages list so GRPOTrainer can apply the chat template with add_generation_prompt=True.
    """
    import json

    # Extract prompt messages and metadata directly
    def extract_components(example):
        # Get data from example
        task = example.get("task", "")
        kind = example.get("KIND", "")
        input_text = example.get("input", "")
        expected_output = example.get("expected_output", example.get("output", ""))

        # Parse moves from input JSON (MLM_ON_MOVES task)
        # The input contains FEN and moves as a JSON string
        moves = []
        fens = []
        try:
            input_json = json.loads(input_text)
            if "moves" in input_json:
                moves = input_json["moves"]
            if "FEN" in input_json:
                fens = [input_json["FEN"]]
        except (json.JSONDecodeError, TypeError):
            # Use top-level moves/fens if available
            moves = example.get("moves", moves)
            fens = example.get("FENs", fens)

        # Determine whose turn it is based on move count
        # Even number of moves = White to move, odd = Black to move
        total_moves = len(moves)
        side_to_move = "White" if total_moves % 2 == 0 else "Black"

        # Build messages from the available fields
        messages = []
        if task or input_text:
            system_prompt = f"You are an expert chess assistant."
            if task:
                system_prompt += f" Your task is: {task}"
            if kind:
                system_prompt += f"\n\nThe move classification is: {kind}"
                # Add format instructions based on task type
                if kind == "FIND_NEXT_BEST_MOVE":
                    input_text = f'{input_text}\n\nSide to move: {side_to_move}\n\nRespond in a json in the format {{"next best move": "<move>"}}'
                elif kind == "MLM_ON_MOVES":
                    input_text = f'{input_text} Respond in a json in the format {{"missing moves": ["<move1>", "<move2>", ...]}}'
                elif kind == "FIND_ADVANTAGED_PLAYER":
                    input_text = f'{input_text}\n\nSide to move: {side_to_move}\n\nRespond in a json in the format {{"Most advantaged": "<white/black/draw>"}}'
                elif kind == "FIND_FINAL_SCORE":
                    input_text = f'{input_text}\n\nSide to move: {side_to_move}\n\nRespond in a json in the format {{"score": "<score>"}}'
                elif kind == "FIND_LAST_MOVE":
                    input_text = f'{input_text}\n\nSide to move: {side_to_move}\n\nRespond in a json in the format {{"missing move": "<move>"}}'
                elif kind == "SORT_FENS":
                    input_text = f'{input_text} Respond in a json in the format {{"sorted FENs": ["<FEN1>", "<FEN2>", ...]}}'
                else:
                    input_text = f'{input_text} Respond in a json'

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text},
            ]

        # GRPOTrainer will apply chat_template with add_generation_prompt=True
        # We return the messages list so GRPOTrainer can apply the template correctly
        return {
            "prompt": messages,
            "category": kind,
            "moves": moves,
            "fens": fens,
        }

    dataset = dataset.map(
        extract_components,
        batched=False,
        remove_columns=[col for col in dataset.column_names if col not in ["prompt", "category", "moves", "fens"]],
    )

    return dataset


def stockfish_reward_func(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """Reward function that uses Stockfish to compute rewards.

    TRL passes the metadata columns as keyword arguments:
    - category: chess task category
    - moves: list of moves
    - fens: list of FEN positions
    """
    # Get Stockfish from the trainer
    stockfish = None
    if hasattr(stockfish_reward_func, "stockfish"):
        stockfish = stockfish_reward_func.stockfish

    if stockfish is None:
        # Fall back to creating a new instance (not ideal but works)
        stockfish = StockfishReward(depth=15, threads=1)

    # Extract metadata from kwargs
    categories = kwargs.get("category", [None] * len(prompts))
    moves_list = kwargs.get("moves", [[]] * len(prompts))
    fens_list = kwargs.get("fens", [[]] * len(prompts))

    rewards = []
    for i, (completion, category, moves, fens) in enumerate(zip(completions, categories, moves_list, fens_list)):
        # Debug: print first few completions
        if i == 0 and isinstance(completion, str) and len(completion) > 100:
            print(f"=== DEBUG COMPLETION (first 200 chars) ===")
            print(repr(completion[:200]))
            print()

        # Convert completion to string if needed (defense against token IDs)
        if not isinstance(completion, str):
            completion = str(completion)

        reward = compute_reward(
            completion,
            category or "",
            stockfish,
            moves=moves or [],
            fens=fens or [],
        )

        # Penalty for reasoning - continuous signal based on reasoning length
        # Exponential penalty increases gradually after 500 tokens
        if "<|im_start|>assistant\n" in completion and "<think>" in completion:
            import re
            match = re.search(r'<|im_start|>assistant\n<think>\n(.*?)\n</think>', completion, re.DOTALL)
            if match:
                reasoning_text = match.group(1)
                # Count tokens approximately (1 token ~ 4 chars in English)
                reasoning_tokens = len(reasoning_text) / 4

                # Continuous penalty: 0 up to 500 tokens, then grows exponentially
                if reasoning_tokens > 500:
                    # Exponential penalty - grows slowly at first, then faster
                    overage = reasoning_tokens - 500
                    penalty = 0.1 * (1 - math.exp(-overage / 300))
                    reward -= penalty

        rewards.append(reward)

    return rewards


def main():
    parser = argparse.ArgumentParser()

    # Model arguments
    parser.add_argument("--model_name", type=str, default="./outputs/qwen-3.5-2b-sft-chess-lr-1e-5-bs4-epochs3/checkpoint-5940", help="Model name or path for RL training")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)

    # GRPO arguments
    parser.add_argument("--num_samples_per_prompt", type=int, default=2)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)

    # Stockfish arguments
    parser.add_argument("--stockfish_path", type=str, default="stockfish")
    parser.add_argument("--stockfish_depth", type=int, default=10)
    parser.add_argument("--stockfish_threads", type=int, default=1)

    # Data arguments
    parser.add_argument("--dataset_path", type=str, default="./splits")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="validation")
    parser.add_argument("--max_seq_length", type=int, default=32768)

    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./outputs/llama-3.2-1b-rl-grpo")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    # For faster training with batch_size > 1: effective_batch = batch_size * grad_accum_steps
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--bf16", action="store_true")

    # Token budget arguments
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum new tokens to generate per rollout")
    parser.add_argument("--token_budget", type=int, default=None, help="Maximum total tokens per batch (approximate)")

    # W&B arguments
    parser.add_argument("--wandb_project", type=str, default="chesslm")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--run_name", type=str, default=None)

    args = parser.parse_args()

    # Set W&B environment variables early (before any imports that might trigger W&B)
    os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity

    # Initialize Stockfish
    print("Initializing Stockfish...")
    stockfish = StockfishReward(
        path=args.stockfish_path,
        depth=args.stockfish_depth,
        threads=args.stockfish_threads,
    )

    # Set stockfish on reward function
    stockfish_reward_func.stockfish = stockfish

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Use the chat template from the tokenizer itself with thinking disabled
    # For RL, we use enable_thinking=False to avoid the model spending tokens on thinking content
    # The SFT checkpoint outputs JSON directly without thinking blocks
    print("Using SFT Qwen 2B with thinking disabled for RL")

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
    )

    # Load dataset
    print("Loading dataset...")
    # Check if dataset_path is a directory (saved with save_to_disk)
    if os.path.isdir(args.dataset_path):
        dataset = load_from_disk(args.dataset_path)
        # load_from_disk returns a DatasetDict with the same split names as saved
        # When we saved with DatasetDict, it should have train/val splits
    else:
        dataset = load_dataset(args.dataset_path)

    # Get training split
    train_split_name = args.train_split
    # Check if train_split exists, otherwise use "train"
    if train_split_name not in dataset:
        if "train" in dataset:
            train_split_name = "train"
        else:
            # For single Dataset objects, use it directly
            train_dataset = dataset
            dataset = None  # No eval dataset available

    if dataset is not None:
        train_dataset = dataset[train_split_name]

        # Get validation split
        eval_dataset = None
        val_split_name = args.val_split
        if val_split_name and val_split_name in dataset:
            eval_dataset = dataset[val_split_name]
        elif "validation" in dataset:
            eval_dataset = dataset["validation"]
    else:
        # Single dataset - no eval split available
        eval_dataset = None

    # Prepare datasets
    print("Preprocessing dataset...")
    train_dataset = preprocess_dataset(train_dataset, tokenizer, args.max_seq_length)
    if eval_dataset is not None:
        eval_dataset = preprocess_dataset(eval_dataset, tokenizer, args.max_seq_length)

    # GRPO Config with proper W&B settings
    # Remove unused columns=False is critical - TRL needs prompt column for generation
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        bf16=args.bf16,
        report_to="wandb",
        run_name=args.run_name,
        logging_dir=f"{args.output_dir}/logs",
        num_generations=args.num_samples_per_prompt,
        beta=args.beta,
        epsilon=args.epsilon,
        remove_unused_columns=False,  # TRL needs the prompt column
        log_completions=True,  # Log sample completions to W&B
        max_completion_length=args.max_new_tokens,  # Maximum new tokens for generation
    )

    # Initialize GRPO Trainer from TRL
    print("Initializing GRPO Trainer...")
    trainer = GRPOTrainer(
        args=training_args,
        model=model,
        processing_class=tokenizer,
        reward_funcs=stockfish_reward_func,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Start training
    print("Starting GRPO training...")
    trainer.train()

    # Save final model
    print("Saving final model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Cleanup
    stockfish.close()
    reset_stockfish()

    print(f"\nTraining complete! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()