#!/usr/bin/env python
"""
Qwen3-VL-2B-Instruct Fine-tuning Script
Designed for multi-GPU training with Accelerate + DeepSpeed/FSDP

Uses Qwen chat template from Hugging Face.
"""

import argparse
import os
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
import torch


@dataclass
class ModelArguments:
    model_name: str = field(
        default="Qwen/Qwen3.5-2B",
        metadata={"help": "Pretrained model name or path"},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Enable remote code execution"},
    )


@dataclass
class DataArguments:
    dataset_path: str = field(
        default="Thytu/ChessInstruct",
        metadata={"help": "Path to dataset or Hugging Face dataset name"},
    )
    train_split: str = field(
        default="sft_train",
        metadata={"help": "Dataset split to use for training"},
    )
    val_split: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset split to use for validation"},
    )
    max_seq_length: int = field(
        default=1024,
        metadata={"help": "Maximum sequence length"},
    )
    chat_template: str = field(
        default="llama3",
        metadata={"help": "Chat template to use"},
    )


@dataclass
class TrainingArgumentsCustom(TrainingArguments):
    optim: str = field(
        default="adamw_torch",
        metadata={"help": "Optimizer to use"},
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "Train batch size per device"},
    )
    per_device_eval_batch_size: int = field(
        default=1,
        metadata={"help": "Eval batch size per device"},
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={"help": "Gradient accumulation steps"},
    )
    learning_rate: float = field(
        default=1e-5,
        metadata={"help": "Learning rate"},
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "Number of training epochs"},
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Logging interval"},
    )
    eval_steps: int = field(
        default=100,
        metadata={"help": "Evaluation interval"},
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "Checkpoint interval"},
    )
    warmup_steps: int = field(
        default=100,
        metadata={"help": "Warmup steps"},
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay"},
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "Learning rate scheduler type"},
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "Use BF16 precision"},
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Use FP16 precision"},
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Use gradient checkpointing"},
    )
    report_to: str = field(
        default="none",
        metadata={"help": "Where to report metrics (none, tensorboard, wandb)"},
    )


def apply_chat_template(example, tokenizer):
    """Apply chat template to the conversation.

    For ChessInstruct:
    - System prompt: task description + KIND label
    - User: JSON with moves array
    - Assistant: JSON with next best move

    Uses add_generation_prompt=True with enable_thinking=False:
    - This makes the prompt end with: <|im_start|>assistant\n
    - The assistant content is included directly (no thinking tokens)
    - Model learns to generate: {"key": "value"} directly after <|im_start|>assistant\n
    """
    messages = example.get("messages", [])

    if not messages:
        # Extract input, expected output, task, and KIND
        task = example.get("task", "")
        kind = example.get("KIND", "")
        input_text = example.get("input", "")
        output_text = example.get("expected_output", example.get("output", ""))

        if input_text:
            # Build system prompt with task and KIND
            system_prompt = f"You are an expert chess assistant. Your task is: {task}\n\nThe move classification is: {kind}"

            # Build messages list
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": output_text},
            ]

    # Apply chat template with add_generation_prompt=True and enable_thinking=False
    # This ensures the assistant response goes directly after <|im_start|>assistant\n
    # WITHOUT the thinking block being added to the prompt
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    return {"text": text}


def preprocess_dataset(dataset, tokenizer, max_seq_length):
    """Preprocess the dataset for training."""

    # Apply chat template
    # Keep columns that are needed or will be preserved
    columns_to_keep = ["text"]
    existing_cols = [c for c in columns_to_keep if c in dataset.column_names]
    # If text doesn't exist yet (first map), keep all columns
    if "text" not in dataset.column_names:
        existing_cols = dataset.column_names

    dataset = dataset.map(
        lambda x: apply_chat_template(x, tokenizer),
        batched=False,
        remove_columns=[col for col in dataset.column_names if col not in existing_cols],
    )

    # Tokenize with label masking
    # Only the assistant's response should contribute to loss
    def tokenize_function(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )

        # Find where the assistant response starts and mask everything before it
        input_ids = result["input_ids"]
        labels = input_ids.copy()

        # Find where the assistant response starts and mask everything before it
        # With add_generation_prompt=True and enable_thinking=False, the format is:
        # ...<|im_start|>assistant\n
        # The assistant content (JSON) comes directly after

        input_ids = result["input_ids"]
        labels = input_ids.copy()

        # Find the position of '<|im_start|>assistant\n' - only train on content after this
        assistant_token_id = 248045  # <|im_start|>
        newline_token_id = 198  # \n

        assistant_pos = -1
        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] == assistant_token_id:
                # Find the newline after <|im_start|>assistant
                for j in range(i + 1, min(i + 10, len(input_ids))):
                    if input_ids[j] == newline_token_id:
                        assistant_pos = j + 1
                        break
                if assistant_pos > 0:
                    break

        # Mask everything before the assistant response
        if assistant_pos > 0:
            for i in range(assistant_pos):
                labels[i] = -100

        result["labels"] = labels
        return result

    dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )

    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Add arguments from dataclasses
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-2B")
    parser.add_argument("--dataset_path", type=str, default="./data/splits")
    parser.add_argument("--train_split", type=str, default="sft_train")
    parser.add_argument("--val_split", type=str, default="sft_val")
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--output_dir", type=str, default="./outputs/qwen-3.5-2b-sft-chess")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="chesslm")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--report_to", type=str, default="none")

    args = parser.parse_args()

    # Set precision flags
    if args.bf16:
        torch_dtype = torch.bfloat16
    elif args.fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=True,
    )

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Use the chat template from the tokenizer itself (no custom template needed)
    print(f"Using chat template from {args.model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    print("Loading dataset...")
    dataset = load_dataset(args.dataset_path)

    # Get the training split
    if args.train_split in dataset:
        train_dataset = dataset[args.train_split]
    else:
        # If split doesn't exist, use the whole dataset
        train_dataset = dataset

    # Handle validation split
    eval_dataset = None
    if args.val_split and args.val_split in dataset:
        eval_dataset = dataset[args.val_split]
        print(f"Using {args.val_split} for validation")
    elif args.val_split:
        print(f"Warning: Validation split '{args.val_split}' not found in dataset")

    # Preprocess datasets
    print("Preprocessing dataset...")
    train_dataset = preprocess_dataset(train_dataset, tokenizer, args.max_seq_length)

    if eval_dataset is not None:
        eval_dataset = preprocess_dataset(eval_dataset, tokenizer, args.max_seq_length)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps if eval_dataset else None,
        # Use same steps for save and eval when eval is enabled
        save_steps=args.eval_steps if eval_dataset else args.save_steps,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=args.report_to,
        run_name=args.run_name,
        # Evaluation and save strategy for best model loading
        eval_strategy="steps" if eval_dataset else "epoch",
        save_strategy="steps" if eval_dataset else "epoch",
        save_total_limit=3,  # Keep only last 3 checkpoints
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        logging_dir=f"{args.output_dir}/logs",
        push_to_hub=False,
    )

    # Configure W&B if enabled
    if args.report_to == "wandb":
        import os
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_entity:
            os.environ["WANDB_ENTITY"] = args.wandb_entity

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    # Save final model
    print("Saving final model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Training complete! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()