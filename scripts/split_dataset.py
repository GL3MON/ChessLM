#!/usr/bin/env python
"""
Split ChessInstruct dataset for SFT training.
Creates train/val/test splits for SFT phase.

TL;DR: Split 99,000 samples into:
  - SFT train: 63,360 (64%) - for supervised fine-tuning
  - SFT val:   5,940  (6%)  - for validation during training
  - RL:        19,800 (20%) - held out for RL phase
  - Test:      9,900  (10%) - held out for final evaluation

Usage:
    python split_dataset.py --dataset_path Thytu/ChessInstruct --with-validation --output_dir ./splits
"""

import argparse
import logging
import time
from datetime import timedelta
from pathlib import Path

from datasets import load_dataset, DatasetDict
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def split_dataset(
    dataset_path: str,
    sft_train_ratio: float = 0.70,
    rl_ratio: float = 0.20,
    test_ratio: float = 0.10,
    seed: int = 42,
    output_dir: str = "./splits",
):
    """Split the dataset into SFT train, RL, and Test sets."""
    start_time = time.time()

    logger.info(f"Loading dataset from {dataset_path}...")
    dataset = load_dataset(dataset_path)

    # Handle different dataset structures
    if "train" in dataset:
        full_dataset = dataset["train"]
    else:
        full_dataset = dataset

    total_samples = len(full_dataset)
    logger.info(f"Total samples: {total_samples:,}")

    # Calculate split sizes
    sft_size = int(total_samples * sft_train_ratio)
    rl_size = int(total_samples * rl_ratio)
    test_size = int(total_samples * test_ratio)

    # Adjust for rounding
    remaining = total_samples - (sft_size + rl_size + test_size)
    sft_size += remaining

    logger.info(f"\n{'=' * 50}")
    logger.info("Split sizes:")
    logger.info(f"  SFT train: {sft_size:,} ({sft_size/total_samples*100:.1f}%)")
    logger.info(f"  RL:        {rl_size:,} ({rl_size/total_samples*100:.1f}%)")
    logger.info(f"  Test:      {test_size:,} ({test_size/total_samples*100:.1f}%)")
    logger.info(f"{'=' * 50}")

    # Split the dataset with progress bar
    logger.info("Shuffling and splitting dataset...")
    shuffled = full_dataset.shuffle(seed=seed)

    sft_train = shuffled.select(range(sft_size))
    remaining_after_sft = shuffled.select(range(sft_size, total_samples))
    rl = remaining_after_sft.select(range(rl_size))
    test = remaining_after_sft.select(range(rl_size, rl_size + test_size))

    # Create output structure
    splits = DatasetDict({
        "sft_train": sft_train,
        "rl": rl,
        "test": test,
    })

    # Save splits with progress bar
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving splits to {output_dir}/...")
    for split_name, split_data in tqdm(splits.items(), desc="Saving splits"):
        split_output_dir = output_path / split_name
        split_data.save_to_disk(split_output_dir)
        logger.info(f"  Saved {split_name}: {len(split_data):,} samples")

    elapsed = time.time() - start_time
    logger.info(f"\n{'=' * 50}")
    logger.info(f"Done! Splits saved to {output_dir}/")
    logger.info(f"Total time: {timedelta(seconds=elapsed)}")
    logger.info(f"{'=' * 50}")

    return str(output_path)


def split_for_sft_with_validation(
    dataset_path: str,
    sft_train_ratio: float = 0.64,
    sft_val_ratio: float = 0.06,
    rl_ratio: float = 0.20,
    test_ratio: float = 0.10,
    seed: int = 42,
    output_dir: str = "./splits",
):
    """Split dataset with validation set from SFT portion."""
    start_time = time.time()

    logger.info(f"Loading dataset from {dataset_path}...")
    dataset = load_dataset(dataset_path)

    if "train" in dataset:
        full_dataset = dataset["train"]
    else:
        full_dataset = dataset

    total_samples = len(full_dataset)
    logger.info(f"Total samples: {total_samples:,}")

    # Calculate split sizes
    sft_train_size = int(total_samples * sft_train_ratio)
    sft_val_size = int(total_samples * sft_val_ratio)
    rl_size = int(total_samples * rl_ratio)
    test_size = int(total_samples * test_ratio)

    # Adjust for rounding
    remaining = total_samples - (sft_train_size + sft_val_size + rl_size + test_size)
    sft_train_size += remaining

    logger.info(f"\n{'=' * 50}")
    logger.info("Split sizes:")
    logger.info(f"  SFT train:  {sft_train_size:,} ({sft_train_size/total_samples*100:.1f}%)")
    logger.info(f"  SFT val:    {sft_val_size:,} ({sft_val_size/total_samples*100:.1f}%)")
    logger.info(f"  RL:         {rl_size:,} ({rl_size/total_samples*100:.1f}%)")
    logger.info(f"  Test:       {test_size:,} ({test_size/total_samples*100:.1f}%)")
    logger.info(f"{'=' * 50}")

    # Split with progress bar
    logger.info("Shuffling and splitting dataset...")
    shuffled = full_dataset.shuffle(seed=seed)

    sft_train = shuffled.select(range(sft_train_size))
    remaining_after_sft_train = shuffled.select(range(sft_train_size, total_samples))

    # remaining_after_sft_train has size = total_samples - sft_train_size
    # We need to split it into: sft_val + rl + test
    sft_val = remaining_after_sft_train.select(range(sft_val_size))

    # rl starts after sft_val within remaining_after_sft_train
    rl_start = sft_val_size
    rl_end = sft_val_size + rl_size
    rl = remaining_after_sft_train.select(range(rl_start, rl_end))

    # test comes after rl
    test_start = rl_end
    test_end = rl_end + test_size
    test = remaining_after_sft_train.select(range(test_start, test_end))

    splits = DatasetDict({
        "sft_train": sft_train,
        "sft_val": sft_val,
        "rl": rl,
        "test": test,
    })

    # Save splits with progress bar
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving splits to {output_dir}/...")
    for split_name, split_data in tqdm(splits.items(), desc="Saving splits"):
        split_output_dir = output_path / split_name
        split_data.save_to_disk(split_output_dir)
        logger.info(f"  Saved {split_name}: {len(split_data):,} samples")

    elapsed = time.time() - start_time
    logger.info(f"\n{'=' * 50}")
    logger.info(f"Done! Splits saved to {output_dir}/")
    logger.info(f"Total time: {timedelta(seconds=elapsed)}")
    logger.info(f"{'=' * 50}")

    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split ChessInstruct dataset for SFT training"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="Thytu/ChessInstruct",
        help="Path to dataset or Hugging Face dataset name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./splits",
        help="Directory to save splits",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling",
    )
    parser.add_argument(
        "--with-validation",
        action="store_true",
        help="Create validation split from SFT data",
    )

    args = parser.parse_args()

    if args.with_validation:
        split_for_sft_with_validation(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            seed=args.seed,
        )
    else:
        split_dataset(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            seed=args.seed,
        )