#!/usr/bin/env python3
"""
Loads Amazon Polarity from local parquet files (data/amazon_polarity/),
generates neutral examples via heuristic scan of the same files,
and writes balanced train/val/test JSONL splits.

No network calls — fully offline.

Usage:
    python data/prepare_dataset.py
    python data/prepare_dataset.py --config configs/training_config.yaml

Expected local files (download from HuggingFace manually): https://huggingface.co/datasets/fancyzhx/amazon_polarity/tree/main/amazon_polarity
    data/amazon_polarity/train-00000-of-00004.parquet
    data/amazon_polarity/train-00001-of-00004.parquet
    data/amazon_polarity/train-00002-of-00004.parquet
    data/amazon_polarity/train-00003-of-00004.parquet
"""

import argparse
import json
import random
from pathlib import Path

import yaml
from datasets import load_dataset
from tqdm import tqdm

PARQUET_DIR = "data/amazon_polarity"

MIXED_PHRASES = [
    "okay", "decent", "average", "not bad", "could be better",
    "mixed", "so-so", "mediocre", "does the job", "nothing special",
    "it's fine", "its fine", "not great", "not terrible", "alright",
]


def load_local_dataset():
    """Load Amazon Polarity from local parquet files."""
    parquet_files = sorted(Path(PARQUET_DIR).glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found in {PARQUET_DIR}/\n"
            "Download them from: https://huggingface.co/datasets/fancyzhx/amazon_polarity/tree/main/amazon_polarity"
        )
    
    print(f"📂 Loading from {len(parquet_files)} local parquet file(s): {PARQUET_DIR}/")
    ds = load_dataset("parquet", data_files=[str(p) for p in parquet_files], split="train")
    print(f"  ✅ Loaded {len(ds):,} total rows")

    return ds


def load_positive_negative(ds, n_each: int) -> list[dict]:
    """Collect n_each positive and n_each negative examples from the dataset."""
    print(f"🔍 Sampling {n_each} positive + {n_each} negative examples...")

    label_map = {0: "negative", 1: "positive"}
    pos, neg = [], []

    for item in tqdm(ds, desc="Scanning pos/neg", total=len(ds)):
        text = f"{item['title']}. {item['content']}".strip()
        words = text.split()

        if len(words) < 5 or len(words) > 120:
            continue

        label = label_map[item["label"]]

        if label == "positive" and len(pos) < n_each:
            pos.append({"review": text, "label": "positive"})
        elif label == "negative" and len(neg) < n_each:
            neg.append({"review": text, "label": "negative"})

        if len(pos) >= n_each and len(neg) >= n_each:
            break

    print(f"  ✅ positive: {len(pos)}  negative: {len(neg)}")
    return pos + neg


def load_neutral_heuristic(ds, n: int) -> list[dict]:
    """Scan the local dataset for mixed-sentiment reviews and label them neutral."""
    print(f"🔍 Scanning for {n} neutral (mixed-sentiment) examples...")
    
    neutral = []

    for item in tqdm(ds, desc="Scanning neutral", total=len(ds)):
        text = f"{item['title']}. {item['content']}".strip()
        words = text.split()
        if len(words) < 10 or len(words) > 120:
            continue
        if any(p in text.lower() for p in MIXED_PHRASES):
            neutral.append({"review": text, "label": "neutral"})
        if len(neutral) >= n:
            break

    print(f"  ✅ neutral: {len(neutral)}")
    return neutral


def split(examples: list[dict], n_train: int, n_val: int, n_test: int, seed: int = 42):
    """Stratified split: equal class counts in every partition."""
    random.seed(seed)
    by_label: dict[str, list] = {}
    for ex in examples:
        by_label.setdefault(ex["label"], []).append(ex)

    train, val, test = [], [], []
    for items in by_label.values():
        random.shuffle(items)
        train += items[:n_train]
        val   += items[n_train : n_train + n_val]
        test  += items[n_train + n_val : n_train + n_val + n_test]

    random.shuffle(train)
    return train, val, test


def write_jsonl(examples: list[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"  💾 {path}  ({len(examples)} examples)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    spc = cfg["data"]["samples_per_class"]
    n_train, n_val, n_test = spc["train"], spc["val"], spc["test"]
    n_each = n_train + n_val + n_test + 20   # small buffer

    # Load local dataset once, reuse for both passes
    ds = load_local_dataset()
    binary  = load_positive_negative(ds, n_each)
    neutral = load_neutral_heuristic(ds, n_each)

    all_examples = binary + neutral

    # Stratified split
    train, val, test = split(all_examples, n_train, n_val, n_test,
                             seed=cfg["training"]["seed"])

    print("\n📝 Writing splits...")
    write_jsonl(train, cfg["data"]["train_file"])
    write_jsonl(val,   cfg["data"]["val_file"])
    write_jsonl(test,  cfg["data"]["test_file"])

    # Distribution summary
    print("\n📊 Class distribution:")
    for name, split_data in [("train", train), ("val", val), ("test", test)]:
        from collections import Counter
        counts = Counter(ex["label"] for ex in split_data)
        print(f"  {name:5s}: " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    print("\n✅ Dataset ready. Next: python scripts/train.py")


if __name__ == "__main__":
    main()