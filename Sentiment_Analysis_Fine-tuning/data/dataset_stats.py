"""
data/dataset_stats.py  —  Quick inspection of prepared splits.
Usage: python data/dataset_stats.py
"""
import json, random
from collections import Counter
from pathlib import Path

FILES = {"train": "data/train.jsonl", "val": "data/val.jsonl", "test": "data/test.jsonl"}

for split, path in FILES.items():
    if not Path(path).exists():
        print(f"⚠️  {path} not found — run prepare_dataset.py first"); continue

    rows = [json.loads(l) for l in open(path) if l.strip()]
    counts = Counter(r["label"] for r in rows)
    lengths = [len(r["review"].split()) for r in rows]

    print(f"\n── {split.upper()} ({len(rows)} examples) ──")

    for label in ["negative", "neutral", "positive"]:
        n = counts.get(label, 0)
        bar = "█" * (n // 5)
        print(f"  {label:10s}: {n:4d}  {bar}")
        
    print(f"  word len  : min={min(lengths)}  max={max(lengths)}  mean={sum(lengths)/len(lengths):.0f}")

    print("  samples   :")
    for ex in random.sample(rows, min(2, len(rows))):
        print(f"    [{ex['label']}] {ex['review'][:100]}...")
