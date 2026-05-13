#!/usr/bin/env python3
"""
Score the fine-tuned DistilBERT on the held-out test set.

Usage:
    python evaluation/evaluate_finetuned.py
    python evaluation/evaluate_finetuned.py --model-path ./checkpoints/final
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.inference import SentimentClassifier

VALID_CLASSES = ["negative", "neutral", "positive"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="./checkpoints/final")
    parser.add_argument("--test-file",  default="data/test.jsonl")
    parser.add_argument("--output",     default="results/finetuned_results.json")
    args = parser.parse_args()

    if not Path(args.test_file).exists():
        print(f"❌ {args.test_file} not found. Run data/prepare_dataset.py first."); return

    rows = [json.loads(l) for l in open(args.test_file) if l.strip()]
    reviews     = [r["review"] for r in rows]
    true_labels = [r["label"]  for r in rows]

    clf = SentimentClassifier(args.model_path)

    print(f"\n🔍 Evaluating fine-tuned model on {len(rows)} examples...")
    t0 = time.perf_counter()
    predictions = clf.predict_batch(reviews)
    total_ms = (time.perf_counter() - t0) * 1000

    pred_labels = [p["label"] for p in predictions]
    latencies   = [p["latency_ms"] for p in predictions]
    avg_latency = sum(latencies) / len(latencies)

    accuracy = accuracy_score(true_labels, pred_labels)
    f1_macro = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    report   = classification_report(
        true_labels, pred_labels,
        labels=VALID_CLASSES, zero_division=0, output_dict=True,
    )
    cm = confusion_matrix(true_labels, pred_labels, labels=VALID_CLASSES).tolist()

    print(f"\n{'─'*50}")
    print(f"  Fine-tuned DistilBERT")
    print(f"  Accuracy   : {accuracy:.1%}")
    print(f"  F1 (Macro) : {f1_macro:.4f}")
    print(f"  Avg Latency: {avg_latency:.0f} ms/sample")
    print(f"{'─'*50}")
    print("  Per-class F1:")
    for cls in VALID_CLASSES:
        f1 = report.get(cls, {}).get("f1-score", 0.0)
        print(f"    {cls:10s}: {f1:.3f}")

    results = {
        "mode": "finetuned",
        "model": f"DistilBERT fine-tuned ({args.model_path})",
        "n_samples": len(rows),
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1_macro, 4),
        "avg_latency_ms": round(avg_latency, 1),
        "classification_report": report,
        "confusion_matrix": cm,
        "confusion_matrix_labels": VALID_CLASSES,
        "raw_predictions": [
            {"review": rev[:200], "true": true, "pred": pred["label"],
             "confidence": pred["confidence"], "latency_ms": pred["latency_ms"]}
            for rev, true, pred in zip(reviews, true_labels, predictions)
        ],
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Saved → {args.output}")
    print(f"   Next: python evaluation/compare_results.py")


if __name__ == "__main__":
    main()
