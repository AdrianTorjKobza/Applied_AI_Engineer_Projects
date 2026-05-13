#!/usr/bin/env python3
"""
Load the fine-tuned DistilBERT model and classify new reviews.

Usage (Python):
    from scripts.inference import SentimentClassifier
    clf = SentimentClassifier("./checkpoints/final")
    clf.predict("Great product, love it!")
    # → {'label': 'positive', 'confidence': 0.97, 'latency_ms': 18.4}

Usage (CLI):
    python scripts/inference.py --model-path ./checkpoints/final \
        --reviews "Amazing!" "Garbage." "It is what it is."
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class SentimentClassifier:
    def __init__(self, model_path: str):
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run scripts/train.py first."
            )

        print(f"📦 Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        self.model.eval()

        # Load label map
        label_map_path = model_path / "label_map.json"

        if label_map_path.exists():
            with open(label_map_path) as f:
                maps = json.load(f)
            self.id2label = {int(k): v for k, v in maps["id2label"].items()}
        else:
            self.id2label = self.model.config.id2label

        print("✅ Model ready")

    @torch.no_grad()
    def predict(self, review: str, max_length: int = 128) -> dict:
        """Classify a single review string."""
        t0 = time.perf_counter()

        inputs = self.tokenizer(
            review,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        pred_id = int(probs.argmax())

        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "label":      self.id2label[pred_id],
            "confidence": round(float(probs[pred_id]), 4),
            "scores":     {self.id2label[i]: round(float(p), 4) for i, p in enumerate(probs)},
            "latency_ms": round(latency_ms, 1),
        }

    def predict_batch(self, reviews: list[str], show_progress: bool = True) -> list[dict]:
        from tqdm import tqdm
        iterator = tqdm(reviews, desc="Classifying") if show_progress else reviews
        return [self.predict(r) for r in iterator]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="./checkpoints/final")
    parser.add_argument("--reviews", nargs="+", required=True)
    args = parser.parse_args()

    clf = SentimentClassifier(args.model_path)
    print()
    for review in args.reviews:
        r = clf.predict(review)
        print(f"  Review    : {review[:80]}")
        print(f"  Label     : {r['label']}  (confidence: {r['confidence']:.0%})")
        print(f"  All scores: {r['scores']}")
        print(f"  Latency   : {r['latency_ms']} ms\n")


if __name__ == "__main__":
    main()