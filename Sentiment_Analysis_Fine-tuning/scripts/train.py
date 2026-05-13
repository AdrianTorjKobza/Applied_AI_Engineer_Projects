#!/usr/bin/env python3
"""
Fine-tune DistilBERT for 3-class sentiment classification.
Runs on CPU in ~15–20 minutes. No GPU or special tokens required.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/training_config.yaml
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)


def load_jsonl(path: str) -> Dataset:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    labels: list[str] = cfg["model"]["labels"]          # ["negative", "neutral", "positive"]
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for i, l in enumerate(labels)}

    # ── Tokenizer ────────────────────────────────────────────────────────────
    model_name = cfg["model"]["name"]
    print(f"📦 Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        enc = tokenizer(
            batch["review"],
            truncation=True,
            padding="max_length",
            max_length=cfg["data"]["max_length"],
        )
        enc["labels"] = [label2id[l] for l in batch["label"]]
        return enc

    # ── Datasets ─────────────────────────────────────────────────────────────
    print("📂 Loading datasets...")
    train_ds = load_jsonl(cfg["data"]["train_file"]).map(tokenize, batched=True)
    val_ds   = load_jsonl(cfg["data"]["val_file"]).map(tokenize, batched=True)

    # Keep only the columns the model needs
    cols = ["input_ids", "attention_mask", "labels"]
    train_ds = train_ds.select_columns(cols)
    val_ds   = val_ds.select_columns(cols)
    train_ds.set_format("torch")
    val_ds.set_format("torch")

    print(f"  Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"🧠 Loading model: {model_name}")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=cfg["model"]["num_labels"],
        id2label=id2label,
        label2id=label2id,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}  ({total_params/1e6:.1f} M)")

    # ── Metrics ───────────────────────────────────────────────────────────────
    def compute_metrics(eval_pred):
        logits, label_ids = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = float((preds == label_ids).mean())
        f1  = f1_score(label_ids, preds, average="macro", zero_division=0)
        return {"accuracy": acc, "f1": f1}

    # ── Training args ─────────────────────────────────────────────────────────
    t = cfg["training"]
    output_dir = t["output_dir"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        learning_rate=t["learning_rate"],
        weight_decay=t["weight_decay"],
        warmup_ratio=t["warmup_ratio"],
        lr_scheduler_type=t["lr_scheduler_type"],
        eval_strategy=t["eval_strategy"],
        save_strategy=t["save_strategy"],
        load_best_model_at_end=t["load_best_model_at_end"],
        metric_for_best_model=t["metric_for_best_model"],
        logging_steps=t["logging_steps"],
        seed=t["seed"],
        report_to="none",           # disable W&B/wandb — not needed on laptop
        use_cpu=True,               # force CPU — remove this line if you have a GPU
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n🏋️  Training started (this takes ~15–20 min on CPU)...\n")
    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    final_path = Path(output_dir) / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    # Save label map alongside the model
    with open(final_path / "label_map.json", "w") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f, indent=2)

    print(f"\n✅ Training complete!")
    print(f"   Model saved → {final_path}")
    print(f"   Next: python evaluation/evaluate_finetuned.py")


if __name__ == "__main__":
    main()