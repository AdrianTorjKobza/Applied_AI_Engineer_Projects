#!/usr/bin/env python3
"""
Load baseline + fine-tuned results, print a delta table, and save plots.

Usage:
    python evaluation/compare_results.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

RESULT_FILES = {
    "Heuristic Baseline": "results/baseline_results.json",
    "Fine-tuned DistilBERT": "results/finetuned_results.json",
}

COLORS = {"Heuristic Baseline": "#6B7280", "Fine-tuned DistilBERT": "#10B981"}
VALID_CLASSES = ["negative", "neutral", "positive"]


def load(path):
    p = Path(path)
    
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def print_table(results):
    print("\n" + "=" * 62)
    print("  📊 EVALUATION RESULTS — Quality Delta Report")
    print("=" * 62)
    print(f"\n  {'Model':<26} {'Accuracy':>10} {'F1 Macro':>10} {'Latency':>10}")
    print("  " + "─" * 58)
    
    for name, res in results.items():
        print(f"  {name:<26} {res['accuracy']:>9.1%} {res['f1_macro']:>10.4f} {res['avg_latency_ms']:>8.1f}ms")

    if len(results) == 2:
        vals = list(results.values())
        acc_delta = vals[1]["accuracy"] - vals[0]["accuracy"]
        f1_delta  = vals[1]["f1_macro"]  - vals[0]["f1_macro"]
        print(f"\n  🎯 Delta (Fine-tuned vs Baseline):")
        print(f"     Accuracy : {acc_delta:+.1%}")
        print(f"     F1 Macro : {f1_delta:+.4f}")

    print("\n  Per-class F1:")
    print(f"  {'Label':<12}" + "".join(f"{n[:16]:<18}" for n in results))
    
    for cls in VALID_CLASSES:
        row = f"  {cls:<12}"
        for res in results.values():
            f1 = res.get("classification_report", {}).get(cls, {}).get("f1-score", 0.0)
            row += f"{f1:<18.3f}"
        print(row)
    print("=" * 62)


def plot_bars(results, out_dir):
    names = list(results.keys())
    colors = [COLORS.get(n, "#3B82F6") for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor("#0F172A")

    for ax, (metric, ylabel) in zip(axes, [("accuracy", "Accuracy"), ("f1_macro", "F1 Macro")]):
        vals = [results[n][metric] for n in names]
        ax.set_facecolor("#1E293B")
        bars = ax.bar(names, vals, color=colors, width=0.45, edgecolor="#334155")
        ax.set_title(ylabel, color="white", fontsize=13, fontweight="bold", pad=10)
        ax.set_ylim(0, 1.1)
        ax.tick_params(colors="white", labelsize=9)
        ax.spines[:].set_color("#334155")
        ax.yaxis.grid(True, color="#334155", alpha=0.4, linestyle="--")
        ax.set_axisbelow(True)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{v:.1%}" if metric == "accuracy" else f"{v:.3f}",
                    ha="center", color="white", fontsize=11, fontweight="bold")

    fig.suptitle("Baseline vs Fine-tuned DistilBERT — Sentiment", color="white",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "accuracy_f1.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  📈 {path}")


def plot_confusion(results, out_dir):
    ft = results.get("Fine-tuned DistilBERT")
    if not ft or "confusion_matrix" not in ft:
        return

    cm = np.array(ft["confusion_matrix"], dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums

    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#1E293B")
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=VALID_CLASSES, yticklabels=VALID_CLASSES,
                ax=ax, linewidths=0.5, linecolor="#334155", cbar=False)
    ax.set_title("Confusion Matrix (Fine-tuned, row-norm)", color="white",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted", color="#94A3B8")
    ax.set_ylabel("True", color="#94A3B8")
    ax.tick_params(colors="white", labelsize=9)
    plt.tight_layout()
    path = out_dir / "confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  🔲 {path}")


def main():
    print("\n📂 Loading results...")
    results = {}
    for name, path in RESULT_FILES.items():
        r = load(path)
        if r:
            results[name] = r
            print(f"  ✅ {name}")
        else:
            print(f"  ⚠️  {name} — {path} not found")

    if not results:
        print("\n❌ No results found. Run the eval scripts first."); return

    print_table(results)

    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    print("\n📊 Saving plots...")
    plot_bars(results, fig_dir)
    plot_confusion(results, fig_dir)

    # Save summary JSON
    summary = {
        name: {k: res[k] for k in ("accuracy", "f1_macro", "avg_latency_ms")}
        for name, res in results.items()
    }
    
    if len(results) == 2:
        vals = list(results.values())
        summary["delta"] = {
            "accuracy": round(vals[1]["accuracy"] - vals[0]["accuracy"], 4),
            "f1_macro":  round(vals[1]["f1_macro"]  - vals[0]["f1_macro"],  4),
        }

    with open("results/eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("  💾 results/eval_summary.json")
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
