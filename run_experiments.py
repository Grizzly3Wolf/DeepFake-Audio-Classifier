"""
run_experiments.py – Orchestrator: preprocess -> train -> compare all pipelines.

Usage:
    # Run everything end-to-end (all 5 pipelines)
    python run_experiments.py

    # Only preprocess, skip training
    python run_experiments.py --skip_train

    # Only train (assumes preprocessing is done)
    python run_experiments.py --skip_preprocess

    # Run specific pipelines only
    python run_experiments.py --pipeline scalogram lfcc

    # Quick debug with limited samples
    python run_experiments.py --max_samples 200 --epochs 2
"""

import argparse
import csv
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as cfg
from preprocessors import EXTRACTORS


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all preprocessing + Swin Transformer training pipelines.")
    parser.add_argument(
        "--pipeline", nargs="+",
        choices=list(EXTRACTORS.keys()),
        default=list(EXTRACTORS.keys()),
        help="Which pipeline(s) to run (default: all 5).",
    )
    parser.add_argument("--skip_preprocess", action="store_true",
                        help="Skip preprocessing (features already extracted).")
    parser.add_argument("--skip_train",      action="store_true",
                        help="Skip training (only run preprocessing).")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit samples per split for quick debug.")
    parser.add_argument("--epochs", type=int, default=cfg.NUM_EPOCHS,
                        help=f"Number of training epochs (default {cfg.NUM_EPOCHS}).")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Comparison report
# ─────────────────────────────────────────────────────────────────────────────

METRIC_ORDER = ["accuracy", "auc", "eer", "min_tdcf"]
HIGHER_BETTER = {"accuracy": True, "auc": True, "eer": False, "min_tdcf": False}


def print_comparison_table(all_results: list):
    """Pretty-print results table to stdout."""
    print("\n" + "=" * 80)
    print("  FINAL COMPARISON TABLE")
    print("=" * 80)
    header = f"{'Pipeline':<20} {'Accuracy':>10} {'AUC-ROC':>10} {'EER':>10} {'min-tDCF':>10}"
    print(header)
    print("-" * 80)
    for r in sorted(all_results, key=lambda x: x["eer"]):
        print(f"{r['pipeline']:<20} "
              f"{r['accuracy']:>10.4f} "
              f"{r['auc']:>10.4f} "
              f"{r['eer']:>10.4f} "
              f"{r['min_tdcf']:>10.4f}")
    print("=" * 80 + "\n")


def save_comparison_csv(all_results: list, path: str):
    if not all_results:
        return
    fields = ["pipeline", "accuracy", "auc", "eer", "min_tdcf", "best_epoch"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    print(f"  Saved comparison CSV -> {path}")


def save_comparison_plots(all_results: list, plot_dir: str):
    if not all_results:
        return

    metrics   = ["accuracy", "auc", "eer", "min_tdcf"]
    titles    = ["Accuracy (higher=better)", "AUC-ROC (higher=better)",
                 "EER (lower=better)", "min-tDCF (lower=better)"]
    pipelines = [r["pipeline"] for r in all_results]
    x         = np.arange(len(pipelines))
    colors    = ["#4a90d9", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Pipeline Comparison – Swin Transformer on ASVspoof 2019 LA",
                 fontsize=14, fontweight="bold")

    for ax, metric, title in zip(axes.flat, metrics, titles):
        vals  = [r[metric] for r in all_results]
        bars  = ax.bar(x, vals, color=colors[:len(pipelines)], edgecolor="white",
                       linewidth=0.8, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(pipelines, rotation=20, ha="right", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(plot_dir, "pipeline_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved comparison plot -> {path}")


def save_markdown_report(all_results: list, path: str):
    lines = [
        "# Pipeline Comparison Report – ASVspoof 2019 LA\n",
        f"**Dataset**: `{cfg.DATASET_NAME}`  \n",
        f"**Model**: `{cfg.SWIN_MODEL}`  \n",
        f"**Epochs**: `{cfg.NUM_EPOCHS}`  \n",
        f"**Batch size**: `{cfg.BATCH_SIZE}`  \n\n",
        "## Results (sorted by EER)\n\n",
        "| Pipeline | Accuracy | AUC-ROC | EER | min-tDCF | Best Epoch |\n",
        "|---|---|---|---|---|---|\n",
    ]
    for r in sorted(all_results, key=lambda x: x["eer"]):
        lines.append(
            f"| {r['pipeline']} | {r['accuracy']:.4f} | {r['auc']:.4f} | "
            f"{r['eer']:.4f} | {r['min_tdcf']:.4f} | {r.get('best_epoch', '-')} |\n"
        )
    lines += [
        "\n## Metric Descriptions\n\n",
        "- **Accuracy**: Standard classification accuracy on test set\n",
        "- **AUC-ROC**: Area under the ROC curve\n",
        "- **EER**: Equal Error Rate (primary ASVspoof metric, lower is better)\n",
        "- **min-tDCF**: Minimum tandem Detection Cost Function (lower is better)\n",
        "\n## Pipeline Descriptions\n\n",
        "| Pipeline | Feature Type | Native Swin Support |\n",
        "|---|---|---|\n",
        "| scalogram | CWT 2-D image | Native 2D (Ideal) |\n",
        "| linear_spec | CQT 2-D image | Native 2D (Ideal) |\n",
        "| lfcc | Linear Cepstral 2D matrix | 2D Matrix |\n",
        "| logmel | Log-Mel + MFCC 2D matrix | 2D Matrix |\n",
        "| ssl_embeddings | Wav2Vec2 hidden states | Reshaped 2D |\n",
    ]
    with open(path, "w") as f:
        f.writelines(lines)
    print(f"  Saved Markdown report -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Override config with CLI args
    if args.max_samples:
        cfg.MAX_SAMPLES = args.max_samples
    cfg.NUM_EPOCHS = args.epochs

    selected_pipelines = args.pipeline
    print(f"\nSelected pipelines: {selected_pipelines}")
    print(f"Epochs: {cfg.NUM_EPOCHS}  |  Batch: {cfg.BATCH_SIZE}")
    print(f"Output root: {cfg.OUTPUT_ROOT}\n")

    # ── Step 1: Preprocessing ────────────────────────────────────────────────
    if not args.skip_preprocess:
        from preprocess_pipeline import run_pipeline
        from datasets import load_dataset

        print("=" * 60)
        print("  STEP 1: PREPROCESSING")
        print("=" * 60)
        dataset = load_dataset(cfg.DATASET_NAME)
        label_names = dataset["train"].features["key"].names

        for name in selected_pipelines:
            run_pipeline(name, dataset, label_names)
    else:
        print("  [SKIP] Preprocessing skipped.\n")

    # ── Step 2: Training ─────────────────────────────────────────────────────
    all_results = []

    if not args.skip_train:
        from trainer import train_pipeline

        print("=" * 60)
        print("  STEP 2: TRAINING")
        print("=" * 60)

        for name in selected_pipelines:
            t0 = time.time()
            try:
                metrics = train_pipeline(name)
                all_results.append(metrics)
            except Exception as exc:
                print(f"  [ERROR] Training {name} failed: {exc}")
            elapsed = time.time() - t0
            print(f"  {name} total time: {elapsed/60:.1f} min\n")

        # ── Step 3: Reporting ─────────────────────────────────────────────────
        if all_results:
            print_comparison_table(all_results)
            csv_path = os.path.join(cfg.RESULTS_DIR, "comparison.csv")
            md_path  = os.path.join(cfg.RESULTS_DIR, "report.md")
            json_path = os.path.join(cfg.RESULTS_DIR, "results.json")

            save_comparison_csv(all_results, csv_path)
            save_comparison_plots(all_results, cfg.RESULTS_DIR)
            save_markdown_report(all_results, md_path)

            with open(json_path, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"  Saved raw JSON -> {json_path}")
    else:
        print("  [SKIP] Training skipped.\n")

    print("\nAll done!")


if __name__ == "__main__":
    main()
