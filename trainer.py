"""
trainer.py – Swin Transformer training loop for one pipeline.

Features:
  - Mixed precision (torch.cuda.amp)
  - CosineAnnealingLR scheduler
  - Best-model checkpoint saved per pipeline
  - Per-epoch metrics logged to CSV
  - Training curves saved as PNG plots
"""

import csv
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg
from dataset_loader import get_datasets
from metrics import evaluate
from swin_model import build_model


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _set_seed(seed: int = cfg.SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_curves(history: dict, plot_dir: str, pipeline_name: str):
    """Save loss and accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Pipeline: {pipeline_name}", fontsize=14, fontweight="bold")

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#4a90d9")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   color="#e74c3c")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["val_acc"], label="Val Accuracy", color="#2ecc71")
    axes[1].set_title("Validation Accuracy"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1); axes[1].legend(); axes[1].grid(alpha=0.3)

    # EER
    axes[2].plot(epochs, history["val_eer"], label="Val EER", color="#f39c12")
    axes[2].set_title("Validation EER (lower is better)"); axes[2].set_xlabel("Epoch")
    axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plot_dir, f"{pipeline_name}_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved curves -> {path}")


def _log_csv(csv_path: str, row: dict, write_header: bool = False):
    mode = "w" if write_header else "a"
    with open(csv_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# Core training function
# ─────────────────────────────────────────────────────────────────────────────

def train_pipeline(pipeline_name: str) -> dict:
    """
    Train Swin-Tiny on pre-extracted features for one pipeline.

    Returns a dict of final test metrics.
    """
    _set_seed()
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  TRAINING: {pipeline_name.upper()}  |  device={device}")
    print(f"{'='*60}")

    # ── Dirs ─────────────────────────────────────────────────────────────────
    pipeline_dir  = cfg.PIPELINE_DIRS[pipeline_name]
    ckpt_dir      = os.path.join(pipeline_dir, "checkpoints")
    plot_dir      = os.path.join(pipeline_dir, "plots")
    best_ckpt     = os.path.join(ckpt_dir, "best_model.pth")
    log_csv_path  = os.path.join(pipeline_dir, "epoch_log.csv")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_ds, val_ds, test_ds = get_datasets(pipeline_name)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                              shuffle=True,  num_workers=cfg.NUM_WORKERS,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE,
                              shuffle=False, num_workers=cfg.NUM_WORKERS,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE,
                              shuffle=False, num_workers=cfg.NUM_WORKERS,
                              pin_memory=True)

    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = build_model(pretrained=True).to(device)
    
    # Calculate class weights from training labels to handle extreme class imbalance
    train_labels = train_ds.labels
    class_counts = np.bincount(train_labels)
    # Inverse frequency weighting
    class_weights = len(train_labels) / (len(class_counts) * class_counts.astype(np.float32))
    weight_tensor = torch.FloatTensor(class_weights).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.NUM_EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == "cuda")

    # ── History ───────────────────────────────────────────────────────────────
    history = {k: [] for k in ["train_loss", "val_loss", "val_acc", "val_eer"]}
    best_eer = float("inf")
    best_epoch = 0

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        t_ep = time.time()

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for feats, labels in tqdm(train_loader,
                                  desc=f"Ep{epoch:02d} train",
                                  leave=False, ncols=80):
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
                logits = model(feats)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * feats.size(0)
        train_loss /= len(train_ds)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss   = 0.0
        all_labels = []
        all_scores = []
        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(device), labels.to(device)
                with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
                    logits = model(feats)
                    loss   = criterion(logits, labels)
                val_loss += loss.item() * feats.size(0)
                probs = torch.softmax(logits, dim=1)[:, 1]
                all_scores.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_loss /= len(val_ds)

        val_metrics = evaluate(
            np.array(all_labels), np.array(all_scores))

        scheduler.step()

        # Save history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_eer"].append(val_metrics["eer"])

        # Log to CSV
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               **{f"val_{k}": v for k, v in val_metrics.items()},
               "lr": scheduler.get_last_lr()[0]}
        _log_csv(log_csv_path, row, write_header=(epoch == 1))

        # Checkpoint
        if val_metrics["eer"] < best_eer:
            best_eer   = val_metrics["eer"]
            best_epoch = epoch
            torch.save({
                "epoch":       epoch,
                "state_dict":  model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "best_eer":    best_eer,
                "pipeline":    pipeline_name,
            }, best_ckpt)

        elapsed = time.time() - t_ep
        print(
            f"  Ep {epoch:02d}/{cfg.NUM_EPOCHS}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"acc={val_metrics['accuracy']:.4f}  eer={val_metrics['eer']:.4f}  "
            f"auc={val_metrics['auc']:.4f}  [{elapsed:.1f}s]"
        )

    print(f"\n  Best EER {best_eer:.4f} at epoch {best_epoch}")

    # ── Test ──────────────────────────────────────────────────────────────────
    print("  Loading best checkpoint for test evaluation …")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    all_labels = []
    all_scores = []
    with torch.no_grad():
        for feats, labels in test_loader:
            feats = feats.to(device)
            with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
                logits = model(feats)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_scores.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    test_metrics = evaluate(
        np.array(all_labels), np.array(all_scores))
    test_metrics["pipeline"]    = pipeline_name
    test_metrics["best_epoch"]  = best_epoch

    print(f"  TEST: acc={test_metrics['accuracy']:.4f}  "
          f"eer={test_metrics['eer']:.4f}  "
          f"auc={test_metrics['auc']:.4f}  "
          f"tdcf={test_metrics['min_tdcf']:.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    _save_curves(history, plot_dir, pipeline_name)

    return test_metrics
