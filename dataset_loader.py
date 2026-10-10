"""
dataset_loader.py – PyTorch Dataset that loads pre-extracted .npy features.
Used by the Swin Transformer trainer.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import config as cfg


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation (train only)
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
])

EVAL_TRANSFORM = None   # no augmentation for val / test


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class NPYFeatureDataset(Dataset):
    """
    Loads pre-extracted (3, H, W) .npy feature tensors from disk.

    Directory structure expected:
        <feat_dir>/
            <stem>.npy          # feature array (3, H, W) float32
            labels.npy          # int64 label per sample
            file_ids.npy        # string stems in order
    """

    def __init__(self, feat_dir: str, split: str, augment: bool = False):
        self.feat_dir = feat_dir
        self.augment  = augment

        # Load labels and file IDs
        self.labels   = np.load(os.path.join(feat_dir, "labels.npy"))
        self.file_ids = np.load(os.path.join(feat_dir, "file_ids.npy"),
                                allow_pickle=True)

        # Filter to only files that actually exist on disk
        valid_mask = np.array([
            os.path.isfile(os.path.join(feat_dir, f"{fid}.npy"))
            for fid in self.file_ids
        ])
        self.labels   = self.labels[valid_mask]
        self.file_ids = self.file_ids[valid_mask]

        if len(self.labels) == 0:
            raise RuntimeError(
                f"No valid .npy feature files found in {feat_dir}. "
                "Run preprocess_pipeline.py first."
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        fid  = self.file_ids[idx]
        path = os.path.join(self.feat_dir, f"{fid}.npy")
        feat = np.load(path).astype(np.float32)   # (3, H, W)
        feat = torch.from_numpy(feat)

        if self.augment:
            feat = TRAIN_TRANSFORM(feat)

        label = int(self.labels[idx])
        return feat, label


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_datasets(pipeline_name: str):
    """Return (train_ds, val_ds, test_ds) for the given pipeline."""
    base = cfg.PIPELINE_DIRS[pipeline_name]

    train_ds = NPYFeatureDataset(
        os.path.join(base, "features", "train"),
        split="train",
        augment=True,
    )
    val_ds = NPYFeatureDataset(
        os.path.join(base, "features", "validation"),
        split="validation",
        augment=False,
    )
    test_ds = NPYFeatureDataset(
        os.path.join(base, "features", "test"),
        split="test",
        augment=False,
    )

    return train_ds, val_ds, test_ds
