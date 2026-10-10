"""
metrics.py – Evaluation metrics for ASVspoof 2019 LA.

Computes:
  - Accuracy
  - AUC-ROC
  - Equal Error Rate (EER)
  - minimum t-DCF (tandem Detection Cost Function)
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

import config as cfg


# ─────────────────────────────────────────────────────────────────────────────
# EER
# ─────────────────────────────────────────────────────────────────────────────

def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute Equal Error Rate.

    Parameters
    ----------
    labels : int array, 1 = bonafide (target), 0 = spoof (non-target)
    scores : float array, higher = more likely bonafide

    Returns
    -------
    eer : float  (0–1 scale)
    """
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr

    # Find the threshold where FPR ≈ FNR
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer)


# ─────────────────────────────────────────────────────────────────────────────
# t-DCF
# ─────────────────────────────────────────────────────────────────────────────

def compute_min_tdcf(labels: np.ndarray, scores: np.ndarray,
                     c_miss: float = cfg.C_MISS,
                     c_fa:   float = cfg.C_FA,
                     p_tar:  float = cfg.P_TARGET) -> float:
    """
    Compute the minimum tandem Detection Cost Function (t-DCF).

    Parameters
    ----------
    labels : int array, 1 = bonafide, 0 = spoof
    scores : float array, higher = more likely bonafide
    c_miss : cost of a miss (bonafide rejected)
    c_fa   : cost of a false alarm (spoof accepted)
    p_tar  : prior probability of target (bonafide)

    Returns
    -------
    min_tdcf : float
    """
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr

    # Normalise to detection cost
    dcf = c_miss * fnr * p_tar + c_fa * fpr * (1.0 - p_tar)

    # Normalise by the cost of a trivial system
    c_def = min(c_miss * p_tar, c_fa * (1 - p_tar))
    min_tdcf = dcf.min() / (c_def + 1e-12)
    return float(min_tdcf)


# ─────────────────────────────────────────────────────────────────────────────
# Aggregator
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(labels: np.ndarray, scores: np.ndarray) -> dict:
    """
    Run all metrics and return a dict.

    Parameters
    ----------
    labels : int array  (1 = bonafide, 0 = spoof)
    scores : float array (bonafide likelihood, e.g. softmax[:,1])
    """
    preds = (scores >= 0.5).astype(int)
    acc   = float((preds == labels).mean())
    auc   = float(roc_auc_score(labels, scores))
    eer   = compute_eer(labels, scores)
    tdcf  = compute_min_tdcf(labels, scores)

    return {
        "accuracy":  acc,
        "auc":       auc,
        "eer":       eer,
        "min_tdcf":  tdcf,
    }
