"""
config.py – Central configuration for all 5 preprocessing pipelines
and Swin Transformer training experiments.
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────
DATASET_NAME   = "Bisher/ASVspoof_2019_LA"
SAMPLE_RATE    = 16_000          # Hz
CLIP_DURATION  = 4.0             # seconds – pad / truncate all clips to this
MAX_SAMPLES    = 300             # Samples per split – keeps a full 5-pipeline run within ~5-10 min on RTX 4050

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT ROOT
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "outputs")

PIPELINE_DIRS = {
    "scalogram":      os.path.join(OUTPUT_ROOT, "pipeline_1_scalogram"),
    "linear_spec":    os.path.join(OUTPUT_ROOT, "pipeline_2_linear_spec"),
    "lfcc":           os.path.join(OUTPUT_ROOT, "pipeline_3_lfcc"),
    "logmel":         os.path.join(OUTPUT_ROOT, "pipeline_4_logmel"),
    "ssl_embeddings": os.path.join(OUTPUT_ROOT, "pipeline_5_ssl_embeddings"),
}

# Sub-folders inside each pipeline dir
for _pdir in PIPELINE_DIRS.values():
    for _split in ("train", "validation", "test"):
        os.makedirs(os.path.join(_pdir, "features", _split), exist_ok=True)
    os.makedirs(os.path.join(_pdir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(_pdir, "plots"),       exist_ok=True)

RESULTS_DIR = os.path.join(OUTPUT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SHAPES expected by Swin (H x W, single channel -> 3-channel repeat)
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE = 224    # Swin-Tiny expects 224x224

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING HYPER-PARAMS
# ─────────────────────────────────────────────────────────────────────────────
N_FFT         = 512
HOP_LENGTH    = 128
N_MELS        = 128
N_MFCC        = 40
N_LFCC        = 40
CWT_SCALES    = 128   # number of CWT scales (rows of the scalogram)

# CQT
N_BINS        = 84    # CQT bins covering ~7 octaves
BINS_PER_OCT  = 12

# SSL
SSL_MODEL_NAME = "facebook/wav2vec2-base-960h"   # or "facebook/wav2vec2-large-xlsr-53"

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING HYPER-PARAMS
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE    = 16    # 16 is safer for 6 GB VRAM (especially wav2vec2 SSL pipeline)
NUM_EPOCHS    = 3     # 3 epochs across 5 pipelines fits in ~5-10 min on laptop GPU
LR            = 3e-4
WEIGHT_DECAY  = 1e-4
NUM_WORKERS   = 2     # 2 workers reduces thermal load on laptop
DEVICE        = "cuda"   # falls back to "cpu" at runtime if CUDA not available
SEED          = 42
SWIN_MODEL    = "swin_tiny_patch4_window7_224"   # timm model name
NUM_CLASSES   = 2        # bonafide vs spoof

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
# t-DCF costs (ASVspoof 2019 defaults)
C_MISS   = 1
C_FA     = 10
P_TARGET = 0.05
