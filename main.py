"""
main.py – Entry point for the ASVspoof 2019 LA multi-pipeline experiment.

Project Structure
-----------------
config.py               – All hyperparameters and output paths
preprocessors.py        – 5 feature extractors (CWT, CQT, LFCC, LogMel, SSL)
preprocess_pipeline.py  – Dataset downloader + feature extraction runner
dataset_loader.py       – PyTorch Dataset for .npy features
swin_model.py           – Swin-Tiny classifier
trainer.py              – Training + evaluation loop
metrics.py              – EER, t-DCF, AUC-ROC
run_experiments.py      – Master orchestrator (preprocess -> train -> compare)
requirements.txt        – Python dependencies

Quick Start
-----------
1. Install dependencies:
       pip install -r requirements.txt

2. Run everything (all 5 pipelines):
       python run_experiments.py

3. Quick debug run (200 samples, 2 epochs):
       python run_experiments.py --max_samples 200 --epochs 2

4. Only preprocess (no training):
       python run_experiments.py --skip_train

5. Only train (preprocessing already done):
       python run_experiments.py --skip_preprocess

6. Run specific pipelines:
       python run_experiments.py --pipeline scalogram lfcc ssl_embeddings

Output Layout
-------------
outputs/
  pipeline_1_scalogram/
    features/{train,validation,test}/*.npy   <- per-sample feature arrays
    checkpoints/best_model.pth               <- best checkpoint (lowest EER)
    plots/scalogram_training_curves.png
    epoch_log.csv
  pipeline_2_linear_spec/  ...
  pipeline_3_lfcc/         ...
  pipeline_4_logmel/       ...
  pipeline_5_ssl_embeddings/ ...
  results/
    comparison.csv          <- test metrics for all pipelines
    report.md               <- Markdown comparison report
    results.json            <- raw JSON
    pipeline_comparison.png <- bar chart comparing all pipelines
"""

import sys
import os

# Ensure the project root is in sys.path
sys.path.insert(0, os.path.dirname(__file__))

from run_experiments import main

if __name__ == "__main__":
    main()
