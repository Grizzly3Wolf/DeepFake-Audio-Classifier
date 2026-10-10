"""
preprocess_pipeline.py – Download dataset and run all 5 feature extraction
pipelines, saving .npy files for each split.

Usage:
    python preprocess_pipeline.py                    # all pipelines
    python preprocess_pipeline.py --pipeline lfcc    # single pipeline
    python preprocess_pipeline.py --pipeline scalogram linear_spec
    python preprocess_pipeline.py --max_samples 500  # quick debug run
"""

import argparse
import io
import os
import time

import numpy as np
import soundfile as sf
from datasets import load_dataset, Audio
from tqdm import tqdm

import config as cfg
from preprocessors import EXTRACTORS


# ─────────────────────────────────────────────────────────────────────────────
# Audio loading helper
# ─────────────────────────────────────────────────────────────────────────────

def load_audio(sample: dict) -> np.ndarray:
    """
    Decode one audio sample from the HuggingFace dataset.
    Returns float32 mono waveform at cfg.SAMPLE_RATE.
    """
    audio = sample["audio"]
    wav   = audio["array"].astype(np.float32)
    # The dataset is already resampled to SAMPLE_RATE via cast_column
    return wav


# ─────────────────────────────────────────────────────────────────────────────
# Per-pipeline runner
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(pipeline_name: str, dataset: dict, label_names: list):
    """
    Extract features for all splits using a single pipeline and save .npy files.
    
    File layout:
        outputs/pipeline_X_<name>/features/<split>/<file_stem>.npy
        outputs/pipeline_X_<name>/features/<split>/labels.npy
    """
    extractor = EXTRACTORS[pipeline_name]
    out_dir   = cfg.PIPELINE_DIRS[pipeline_name]

    print(f"\n{'='*60}")
    print(f"  PIPELINE: {pipeline_name.upper()}")
    print(f"  Output  : {out_dir}")
    print(f"{'='*60}")

    for split in ("train", "validation", "test"):
        feat_dir = os.path.join(out_dir, "features", split)
        label_path = os.path.join(feat_dir, "labels.npy")

        # Skip if already done
        if os.path.isfile(label_path):
            print(f"  [{split}] Already exists – skipping.")
            continue

        # Shuffle dataset before casting and subsetting to ensure both classes are represented
        ds_split = dataset[split].shuffle(seed=cfg.SEED).cast_column("audio", Audio(sampling_rate=cfg.SAMPLE_RATE))
        n = cfg.MAX_SAMPLES or len(ds_split)
        n = min(n, len(ds_split))

        labels   = []
        file_ids = []
        errors   = 0

        print(f"\n  [{split}] Extracting {n} samples …")
        t0 = time.time()

        for i in tqdm(range(n), desc=f"{pipeline_name}/{split}", ncols=80):
            sample = ds_split[i]
            label  = sample["key"]          # integer label (bonafide / spoof)
            fname  = sample.get("audio_file_name", str(i))
            stem   = os.path.splitext(os.path.basename(fname))[0]

            save_path = os.path.join(feat_dir, f"{stem}.npy")

            if os.path.isfile(save_path):
                labels.append(label)
                file_ids.append(stem)
                continue

            try:
                wav     = load_audio(sample)
                feature = extractor(wav)    # (3, IMG_SIZE, IMG_SIZE) float32
                np.save(save_path, feature)
                labels.append(label)
                file_ids.append(stem)
            except Exception as exc:
                errors += 1
                tqdm.write(f"  [WARN] {stem}: {exc}")

        elapsed = time.time() - t0
        np.save(label_path, np.array(labels, dtype=np.int64))
        np.save(os.path.join(feat_dir, "file_ids.npy"), np.array(file_ids, dtype=str))

        print(f"  [{split}] Done: {len(labels)} samples, {errors} errors, "
              f"{elapsed:.1f}s  ({len(labels)/elapsed:.1f} samples/s)")

    print(f"\n  {pipeline_name} preprocessing complete.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess ASVspoof 2019 LA with multiple pipelines.")
    parser.add_argument(
        "--pipeline", nargs="+",
        choices=list(EXTRACTORS.keys()),
        default=list(EXTRACTORS.keys()),
        help="Which pipeline(s) to run (default: all).",
    )
    parser.add_argument(
        "--max_samples", type=int, default=cfg.MAX_SAMPLES,
        help="Limit samples per split (useful for quick debug).",
    )
    args = parser.parse_args()

    if args.max_samples:
        cfg.MAX_SAMPLES = args.max_samples

    print("Loading dataset from HuggingFace Hub …")
    dataset = load_dataset(cfg.DATASET_NAME)
    label_names = dataset["train"].features["key"].names
    print(f"Label names: {label_names}")
    print(f"Splits: { {s: len(dataset[s]) for s in dataset} }")

    for pipeline_name in args.pipeline:
        run_pipeline(pipeline_name, dataset, label_names)

    print("\nAll requested pipelines finished!")
    print(f"Features saved under: {cfg.OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
