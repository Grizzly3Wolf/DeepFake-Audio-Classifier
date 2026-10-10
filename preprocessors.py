"""
preprocessors.py – 5 feature extractors for ASVspoof 2019 LA

Each extractor receives a raw waveform (numpy float32, 16 kHz) and returns
a (H, W) float32 numpy array that will be resized to IMG_SIZE x IMG_SIZE
and saved as a .npy file.

Pipelines:
  1. Scalogram         – CWT (pywt) -> 2-D image
  2. Linear Spectrogram / CQT – librosa -> 2-D image
  3. Log-Mel Spectrogram – librosa mel filterbank -> 2-D matrix
  4. MFCC              – librosa MFCC (+ deltas) -> 2-D matrix
  5. SSL Embeddings    – Wav2Vec 2.0 hidden states -> reshaped 2-D
"""

import numpy as np
import librosa
import cv2

import config as cfg


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pad_or_trim(wav: np.ndarray, sr: int = cfg.SAMPLE_RATE,
                 duration: float = cfg.CLIP_DURATION) -> np.ndarray:
    """Pad with zeros or trim to exactly `duration` seconds."""
    target = int(sr * duration)
    if len(wav) >= target:
        return wav[:target]
    return np.pad(wav, (0, target - len(wav)), mode="constant")


def _normalize_2d(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize a 2-D array to [0, 1]."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _resize(arr: np.ndarray, size: int = cfg.IMG_SIZE) -> np.ndarray:
    """Resize 2-D array to (size, size) using bicubic interpolation."""
    return cv2.resize(arr, (size, size), interpolation=cv2.INTER_CUBIC)


def to_swin_input(feature_2d: np.ndarray) -> np.ndarray:
    """
    Normalize + resize a 2-D feature map to (IMG_SIZE, IMG_SIZE),
    then expand to 3 channels (C, H, W) by repeating the channel axis,
    as Swin-Tiny expects 3-channel input.
    """
    arr = _normalize_2d(feature_2d)
    arr = _resize(arr, cfg.IMG_SIZE)          # (H, W)
    arr = np.stack([arr, arr, arr], axis=0)   # (3, H, W)
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline 1 – Scalogram (CWT)
# ─────────────────────────────────────────────────────────────────────────────

def extract_scalogram(wav: np.ndarray) -> np.ndarray:
    """
    Continuous Wavelet Transform scalogram using PyWavelets.
    Returns (CWT_SCALES, T) float32 array.
    """
    try:
        import pywt
    except ImportError:
        raise ImportError("PyWavelets is required: pip install PyWavelets")

    wav = _pad_or_trim(wav)
    scales = np.arange(1, cfg.CWT_SCALES + 1)
    coeffs, _ = pywt.cwt(wav, scales, "morl")   # (scales, T)
    power = np.abs(coeffs) ** 2
    return to_swin_input(power.astype(np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline 2 – Linear Spectrogram + CQT (combined)
# ─────────────────────────────────────────────────────────────────────────────

def extract_linear_spec(wav: np.ndarray) -> np.ndarray:
    """
    Constant-Q Transform (CQT) spectrogram.
    Returns amplitude in log scale. Shape: (N_BINS, T).
    """
    wav = _pad_or_trim(wav)
    C = np.abs(
        librosa.cqt(
            wav,
            sr=cfg.SAMPLE_RATE,
            hop_length=cfg.HOP_LENGTH,
            n_bins=cfg.N_BINS,
            bins_per_octave=cfg.BINS_PER_OCT,
        )
    )
    log_C = librosa.amplitude_to_db(C, ref=np.max)
    return to_swin_input(log_C.astype(np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline 3 – Log-Mel Spectrogram
# ─────────────────────────────────────────────────────────────────────────────

def extract_logmel(wav: np.ndarray) -> np.ndarray:
    """
    Log-Mel spectrogram using a mel-scale filterbank.
    Returns (N_MELS, T) float32 array resized to (IMG_SIZE, IMG_SIZE).
    """
    wav = _pad_or_trim(wav)

    mel = librosa.feature.melspectrogram(
        y=wav, sr=cfg.SAMPLE_RATE,
        n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH, n_mels=cfg.N_MELS,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)   # (N_MELS, T)
    return to_swin_input(log_mel.astype(np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline 4 – MFCC (+ delta + delta-delta)
# ─────────────────────────────────────────────────────────────────────────────

def extract_mfcc(wav: np.ndarray) -> np.ndarray:
    """
    MFCC stacked with its first- and second-order deltas.
    Returns (N_MFCC * 3, T) float32 array resized to (IMG_SIZE, IMG_SIZE).

    The three rows per coefficient encode static + velocity + acceleration,
    giving the classifier richer temporal dynamics information.
    """
    wav = _pad_or_trim(wav)

    mfcc = librosa.feature.mfcc(
        y=wav, sr=cfg.SAMPLE_RATE, n_mfcc=cfg.N_MFCC,
        n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH,
    )                                                  # (N_MFCC, T)

    delta1 = librosa.feature.delta(mfcc, order=1)     # (N_MFCC, T)
    delta2 = librosa.feature.delta(mfcc, order=2)     # (N_MFCC, T)

    combined = np.vstack([mfcc, delta1, delta2])       # (N_MFCC*3, T)
    return to_swin_input(combined.astype(np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline 5 – SSL Embeddings (Wav2Vec 2.0)
# ─────────────────────────────────────────────────────────────────────────────

_ssl_processor = None
_ssl_model = None


def _load_ssl_model():
    global _ssl_processor, _ssl_model
    if _ssl_model is None:
        from transformers import Wav2Vec2Processor, Wav2Vec2Model
        import torch
        _ssl_processor = Wav2Vec2Processor.from_pretrained(cfg.SSL_MODEL_NAME)
        _ssl_model = Wav2Vec2Model.from_pretrained(cfg.SSL_MODEL_NAME)
        _ssl_model.eval()
        # Move to GPU if available
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        _ssl_model = _ssl_model.to(device)
    return _ssl_processor, _ssl_model


def extract_ssl_embeddings(wav: np.ndarray) -> np.ndarray:
    """
    Extract hidden-state features from Wav2Vec2 (last hidden layer).
    Shape: (T_frames, hidden_dim) -> reshaped & resized to (IMG_SIZE, IMG_SIZE).
    """
    import torch

    wav = _pad_or_trim(wav)
    processor, model = _load_ssl_model()
    device = next(model.parameters()).device

    inputs = processor(
        wav, sampling_rate=cfg.SAMPLE_RATE,
        return_tensors="pt", padding=True,
    )
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values)

    # hidden_states: (1, T, hidden_dim)
    hidden = outputs.last_hidden_state[0].cpu().numpy()   # (T, hidden_dim)
    return to_swin_input(hidden.astype(np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

EXTRACTORS = {
    "scalogram":      extract_scalogram,
    "linear_spec":    extract_linear_spec,
    "logmel":         extract_logmel,
    "mfcc":           extract_mfcc,
    "ssl_embeddings": extract_ssl_embeddings,
}
