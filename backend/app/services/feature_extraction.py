"""
feature_extraction.py
─────────────────────
Multimodal raw feature extractor for the Authentix deepfake detection pipeline.

Responsibilities (SLIMMED — AI models moved to video_ai_models.py / audio_ai_models.py)
──────────────────────────────────────────────────────────────────────────────────────────
1. extract_video_features()  → raw video statistics for RF model + heuristics
                               (variance / flicker / blur).
2. extract_audio_features()  → raw audio statistics for LR model + heuristics
                               (mfcc_mean / pitch_var / energy).
3. extract_features()        → master extractor that combines raw features
                               and passes frames/audio_path to model_loader.

Feature name ordering MUST stay in sync with the arrays used in models/train.py.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import List

import cv2
import librosa
import numpy as np

from app.utils.config import MAX_AUDIO_SECONDS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature name constants — order must match training data column order exactly
# ---------------------------------------------------------------------------
AUDIO_FEATURE_NAMES: List[str] = ["mfcc_mean", "pitch_var", "energy"]
VIDEO_FEATURE_NAMES: List[str] = ["variance", "flicker", "blur"]


# ===========================================================================
# Raw Feature Extraction — VIDEO
# ===========================================================================

def extract_video_features(frames: List[np.ndarray]) -> dict:
    """
    Extract frame-level statistics for the custom RandomForest model and
    heuristic analysis.

    Features (must match train.py column order):
        variance : Global pixel intensity variance across all frames.
        flicker  : Mean absolute frame-to-frame difference (temporal noise).
        blur     : Mean Laplacian variance per frame (sharpness proxy).

    Args:
        frames: List of uint8 BGR numpy arrays.

    Returns:
        Dict with keys matching VIDEO_FEATURE_NAMES.
    """
    if not frames:
        return {k: 0.0 for k in VIDEO_FEATURE_NAMES}

    # Ensure uint8 — cv2.Laplacian requires it.
    frames_u8 = [
        f.astype(np.uint8) if f.dtype != np.uint8 else f
        for f in frames
    ]

    arr      = np.array(frames_u8, dtype=np.float32)               # (N, H, W, C)
    variance = float(np.var(arr))
    flicker  = float(np.mean(np.abs(np.diff(arr, axis=0)))) if len(arr) > 1 else 0.0
    blur     = float(np.mean([cv2.Laplacian(f, cv2.CV_64F).var() for f in frames_u8]))

    return {"variance": variance, "flicker": flicker, "blur": blur}


# ===========================================================================
# Raw Feature Extraction — AUDIO
# ===========================================================================

def extract_audio_features(audio_path: str) -> dict:
    """
    Extract acoustic statistics for the custom LogisticRegression model and
    heuristic analysis.

    Features (must match train.py column order):
        mfcc_mean : Global mean of all MFCC coefficients.
        pitch_var : Variance of fundamental frequency estimates (YIN algorithm).
        energy    : Mean RMS energy.

    Args:
        audio_path: Absolute path to a WAV file.

    Returns:
        Dict with keys matching AUDIO_FEATURE_NAMES.
    """
    if not audio_path or not os.path.exists(audio_path):
        logger.error(
            "[Librosa] extract_audio_features | Audio file not found: '%s' → "
            "returning zeroed features %s",
            audio_path, {k: 0.0 for k in AUDIO_FEATURE_NAMES},
        )
        print(f"[Librosa] ERROR | Audio file not found: '{audio_path}' — zeroed features")
        return {k: 0.0 for k in AUDIO_FEATURE_NAMES}

    # ── Step 1: Load waveform ─────────────────────────────────────────────
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(audio_path, sr=None, duration=MAX_AUDIO_SECONDS)
        logger.debug("[Librosa] Loaded '%s' | samples=%d | sr=%d", audio_path, len(y), sr)
    except Exception as exc:
        logger.error(
            "[Librosa] LOAD FAILED | extract_audio_features could not read '%s': %s\n"
            "Possible causes: corrupt file, unsupported codec, FFmpeg not available.",
            audio_path, exc,
            exc_info=True,
        )
        print(f"[Librosa] ERROR | Cannot load '{audio_path}': {type(exc).__name__}: {exc} — zeroed features")
        return {k: 0.0 for k in AUDIO_FEATURE_NAMES}

    # ── Step 2: MFCC ──────────────────────────────────────────────────────
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = float(np.mean(mfcc))
        logger.debug("[Librosa] mfcc_mean=%.4f", mfcc_mean)
    except Exception as exc:
        logger.error(
            "[Librosa] MFCC FAILED | librosa.feature.mfcc() raised %s: %s — "
            "substituting mfcc_mean=0.0",
            type(exc).__name__, exc,
            exc_info=True,
        )
        print(f"[Librosa] ERROR | librosa.feature.mfcc() failed: {type(exc).__name__}: {exc} — mfcc_mean=0.0")
        mfcc_mean = 0.0

    # ── Step 3: Pitch variance (YIN) ──────────────────────────────────────
    try:
        pitch_var = float(np.var(librosa.yin(y, fmin=50, fmax=300)))
        logger.debug("[Librosa] pitch_var=%.4f", pitch_var)
    except Exception as exc:
        logger.error(
            "[Librosa] YIN FAILED | librosa.yin() raised %s: %s — "
            "substituting pitch_var=0.0",
            type(exc).__name__, exc,
            exc_info=True,
        )
        print(f"[Librosa] ERROR | librosa.yin() failed: {type(exc).__name__}: {exc} — pitch_var=0.0")
        pitch_var = 0.0

    # ── Step 4: RMS Energy ────────────────────────────────────────────────
    try:
        energy = float(np.mean(librosa.feature.rms(y=y)))
        logger.debug("[Librosa] energy=%.6f", energy)
    except Exception as exc:
        logger.error(
            "[Librosa] RMS FAILED | librosa.feature.rms() raised %s: %s — "
            "substituting energy=0.0",
            type(exc).__name__, exc,
            exc_info=True,
        )
        print(f"[Librosa] ERROR | librosa.feature.rms() failed: {type(exc).__name__}: {exc} — energy=0.0")
        energy = 0.0

    logger.info(
        "[Librosa] SUCCESS (features) | mfcc_mean=%.4f | pitch_var=%.4f | energy=%.6f",
        mfcc_mean, pitch_var, energy,
    )
    return {"mfcc_mean": mfcc_mean, "pitch_var": pitch_var, "energy": energy}


# ===========================================================================
# Master Feature Extractor
# ===========================================================================

def extract_features(frames: List[np.ndarray], audio_path: str,
                     face_crops: List[np.ndarray] = None) -> dict:
    """
    Orchestrate all feature extraction for both modalities.

    Returns a unified dict consumed by model_loader.predict() and
    heuristic_service.heuristic_analysis():

    {
        "video":          { variance, flicker, blur },
        "audio":          { mfcc_mean, pitch_var, energy },
        "frames":         List[np.ndarray],   # original frames for AI models
        "face_crops":     List[np.ndarray],   # face-cropped frames for AI models
        "audio_path":     str,                # path for audio AI models
        "video_available": bool,
        "audio_available": bool,
    }
    """
    video_available = bool(frames)
    audio_available = bool(audio_path and audio_path.strip())

    video_feats = extract_video_features(frames)
    audio_feats = extract_audio_features(audio_path)

    # Debug summary
    logger.info(
        "Features — video: %s | audio: %s | v_avail=%s | a_avail=%s",
        video_feats, audio_feats, video_available, audio_available,
    )

    return {
        "video":          video_feats,
        "audio":          audio_feats,
        "frames":         frames or [],
        "face_crops":     face_crops or frames or [],
        "audio_path":     audio_path or "",
        "video_available": video_available,
        "audio_available": audio_available,
    }
