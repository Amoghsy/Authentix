"""
audio_ai_models.py
──────────────────
Advanced audio deepfake detection models for the Authentix pipeline.

Signals provided:
    1. Librosa Pretrained   → librosa_score     (0.60 weight)
    2. LR Custom Model      → lr_score          (0.20 weight)
    3. Lip-Sync Consistency → lip_sync_score    (0.20 weight)

All functions return float ∈ [0, 1].
Fallback: 0.5 (neutral) on any error.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_FALLBACK = 0.5


# ===========================================================================
# 1. Librosa Pretrained Score
# ===========================================================================

def get_librosa_score(audio_path: str) -> float:
    """
    Compute deepfake likelihood from pitch variation and RMS energy.

    Formula:
        pitch_norm  = min(1, pitch_var / 3000)
        energy_norm = min(1, energy / 0.1)
        score       = (pitch_norm + energy_norm) / 2

    Returns:
        float ∈ [0, 1].
    """
    if not audio_path or not os.path.exists(audio_path):
        logger.warning("[Librosa] Audio not found: %s → fallback", audio_path)
        return _FALLBACK

    try:
        import librosa  # noqa: PLC0415
        from app.utils.config import MAX_AUDIO_SECONDS  # noqa: PLC0415

        y, sr = librosa.load(audio_path, sr=None, duration=MAX_AUDIO_SECONDS)

        # Pitch variance
        try:
            pitch = librosa.yin(y, fmin=50, fmax=300)
            pitch_var = float(np.var(pitch))
        except Exception:
            pitch_var = 0.0

        # RMS Energy
        try:
            energy = float(np.mean(librosa.feature.rms(y=y)))
        except Exception:
            energy = 0.0

        pitch_norm = min(1.0, pitch_var / 3000.0)
        energy_norm = min(1.0, energy / 0.1)
        score = (pitch_norm + energy_norm) / 2.0

        logger.info("[Librosa] pitch_var=%.4f energy=%.6f score=%.4f",
                    pitch_var, energy, score)
        print(f"[Librosa] OK | pitch_var={pitch_var:.4f} energy={energy:.6f} score={score:.4f}")
        return float(np.clip(score, 0.0, 1.0))

    except Exception as exc:
        logger.warning("[Librosa] Failed: %s → fallback", exc)
        print(f"[Librosa] FALLBACK | {exc}")
        return _FALLBACK


# ===========================================================================
# 3. Lip-Sync Consistency (OpenCV-only — no MediaPipe)
# ===========================================================================

def get_lip_sync_score(
    frames: List[np.ndarray],
    audio_path: str,
) -> float:
    """
    Measure audio-visual synchronization by correlating audio energy envelope
    with lower-face motion magnitude.

    Strategy (OpenCV-only, no MediaPipe):
        • Isolate the lower-third of each face frame (mouth region).
        • Measure pixel intensity change in the mouth ROI across frames.
        • Extract per-frame audio energy from the waveform.
        • Compute Pearson correlation between mouth motion and audio energy.
        • Low correlation → audio/visual mismatch → likely fake.

    Returns:
        float ∈ [0, 1] — higher = more likely fake (lower correlation).
    """
    if not frames or not audio_path or not os.path.exists(audio_path):
        return _FALLBACK

    if len(frames) < 3:
        return _FALLBACK

    # --- Extract mouth motion from video (lower-face pixel change) ---
    mouth_motions = []
    prev_mouth = None

    for frame in frames:
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Lower third = mouth region (approximate)
        mouth_roi = gray[int(h * 0.6):, int(w * 0.25):int(w * 0.75)]

        if prev_mouth is not None and prev_mouth.shape == mouth_roi.shape:
            diff = np.abs(mouth_roi.astype(np.float32) - prev_mouth.astype(np.float32))
            motion = float(np.mean(diff))
            mouth_motions.append(motion)

        prev_mouth = mouth_roi

    if len(mouth_motions) < 2:
        return _FALLBACK

    # --- Extract audio energy envelope ---
    try:
        import librosa  # noqa: PLC0415
        from app.utils.config import MAX_AUDIO_SECONDS  # noqa: PLC0415

        y, sr = librosa.load(audio_path, sr=None, duration=MAX_AUDIO_SECONDS)
        rms = librosa.feature.rms(y=y)[0]

        # Resample RMS to match number of motion measurements
        n_motions = len(mouth_motions)
        indices = np.linspace(0, len(rms) - 1, n_motions).astype(int)
        audio_energy = rms[indices]
    except Exception as exc:
        logger.warning("[LipSync] Audio processing failed: %s", exc)
        print(f"[LipSync] FALLBACK | {exc}")
        return _FALLBACK

    # --- Compute correlation ---
    mouth_arr = np.array(mouth_motions)
    audio_arr = np.array(audio_energy)

    # Avoid constant arrays
    if np.std(mouth_arr) < 1e-8 or np.std(audio_arr) < 1e-8:
        return _FALLBACK

    try:
        correlation = float(np.corrcoef(mouth_arr, audio_arr)[0, 1])
        if np.isnan(correlation):
            correlation = 0.0
    except Exception:
        correlation = 0.0

    # Convert: high correlation (>0.5) → real (low score), low correlation → fake (high score)
    # correlation range: -1 to 1, map to score 0-1
    score = max(0.0, min(1.0, (1.0 - correlation) / 2.0))

    logger.info("[LipSync] correlation=%.4f score=%.4f", correlation, score)
    print(f"[LipSync] OK | correlation={correlation:.4f} score={score:.4f}")
    return float(np.clip(score, 0.0, 1.0))


# ===========================================================================
# Master Audio Score Computation
# ===========================================================================

def compute_all_audio_scores(
    frames: List[np.ndarray],
    audio_path: str,
    lr_score: float = 0.5,
    audio_missing: bool = False,
) -> Dict[str, float]:
    """
    Run all audio AI models and return individual + fused scores.

    Audio fusion: 0.60 * librosa + 0.20 * lr + 0.20 * lip_sync
    (if no audio → audio_model_score = 0.5)
    """
    from app.utils.config import (  # noqa: PLC0415
        AUDIO_LIBROSA_WEIGHT, AUDIO_LR_WEIGHT, AUDIO_LIPSYNC_WEIGHT,
    )

    if audio_missing:
        print("\n=== AUDIO AI SCORES (MISSING — neutral) ===")
        return {
            "librosa_score":     0.5,
            "lr_score":          lr_score,
            "lip_sync_score":    0.5,
            "audio_model_score": 0.5,
        }

    librosa_score  = get_librosa_score(audio_path)
    lip_sync_score = get_lip_sync_score(frames, audio_path)

    # All three signals fused: 0.60 * librosa + 0.20 * lr + 0.20 * lip_sync
    audio_model_score = (
        AUDIO_LIBROSA_WEIGHT * librosa_score +
        AUDIO_LR_WEIGHT      * lr_score +
        AUDIO_LIPSYNC_WEIGHT * lip_sync_score
    )
    audio_model_score = float(np.clip(audio_model_score, 0.0, 1.0))

    print("\n=== AUDIO AI SCORES ===")
    print(f"  Librosa  ({AUDIO_LIBROSA_WEIGHT:.2f}) : {librosa_score:.4f}")
    print(f"  LR       ({AUDIO_LR_WEIGHT:.2f}) : {lr_score:.4f}")
    print(f"  LipSync  ({AUDIO_LIPSYNC_WEIGHT:.2f}) : {lip_sync_score:.4f}")
    print(f"  → audio_model_score = {audio_model_score:.4f}")

    return {
        "librosa_score":     librosa_score,
        "lr_score":          lr_score,      # kept for logging, NOT in fusion
        "lip_sync_score":    lip_sync_score,
        "audio_model_score": audio_model_score,
    }
