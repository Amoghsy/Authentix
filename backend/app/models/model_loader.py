"""
model_loader.py
───────────────
Hybrid scoring engine for the Authentix deepfake detection pipeline.

Architecture
────────────
Video (6 signals):
    video_model_score = 0.40 * cnn + 0.20 * temporal + 0.15 * fft
                      + 0.10 * deepface + 0.10 * landmark + 0.05 * rf

Audio (3 signals):
    audio_model_score = 0.60 * librosa + 0.20 * lr + 0.20 * lip_sync
    (if no audio → 0.5)

Cross-modal:
    model_score = 0.70 * video_model_score + 0.30 * audio_model_score

Fallback policy:
    • Missing model file     → neutral score 0.5
    • Non-finite features    → neutral score 0.5
    • predict_proba failure  → neutral score 0.5
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import joblib
import numpy as np

from app.utils.config import VIDEO_MODEL_WEIGHT, AUDIO_MODEL_WEIGHT

logger = logging.getLogger(__name__)

MODEL_DIR: str = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Global model cache — populated once via load_models()
# ---------------------------------------------------------------------------
audio_model   = None
video_model   = None
scaler_audio  = None
scaler_video  = None


# ===========================================================================
# Model Loading
# ===========================================================================

def _load(filename: str):
    """
    Load a joblib artifact from MODEL_DIR.  Returns None on any failure
    so callers can fall back gracefully.
    """
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        logger.warning("Model load | File not found: %s → neutral score 0.5 will be used", path)
        print(f"Model load fallback: file not found {path}, using neutral model score 0.5")
        return None
    try:
        artifact = joblib.load(path)
        logger.info("Model load | Loaded %s", filename)
        return artifact
    except Exception as exc:
        logger.warning("Model load | Failed to load %s (%s) → neutral score 0.5", path, exc)
        print(f"Model load fallback: failed to load {path} ({exc}), using neutral model score 0.5")
        return None


def load_models() -> None:
    """Populate the global model cache.  Called once at server start-up."""
    global audio_model, video_model, scaler_audio, scaler_video

    audio_model  = _load("audio_model.pkl")
    video_model  = _load("video_model.pkl")
    scaler_audio = _load("scaler_audio.pkl")
    scaler_video = _load("scaler_video.pkl")

    logger.info(
        "Model cache — audio: %s | video: %s | scaler_audio: %s | scaler_video: %s",
        "✓" if audio_model  is not None else "✗",
        "✓" if video_model  is not None else "✗",
        "✓" if scaler_audio is not None else "✗",
        "✓" if scaler_video is not None else "✗",
    )


# ===========================================================================
# Custom Model Inference
# ===========================================================================

def _predict_proba(
    model,
    scaler,
    feature_dict: dict,
    feature_order: List[str],
    stream_name: str,
) -> float:
    """
    Run predict_proba for the FAKE class and return its probability.

    Returns:
        float in [0, 1] — probability of being FAKE (class label 1).
        Returns 0.5 on any failure.
    """
    if model is None:
        logger.warning("%s | Model missing → neutral 0.5", stream_name)
        print(f"{stream_name} model fallback: model missing, using neutral score 0.5")
        return 0.5

    # Build feature vector
    try:
        vec = np.array(
            [feature_dict[k] for k in feature_order], dtype=float
        ).reshape(1, -1)
    except Exception as exc:
        logger.warning("%s | Malformed feature vector (%s) → neutral 0.5", stream_name, exc)
        print(f"{stream_name} model fallback: malformed feature vector ({exc}), using neutral score 0.5")
        return 0.5

    # Guard against NaN / Inf
    if not np.isfinite(vec).all():
        logger.warning("%s | Non-finite feature values detected → neutral 0.5", stream_name)
        print(f"{stream_name} model fallback: non-finite feature values detected, using neutral score 0.5")
        return 0.5

    try:
        if scaler is not None:
            vec = scaler.transform(vec)

        probs   = model.predict_proba(vec)[0]
        classes = list(model.classes_)

        # --- Debug block (specification requirement) ---
        print(f"\n=== MODEL DEBUG [{stream_name}] ===")
        print(f"  Classes      : {classes}")
        print(f"  Probabilities: {probs.tolist()}")
        print(f"  Prediction   : {model.predict(vec)[0]}")
        # -----------------------------------------------

        try:
            fake_idx = classes.index(1)
        except ValueError:
            fake_idx = 0   # fallback: treat index-0 as FAKE

        return float(np.clip(probs[fake_idx], 0.0, 1.0))

    except Exception as exc:
        logger.warning("%s | predict_proba failed (%s) → neutral 0.5", stream_name, exc)
        print(f"{stream_name} model fallback: predict_proba failed ({exc}), using neutral score 0.5")
        return 0.5


# ===========================================================================
# Hybrid Score Aggregation
# ===========================================================================

def predict(features: dict) -> dict:
    """
    Compute hybrid model scores using new AI model modules.

    This function:
        1. Runs custom RF/LR models for base scores
        2. Delegates to video_ai_models and audio_ai_models for all AI signals
        3. Computes cross-modal fusion

    Returns:
        Full score dictionary for all signals.
    """
    from app.services.feature_extraction import AUDIO_FEATURE_NAMES, VIDEO_FEATURE_NAMES  # noqa: PLC0415
    from app.services.video_ai_models import compute_all_video_scores  # noqa: PLC0415
    from app.services.audio_ai_models import compute_all_audio_scores  # noqa: PLC0415

    audio_available = features.get("audio_available", True)
    video_available = features.get("video_available", True)
    audio_missing   = not audio_available

    # ------------------------------------------------------------------
    # Custom model scores (RF for video, LR for audio)
    # ------------------------------------------------------------------
    if audio_available:
        custom_audio_score = _predict_proba(
            audio_model, scaler_audio,
            features["audio"], AUDIO_FEATURE_NAMES, "Audio",
        )
    else:
        logger.warning("Audio | No audio available → neutral custom score 0.5")
        print("Audio model fallback: no audio available, using neutral custom score 0.5")
        custom_audio_score = 0.5

    if video_available:
        custom_video_score = _predict_proba(
            video_model, scaler_video,
            features["video"], VIDEO_FEATURE_NAMES, "Video",
        )
    else:
        logger.warning("Video | No frames available → neutral custom score 0.5")
        print("Video model fallback: no video frames available, using neutral custom score 0.5")
        custom_video_score = 0.5

    # ------------------------------------------------------------------
    # Video AI models (CNN + DeepFace + Temporal + FFT + Landmark)
    # ------------------------------------------------------------------
    frames = features.get("frames", [])
    face_crops = features.get("face_crops", frames)

    video_scores = compute_all_video_scores(
        frames=frames,
        face_crops=face_crops,
        rf_score=custom_video_score,
    )

    # ------------------------------------------------------------------
    # Audio AI models (Librosa + Lip Sync)
    # ------------------------------------------------------------------
    audio_path = features.get("audio_path", "")
    audio_scores = compute_all_audio_scores(
        frames=frames,
        audio_path=audio_path,
        lr_score=custom_audio_score,
        audio_missing=audio_missing,
    )

    # ------------------------------------------------------------------
    # Cross-modal fusion
    # ------------------------------------------------------------------
    video_model_score = video_scores["video_model_score"]
    audio_model_score = audio_scores["audio_model_score"]

    model_score = (
        VIDEO_MODEL_WEIGHT * video_model_score +
        AUDIO_MODEL_WEIGHT * audio_model_score
    )
    model_score = float(np.clip(model_score, 0.0, 1.0))

    print(f"\n=== CROSS-MODAL FUSION ===")
    print(f"  model_score = {VIDEO_MODEL_WEIGHT:.2f} * {video_model_score:.4f} + {AUDIO_MODEL_WEIGHT:.2f} * {audio_model_score:.4f} = {model_score:.4f}")

    logger.info(
        "Hybrid scores — video: %.4f | audio: %.4f | model: %.4f",
        video_model_score, audio_model_score, model_score,
    )

    return {
        # Video breakdown
        "cnn_score":         video_scores["cnn_score"],
        "deepface_score":    video_scores["deepface_score"],
        "temporal_score":    video_scores["temporal_score"],
        "fft_score":         video_scores["fft_score"],
        "landmark_score":    video_scores["landmark_score"],
        "texture_score":     video_scores["texture_score"],
        "rf_score":          custom_video_score,
        "video_model_score": video_model_score,
        # Audio breakdown
        "librosa_score":     audio_scores["librosa_score"],
        "lr_score":          custom_audio_score,
        "lip_sync_score":    audio_scores["lip_sync_score"],
        "audio_model_score": audio_model_score,
        # Fused
        "model_score":       model_score,
        # Legacy compat
        "pretrained_video_score": video_scores["deepface_score"],
        "custom_video_score":     custom_video_score,
        "pretrained_audio_score": audio_scores["librosa_score"],
        "custom_audio_score":     custom_audio_score,
    }
