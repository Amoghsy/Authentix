"""
fusion_service.py
─────────────────
Final score fusion for the Authentix deepfake detection pipeline.

Fusion formula (3-step direct injection):
    Step 1: model_score = 0.50 * video + 0.50 * audio  (cross-modal)
    Step 2: boosted     = model_score ** 0.85
    Step 3: final_score = 0.65 * boosted
                        + 0.25 * lip_sync_score
                        + 0.10 * deepface_score

Decision engine (binary):
    final_score >= 0.5 → "FAKE"
    else               → "REAL"
"""

from __future__ import annotations

import logging

import numpy as np

from app.utils.config import (
    FAKE_THRESHOLD,
    FUSION_NONLINEAR_POWER,
    FUSION_MODEL_WEIGHT,
    FUSION_LIPSYNC_WEIGHT,
    FUSION_DEEPFACE_WEIGHT,
)

logger = logging.getLogger(__name__)


def _safe(values: dict, key: str, fallback: float = 0.5) -> float:
    """Return values[key] as float, or fallback with a warning if absent/invalid."""
    val = values.get(key)
    if isinstance(val, (int, float)) and np.isfinite(val):
        return float(np.clip(val, 0.0, 1.0))
    logger.warning("Fusion | Key '%s' missing or non-finite → fallback %.2f", key, fallback)
    return fallback


def fuse_scores(model_scores: dict, heuristic_scores: dict) -> dict:
    """
    Combine hybrid model scores and heuristic scores into a final verdict.

    Args:
        model_scores:     Dict from model_loader.predict().
        heuristic_scores: Dict from heuristic_service.heuristic_analysis().

    Returns:
        Fully populated result dict.
    """
    model_score     = _safe(model_scores,     "model_score")

    # Step 2: non-linear boost
    boosted_model = model_score ** FUSION_NONLINEAR_POWER

    # Step 3: direct signal injection
    deepface_score = _safe(model_scores, "deepface_score", 0.0)
    lip_sync_score = _safe(model_scores, "lip_sync_score", 0.0)

    final_score = (
        FUSION_MODEL_WEIGHT    * boosted_model  +
        FUSION_LIPSYNC_WEIGHT  * lip_sync_score +
        FUSION_DEEPFACE_WEIGHT * deepface_score
    )
    final_score = float(np.clip(final_score, 0.0, 1.0))

    # Binary decision
    status = "FAKE" if final_score >= FAKE_THRESHOLD else "REAL"

    logger.info(
        "Fusion — model: %.4f | boosted: %.4f | lip_sync: %.4f | deepface: %.4f | final: %.4f | verdict: %s",
        model_score, boosted_model, lip_sync_score, deepface_score, final_score, status,
    )

    # --- Build result dict ---
    video_model_score = round(_safe(model_scores,     "video_model_score"), 4)
    audio_model_score = round(_safe(model_scores,     "audio_model_score"), 4)
    video_heuristic   = round(_safe(heuristic_scores, "video_heuristic_score"), 4)
    audio_heuristic   = round(_safe(heuristic_scores, "audio_heuristic_score"), 4)
    heuristic_score   = round(_safe(heuristic_scores, "heuristic_score", 0.0), 4)  # info only

    return {
        "status":     status,
        "confidence": round(final_score, 4),
        "model_prediction": {
            "video_model_score": video_model_score,
            "audio_model_score": audio_model_score,
            "model_score":       round(model_score, 4),
        },
        "heuristic_analysis": {
            "video_heur":      video_heuristic,
            "audio_heur":      audio_heuristic,
            "heuristic_score": round(heuristic_score, 4),
        },
        "fusion": {
            "final_score": round(final_score, 4),
        },
        # Flat fields for backward compat
        "final_score":            round(final_score, 4),
        "model_score":            round(model_score, 4),
        "heuristic_score":        round(heuristic_score, 4),
        "video_model_score":      video_model_score,
        "audio_model_score":      audio_model_score,
        "video_heuristic_score":  video_heuristic,
        "audio_heuristic_score":  audio_heuristic,
        # Individual signal scores
        "cnn_score":         round(_safe(model_scores, "cnn_score", 0.0), 4),
        "temporal_score":    round(_safe(model_scores, "temporal_score", 0.0), 4),
        "fft_score":         round(_safe(model_scores, "fft_score", 0.0), 4),
        "deepface_score":    round(_safe(model_scores, "deepface_score", 0.0), 4),
        "landmark_score":    round(_safe(model_scores, "landmark_score", 0.0), 4),
        "lip_sync_score":    round(_safe(model_scores, "lip_sync_score", 0.0), 4),
    }
