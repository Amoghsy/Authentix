"""
heuristic_service.py
────────────────────
Support-layer heuristic analysis for the Authentix pipeline.

Updated weights:
    heuristic_score = 0.80 * video_heur + 0.20 * audio_heur
"""

import math

from app.utils.config import HEURISTIC_VIDEO_WEIGHT, HEURISTIC_AUDIO_WEIGHT


def _sigmoid(x: float, scale: float = 1.0) -> float:
    """Map any real value to (0, 1). scale controls sensitivity."""
    return 1.0 / (1.0 + math.exp(-scale * x))


def compute_video_heuristic(variance: float, flicker: float, blur: float) -> float:
    """
    Properly scaled video heuristic.

    Normalization ranges (empirically tuned):
        blur     : 0–10000  (Laplacian variance)
        flicker  : 0–100    (mean abs frame diff)
        variance : 0–10000  (pixel intensity variance)

    Returns:
        Weighted score in [0, 1].
    """
    var_norm     = min(1.0, variance / 10000)
    flicker_norm = min(1.0, flicker / 100)
    blur_norm    = min(1.0, blur / 10000)

    score = 0.4 * blur_norm + 0.3 * flicker_norm + 0.3 * var_norm
    return score


def heuristic_analysis(features: dict) -> dict:
    """
    Compute heuristic deepfake scores from raw extracted features.

    Args:
        features: dict with "video" and "audio" sub-dicts produced by
                  feature_extraction.extract_features()

    Returns:
        {
            "video_heuristic_score": float,
            "audio_heuristic_score": float,
            "heuristic_score":       float,
            "issues":                List[str]
        }
    """
    video = features.get("video", {})
    audio = features.get("audio", {})

    # --- VIDEO HEURISTICS (properly scaled) ---
    variance = video.get("variance", 0.0)
    flicker  = video.get("flicker", 0.0)
    blur     = video.get("blur", 0.0)

    video_heuristic_score = compute_video_heuristic(variance, flicker, blur)

    # --- AUDIO HEURISTICS ---
    # Low pitch variance → monotone / synthetic voice
    pitch_var = audio.get("pitch_var", 0.0)
    pitch_score = 1.0 - _sigmoid(pitch_var - 500.0, scale=0.002)  # low var → high score

    # Low energy → silence padding common in voice clones
    energy = audio.get("energy", 0.0)
    energy_score = 1.0 - _sigmoid(energy - 0.05, scale=20.0)

    audio_heuristic_score = (pitch_score + energy_score) / 2

    # ── Weighted combination (0.8 video + 0.2 audio) ──────────────────────
    heuristic_score = (
        HEURISTIC_VIDEO_WEIGHT * video_heuristic_score +
        HEURISTIC_AUDIO_WEIGHT * audio_heuristic_score
    )

    # --- ISSUES LIST ---
    issues = []
    if video.get("blur", 0.0) > 2000:
        issues.append("High blur detected – possible GAN face synthesis")
    if video.get("flicker", 0.0) > 30:
        issues.append("Temporal flicker detected – frame inconsistency")
    if pitch_score > 0.6:
        issues.append("Low pitch variation – synthetic or cloned voice suspected")
    if energy_score > 0.6:
        issues.append("Abnormally low audio energy – silence padding detected")

    return {
        "video_heuristic_score": round(video_heuristic_score, 4),
        "audio_heuristic_score": round(audio_heuristic_score, 4),
        "heuristic_score":       round(heuristic_score, 4),
        "issues":                issues,
    }