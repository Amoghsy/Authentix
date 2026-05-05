"""
explanation_service.py
──────────────────────
Smart, signal-driven explanation and attack attribution for Authentix.

Generates dynamic explanations based on individual model signal strengths,
not just heuristic thresholds.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# Signal thresholds for triggering explanations
_HIGH = 0.6    # signal is strong
_MEDIUM = 0.4  # signal is moderate


def generate_explanation(features: dict, scores: dict) -> dict:
    """
    Generate a prioritised list of deepfake indicators and an attack attribution.

    Args:
        features : Feature dict from feature_extraction.extract_features().
        scores   : Fused score dict from fusion_service.fuse_scores().

    Returns:
        { "explanation": List[str], "attribution": str }
    """
    video = features.get("video", {})
    audio = features.get("audio", {})

    final_score: float    = scores.get("final_score", 0.0)
    status: str           = scores.get("status", "UNKNOWN")
    model_score: float    = scores.get("model_score", 0.0)
    heuristic_score: float = scores.get("heuristic_score", 0.0)

    # Individual signal scores
    cnn_score       = scores.get("cnn_score", 0.0)
    temporal_score  = scores.get("temporal_score", 0.0)
    fft_score       = scores.get("fft_score", 0.0)
    deepface_score  = scores.get("deepface_score", 0.0)
    landmark_score  = scores.get("landmark_score", 0.0)
    lip_sync_score  = scores.get("lip_sync_score", 0.0)

    video_model = scores.get("video_model_score", 0.0)
    audio_model = scores.get("audio_model_score", 0.0)

    explanation: List[str] = []
    flags: List[Tuple[str, str]] = []

    # ── CNN-based detection ────────────────────────────────────────────────
    if cnn_score >= _HIGH:
        explanation.append(
            f"CNN deepfake detector flagged synthetic patterns (score: {cnn_score:.2f}). "
            "EfficientNet feature analysis indicates non-natural image generation."
        )
        flags.append(("video", "cnn"))
    elif cnn_score >= _MEDIUM:
        explanation.append(
            f"CNN detector shows moderate synthetic indicators (score: {cnn_score:.2f})."
        )
        flags.append(("video", "cnn"))

    # ── Temporal consistency ───────────────────────────────────────────────
    if temporal_score >= _HIGH:
        explanation.append(
            f"Significant frame-to-frame identity instability detected (score: {temporal_score:.2f}). "
            "Facial embeddings drift across frames, inconsistent with real video."
        )
        flags.append(("video", "temporal"))
    elif temporal_score >= _MEDIUM:
        explanation.append(
            f"Moderate temporal inconsistency detected (score: {temporal_score:.2f})."
        )
        flags.append(("video", "temporal"))

    # ── FFT frequency artifacts ────────────────────────────────────────────
    if fft_score >= _HIGH:
        explanation.append(
            f"GAN frequency-domain artifacts detected (score: {fft_score:.2f}). "
            "Abnormal high-frequency energy pattern is characteristic of synthetic generation."
        )
        flags.append(("video", "fft"))
    elif fft_score >= _MEDIUM:
        explanation.append(
            f"Elevated high-frequency noise detected in spectral analysis (score: {fft_score:.2f})."
        )
        flags.append(("video", "fft"))

    # ── DeepFace stability ─────────────────────────────────────────────────
    if deepface_score >= _HIGH:
        explanation.append(
            f"Facial embedding variance indicates identity manipulation (score: {deepface_score:.2f})."
        )
        flags.append(("video", "deepface"))

    # ── Landmark distortion ────────────────────────────────────────────────
    if landmark_score >= _HIGH:
        explanation.append(
            f"Facial landmark distortion detected (score: {landmark_score:.2f}). "
            "Geometric asymmetry or landmark jitter suggests face manipulation."
        )
        flags.append(("video", "landmark"))
    elif landmark_score >= _MEDIUM:
        explanation.append(
            f"Moderate facial geometry inconsistency detected (score: {landmark_score:.2f})."
        )
        flags.append(("video", "landmark"))

    # ── Lip sync ───────────────────────────────────────────────────────────
    if lip_sync_score >= _HIGH:
        explanation.append(
            f"Audio-visual desynchronization detected (score: {lip_sync_score:.2f}). "
            "Mouth motion does not correlate with audio energy — possible voice clone or dub."
        )
        flags.append(("audio", "lip_sync"))
    elif lip_sync_score >= _MEDIUM:
        explanation.append(
            f"Moderate lip-sync mismatch detected (score: {lip_sync_score:.2f})."
        )
        flags.append(("audio", "lip_sync"))

    # ── Heuristic indicators (low-level) ───────────────────────────────────
    blur = float(video.get("blur", 0.0))
    flicker = float(video.get("flicker", 0.0))
    variance = float(video.get("variance", 0.0))
    pitch_var = float(audio.get("pitch_var", 0.0))
    energy = float(audio.get("energy", 0.0))

    if blur > 100:
        explanation.append(
            f"Unusually high facial blur ({blur:.1f}) indicates synthetic face rendering."
        )
        flags.append(("video", "blur"))

    if flicker > 15:
        explanation.append(
            f"Temporal flicker detected ({flicker:.2f}) — frame-level manipulation or stitching."
        )
        flags.append(("video", "flicker"))

    if variance > 500:
        explanation.append(
            f"High pixel variance ({variance:.1f}) suggests inconsistent frame textures."
        )
        flags.append(("video", "variance"))

    if pitch_var < 200:
        explanation.append(
            f"Very low pitch variation ({pitch_var:.1f}) — characteristic of TTS or voice cloning."
        )
        flags.append(("audio", "pitch"))

    if energy < 0.02:
        explanation.append(
            f"Abnormally low audio energy ({energy:.4f}) — silence padding or muted track."
        )
        flags.append(("audio", "energy"))

    # ── Model-layer summary ────────────────────────────────────────────────
    explanation.append(
        f"Hybrid confidence {final_score:.4f} ({status}) — "
        f"model {model_score:.4f} [video: {video_model:.4f}, audio: {audio_model:.4f}] "
        f"+ heuristic {heuristic_score:.4f}."
    )

    if len(explanation) == 1:
        explanation.insert(0, "No strong deepfake indicators found in this sample.")

    # ── Attack attribution ─────────────────────────────────────────────────
    # Only assert attack-type attribution when the verdict is FAKE.
    # When REAL, any triggered flags are informational noise, not an attack claim.
    video_flags = {f for m, f in flags if m == "video"}
    audio_flags = {f for m, f in flags if m == "audio"}

    # If verdict is REAL, override attribution regardless of individual flags
    if status == "REAL":
        if flags:
            attribution = (
                "Sample classified as authentic — minor anomalies observed "
                "but within normal thresholds"
            )
        else:
            attribution = "No clear attribution — sample appears authentic"
    else:
        # FAKE — produce specific attack attribution
        has_cnn      = "cnn" in video_flags
        has_temporal = "temporal" in video_flags
        has_fft      = "fft" in video_flags
        has_landmark = "landmark" in video_flags
        has_lip_sync = "lip_sync" in audio_flags
        has_audio    = bool(audio_flags & {"pitch", "energy", "lip_sync"})
        has_video    = bool(video_flags & {"cnn", "temporal", "fft", "blur", "flicker", "landmark"})

        if has_cnn and has_temporal:
            attribution = "GAN / Diffusion-based face swap (CNN + temporal drift)"
        elif has_cnn and has_fft:
            attribution = "GAN-synthesized content (CNN + frequency artifacts)"
        elif has_fft and has_landmark:
            attribution = "Face manipulation with GAN artifacts (FFT + landmark distortion)"
        elif has_lip_sync and has_video:
            attribution = "Face swap with voice clone (visual + lip-sync mismatch)"
        elif has_cnn:
            attribution = "Deep learning-generated synthetic face"
        elif has_temporal and has_landmark:
            attribution = "Face swap with geometric distortion"
        elif has_fft:
            attribution = "GAN frequency-domain artifacts detected"
        elif has_lip_sync:
            attribution = "Audio-visual desynchronization (possible voice clone)"
        elif has_video and has_audio:
            attribution = "Multi-modal deepfake indicators (video + audio anomalies)"
        elif has_video:
            attribution = "Visual deepfake indicators detected"
        elif has_audio:
            attribution = "Audio synthesis / voice cloning indicators"
        else:
            attribution = "Uncharacterised deepfake indicators detected"

    logger.info(
        "Explanation | %d finding(s) | attribution: %s",
        len(explanation), attribution,
    )

    return {
        "explanation": explanation,
        "attribution": attribution,
    }
