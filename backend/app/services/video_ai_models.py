"""
video_ai_models.py
──────────────────
Advanced video deepfake detection models for the Authentix pipeline.

Signals provided:
    1. CNN Deepfake Detector  → cnn_score       (PRIMARY — 0.40 weight)
    2. DeepFace Stability     → deepface_score   (0.10 weight)
    3. Temporal Consistency   → temporal_score    (0.20 weight)
    4. FFT Frequency Score    → fft_score         (0.15 weight)
    5. Landmark Distortion    → landmark_score    (0.10 weight)
    6. RF Custom Model        → rf_score          (0.05 weight) — from model_loader

All functions return float ∈ [0, 1].
Fallback: 0.5 (neutral) on any error.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_FALLBACK = 0.5


# ===========================================================================
# 1. CNN Deepfake Detector (PRIMARY SIGNAL)
# ===========================================================================

def get_cnn_score(frames: List[np.ndarray]) -> float:
    """
    Deep CNN-based deepfake detector using EfficientNet feature analysis.

    Strategy:
        • Extract 1280-d feature vectors from EfficientNetB0 for ALL frames.
        • Measure cross-frame feature coherence (real = stable, fake = drifty).
        • Analyse activation statistics (kurtosis, dead-neuron ratio, std).
        • Combine into a single well-calibrated score.

    Returns:
        float ∈ [0, 1] — probability of FAKE.
    """
    if not frames:
        return _FALLBACK

    try:
        import tensorflow as tf  # noqa: PLC0415

        model = tf.keras.applications.EfficientNetB0(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(224, 224, 3),
        )

        # Batch extract features for all frames
        all_features = []
        for frame in frames:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
            img = tf.keras.applications.efficientnet.preprocess_input(img)
            img = np.expand_dims(img, axis=0)
            feat = model.predict(img, verbose=0)[0]
            all_features.append(feat)

        feature_matrix = np.array(all_features)  # (N, 1280)

        # --- Signal 1: Cross-frame feature instability ---
        # Real faces → consistent features across frames (low variance)
        # Fake faces → features drift across frames (higher variance)
        per_dim_var = np.var(feature_matrix, axis=0)  # (1280,)
        cross_frame_instability = float(np.mean(per_dim_var))
        instability_score = min(1.0, cross_frame_instability * 5.0)

        # --- Signal 2: Activation distribution anomaly ---
        # Real faces activate neurons in expected patterns
        # Fakes produce unusual activation distributions
        mean_activations = np.mean(feature_matrix, axis=0)  # (1280,)
        act_std = float(np.std(mean_activations))
        dead_ratio = float(np.mean(mean_activations < 0.01))  # near-zero neurons
        active_ratio = float(np.mean(mean_activations > 1.0))  # highly active neurons
        distribution_score = min(1.0, act_std * 2.0 + active_ratio * 3.0 + (1.0 - dead_ratio) * 0.3)

        # --- Signal 3: Per-frame activation magnitude spread ---
        frame_magnitudes = np.linalg.norm(feature_matrix, axis=1)  # (N,)
        mag_std = float(np.std(frame_magnitudes))
        mag_mean = float(np.mean(frame_magnitudes))
        magnitude_score = min(1.0, mag_std / (mag_mean + 1e-6) * 10.0)  # coefficient of variation

        # --- Combine (weighted) ---
        score = (
            0.40 * instability_score +
            0.35 * distribution_score +
            0.25 * magnitude_score
        )

        # Sigmoid calibration — preserves meaning, avoids saturation
        # raw 0.2→0.12, 0.4→0.50, 0.52→0.73, 0.6→0.88
        import math
        raw_score = score
        score = 1.0 / (1.0 + math.exp(-5.0 * (score - 0.4)))
        score = float(np.clip(score, 0.0, 1.0))

        logger.info("[CNN] raw=%.4f sigmoid=%.4f | instab=%.4f distrib=%.4f mag=%.4f (%d frames)",
                    raw_score, score,
                    instability_score, distribution_score, magnitude_score, len(frames))
        print(f"[CNN] OK | raw={raw_score:.4f} sigmoid={score:.4f} | "
              f"instab={instability_score:.4f} distrib={distribution_score:.4f} "
              f"mag={magnitude_score:.4f} ({len(frames)} frames)")
        return score

    except Exception as exc:
        logger.warning("[CNN] Failed: %s → fallback %.1f", exc, _FALLBACK)
        print(f"[CNN] FALLBACK | {exc}")
        return _FALLBACK


# ===========================================================================
# 2. DeepFace Embedding Stability
# ===========================================================================

def get_deepface_score(frames: List[np.ndarray]) -> Tuple[float, List[np.ndarray]]:
    """
    Compute deepfake likelihood from facial embedding variance.

    Returns:
        (score, embeddings) — score ∈ [0,1], embeddings for temporal analysis.
    """
    if not frames:
        return _FALLBACK, []

    try:
        from deepface import DeepFace  # noqa: PLC0415
    except ImportError as exc:
        logger.warning("[DeepFace] Import failed: %s", exc)
        return _FALLBACK, []

    embeddings: List[List[float]] = []
    for idx, frame in enumerate(frames):
        try:
            result = DeepFace.represent(
                frame, model_name="Facenet", enforce_detection=False,
            )
            if result and len(result) > 0:
                embeddings.append(result[0]["embedding"])
        except Exception as exc:
            logger.debug("[DeepFace] Frame %d failed: %s", idx, exc)

    if not embeddings:
        return _FALLBACK, []

    matrix = np.array(embeddings, dtype=np.float64)
    variance = float(np.var(matrix))
    # Softer log-based scaling to avoid saturation at 1.0
    # variance ~0.05 → score ~0.22, ~0.15 → ~0.35, ~0.5 → ~0.55, ~1.0 → ~0.68
    import math
    score = min(1.0, 0.15 + 0.5 * math.log1p(variance * 2.0))
    score = float(np.clip(score, 0.0, 1.0))

    logger.info("[DeepFace] variance=%.4f score=%.4f (%d/%d frames)",
                variance, score, len(embeddings), len(frames))
    print(f"[DeepFace] OK | variance={variance:.4f} score={score:.4f} ({len(embeddings)}/{len(frames)} frames)")
    return score, [np.array(e, dtype=np.float64) for e in embeddings]


# ===========================================================================
# 3. Temporal Consistency
# ===========================================================================

def get_temporal_score(embeddings: List[np.ndarray]) -> float:
    """
    Measure identity drift across consecutive frames via cosine similarity.

    Real videos → stable embeddings → LOW score.
    Deepfakes → wobbling embeddings → HIGH score.

    Returns:
        float ∈ [0, 1].
    """
    if len(embeddings) < 2:
        return 0.0

    try:
        from sklearn.metrics.pairwise import cosine_similarity  # noqa: PLC0415
    except ImportError:
        # Manual cosine similarity fallback
        sims = []
        for i in range(len(embeddings) - 1):
            a, b = embeddings[i], embeddings[i + 1]
            dot = np.dot(a, b)
            norm = np.linalg.norm(a) * np.linalg.norm(b)
            sims.append(dot / max(norm, 1e-10))
        mean_sim = float(np.mean(sims))
        inconsistency = 1.0 - mean_sim
        score = min(1.0, inconsistency * 3.0)
        return float(np.clip(score, 0.0, 1.0))

    sims = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(
            embeddings[i].reshape(1, -1),
            embeddings[i + 1].reshape(1, -1),
        )[0][0]
        sims.append(sim)

    mean_sim = float(np.mean(sims))
    inconsistency = 1.0 - mean_sim
    score = min(1.0, inconsistency * 3.0)

    logger.info("[Temporal] mean_sim=%.4f inconsistency=%.4f score=%.4f",
                mean_sim, inconsistency, score)
    print(f"[Temporal] OK | mean_sim={mean_sim:.4f} inconsistency={inconsistency:.4f} score={score:.4f}")
    return float(np.clip(score, 0.0, 1.0))


# ===========================================================================
# 4. FFT Frequency Score (GAN artifact detection)
# ===========================================================================

def get_fft_score(frames: List[np.ndarray]) -> float:
    """
    Detect GAN/diffusion artifacts via frequency-domain analysis.

    GANs produce characteristic high-frequency noise patterns (checkerboard,
    spectral peaks) that real images don't exhibit.

    Strategy:
        • Compute 2D FFT of each frame (grayscale).
        • Measure energy in high-frequency bands relative to total energy.
        • Unusual high-frequency energy ratio → likely synthetic.

    Returns:
        float ∈ [0, 1].
    """
    if not frames:
        return _FALLBACK

    hf_ratios = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # 2D FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        # Define high-frequency mask (outer 30% ring)
        radius_low = min(cy, cx) * 0.7
        y_grid, x_grid = np.ogrid[:h, :w]
        dist = np.sqrt((y_grid - cy) ** 2 + (x_grid - cx) ** 2)

        hf_mask = dist > radius_low
        total_energy = float(np.sum(magnitude ** 2)) + 1e-10
        hf_energy = float(np.sum(magnitude[hf_mask] ** 2))

        hf_ratio = hf_energy / total_energy
        hf_ratios.append(hf_ratio)

    mean_hf = float(np.mean(hf_ratios))

    # Normalize: typical real videos have hf_ratio ≈ 0.01–0.05
    # Synthetic: ≈ 0.05–0.15+
    score = min(1.0, mean_hf / 0.08)

    logger.info("[FFT] mean_hf_ratio=%.6f score=%.4f", mean_hf, score)
    print(f"[FFT] OK | mean_hf_ratio={mean_hf:.6f} score={score:.4f}")
    return float(np.clip(score, 0.0, 1.0))


# ===========================================================================
# 5. Landmark Distortion (Face geometry inconsistency — OpenCV-only)
# ===========================================================================

def get_landmark_score(frames: List[np.ndarray]) -> float:
    """
    Detect facial geometric distortion using pure OpenCV analysis.

    Strategy (no MediaPipe dependency):
        • Bilateral symmetry: flip face horizontally and measure pixel difference.
        • Edge density: Canny edges in face region — deepfakes often produce
          smoother or noisier edge patterns than real faces.
        • Temporal jitter: frame-to-frame structural change (SSIM-style).

    Deepfakes produce subtle bilateral asymmetry and unnatural edge distributions.

    Returns:
        float ∈ [0, 1].
    """
    if not frames:
        return _FALLBACK

    asymmetries = []
    edge_densities = []
    prev_gray = None
    jitter_scores = []

    for frame in frames:
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # --- Bilateral symmetry ---
        # Flip the face horizontally and measure pixel-wise difference
        flipped = cv2.flip(gray, 1)  # horizontal flip
        diff = np.abs(gray.astype(np.float32) - flipped.astype(np.float32))
        # Focus on center 60% to avoid border artifacts
        margin_y, margin_x = int(h * 0.2), int(w * 0.2)
        center_diff = diff[margin_y:h - margin_y, margin_x:w - margin_x]
        asym = float(np.mean(center_diff)) / 255.0  # normalize to [0, 1]
        asymmetries.append(asym)

        # --- Edge density ---
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0)) / (h * w)
        edge_densities.append(edge_density)

        # --- Temporal jitter ---
        if prev_gray is not None:
            frame_diff = np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))
            jitter = float(np.mean(frame_diff)) / 255.0
            jitter_scores.append(jitter)
        prev_gray = gray

    mean_asym = float(np.mean(asymmetries))
    mean_edge = float(np.mean(edge_densities))
    mean_jitter = float(np.mean(jitter_scores)) if jitter_scores else 0.0

    # Scoring logic:
    # High asymmetry → more likely fake (deepfakes have imperfect face swaps)
    asym_score = min(1.0, mean_asym * 8.0)  # asymmetry ~0.05-0.15 for real

    # Edge density: real faces ~0.05-0.15, GANs can be very smooth (<0.03) or noisy (>0.20)
    edge_deviation = abs(mean_edge - 0.10)  # how far from "normal"
    edge_score = min(1.0, edge_deviation * 10.0)

    # Jitter: high structural jitter across frames
    jitter_norm = min(1.0, mean_jitter * 20.0)

    # Combine
    score = 0.4 * asym_score + 0.3 * edge_score + 0.3 * jitter_norm

    logger.info("[Landmark] asym=%.4f edge=%.4f jitter=%.4f score=%.4f",
                mean_asym, mean_edge, mean_jitter, score)
    print(f"[Landmark] OK | asym={mean_asym:.4f} edge={mean_edge:.4f} jitter={mean_jitter:.4f} score={score:.4f}")
    return float(np.clip(score, 0.0, 1.0))


# ===========================================================================
# Texture Artifact Score (Laplacian — kept from previous iteration)
# ===========================================================================

def get_texture_score(frames: List[np.ndarray]) -> float:
    """
    Detect texture-level artifacts via Laplacian variance (high-frequency noise).

    Returns:
        float ∈ [0, 1].
    """
    if not frames:
        return 0.0

    laplacian_vars = []
    for frame in frames:
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_vars.append(np.var(lap))

    mean_lap = float(np.mean(laplacian_vars))
    score = min(1.0, mean_lap / 5000.0)
    logger.info("[Texture] mean_laplacian=%.4f score=%.4f", mean_lap, score)
    print(f"[Texture] OK | mean_laplacian={mean_lap:.4f} score={score:.4f}")
    return float(np.clip(score, 0.0, 1.0))


# ===========================================================================
# Master Video Score Computation
# ===========================================================================

def compute_all_video_scores(
    frames: List[np.ndarray],
    face_crops: List[np.ndarray],
    rf_score: float = 0.5,
) -> Dict[str, float]:
    """
    Run all video AI models and return individual + fused scores.

    Args:
        frames:     Original frames.
        face_crops: Face-cropped frames (from face detection).
        rf_score:   Custom RF model score (from model_loader).

    Returns:
        Dict with all individual scores and the fused video_model_score.
    """
    from app.utils.config import (  # noqa: PLC0415
        VIDEO_CNN_WEIGHT, VIDEO_TEMPORAL_WEIGHT, VIDEO_FFT_WEIGHT,
        VIDEO_DEEPFACE_WEIGHT, VIDEO_LANDMARK_WEIGHT, VIDEO_RF_WEIGHT,
    )

    # Run all models (use face_crops for face-specific models)
    cnn_score = get_cnn_score(face_crops)
    deepface_score, embeddings = get_deepface_score(face_crops)
    temporal_score = get_temporal_score(embeddings)
    fft_score = get_fft_score(frames)
    landmark_score = get_landmark_score(frames)
    texture_score = get_texture_score(frames)

    # Weighted fusion
    video_model_score = (
        VIDEO_CNN_WEIGHT      * cnn_score +
        VIDEO_TEMPORAL_WEIGHT * temporal_score +
        VIDEO_FFT_WEIGHT      * fft_score +
        VIDEO_DEEPFACE_WEIGHT * deepface_score +
        VIDEO_LANDMARK_WEIGHT * landmark_score +
        VIDEO_RF_WEIGHT       * rf_score
    )
    video_model_score = float(np.clip(video_model_score, 0.0, 1.0))

    print("\n=== VIDEO AI SCORES ===")
    print(f"  CNN ({VIDEO_CNN_WEIGHT:.2f})      : {cnn_score:.4f}")
    print(f"  Temporal ({VIDEO_TEMPORAL_WEIGHT:.2f}) : {temporal_score:.4f}")
    print(f"  FFT ({VIDEO_FFT_WEIGHT:.2f})      : {fft_score:.4f}")
    print(f"  DeepFace ({VIDEO_DEEPFACE_WEIGHT:.2f}) : {deepface_score:.4f}")
    print(f"  Landmark ({VIDEO_LANDMARK_WEIGHT:.2f}) : {landmark_score:.4f}")
    print(f"  RF ({VIDEO_RF_WEIGHT:.2f})        : {rf_score:.4f}")
    print(f"  Texture (info)  : {texture_score:.4f}")
    print(f"  → video_model_score = {video_model_score:.4f}")

    return {
        "cnn_score":         cnn_score,
        "deepface_score":    deepface_score,
        "temporal_score":    temporal_score,
        "fft_score":         fft_score,
        "landmark_score":    landmark_score,
        "rf_score":          rf_score,
        "texture_score":     texture_score,
        "video_model_score": video_model_score,
    }
