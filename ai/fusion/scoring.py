"""
scoring.py

This module contains functions for normalizing scores, aggregating confidences,
fusing scores using a weighted system, and performing consensus-based calibration
in the Authentix Fusion Engine.
"""

import logging
import math
from pathlib import Path
import sys

# Ensure project root is in path for direct execution
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Optional, List, Tuple
from ai.fusion.config import ScoreWeights, get_default_config
from ai.fusion.schemas import VideoPrediction, AudioPrediction, LipSyncPrediction

logger = logging.getLogger(__name__)


def clamp(val: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Clamps a value to be within the specified minimum and maximum bounds.
    """
    return max(min_val, min(max_val, val))


def normalize_score(score: float) -> float:
    """
    Ensures that a deepfake score is normalized to be within the range [0.0, 1.0].
    
    Args:
        score (float): Raw model output score.
        
    Returns:
        float: Normalized score.
    """
    # Safe boundary protection
    normalized = clamp(score)
    if normalized != score:
        logger.debug(f"Normalized score adjusted from {score:.4f} to {normalized:.4f}")
    return normalized


def compute_weighted_score(
    video_score: float,
    audio_score: float,
    lip_sync_score: float,
    weights: Optional[ScoreWeights] = None
) -> float:
    """
    Computes the weighted fusion score from the individual normalized scores.
    
    Args:
        video_score (float): Normalized Video deepfake score.
        audio_score (float): Normalized Audio deepfake score.
        lip_sync_score (float): Normalized Lip Sync deepfake score.
        weights (Optional[ScoreWeights]): Configurable fusion weights.
        
    Returns:
        float: Weighted fusion score.
    """
    w = weights or get_default_config().weights
    
    # Normalize inputs to protect bounds
    v_norm = normalize_score(video_score)
    a_norm = normalize_score(audio_score)
    l_norm = normalize_score(lip_sync_score)
    
    fused_score = (w.video * v_norm) + (w.audio * a_norm) + (w.lip_sync * l_norm)
    return clamp(fused_score)


def aggregate_confidence(
    video_conf: float,
    audio_conf: float,
    lip_sync_conf: float,
    weights: Optional[ScoreWeights] = None
) -> float:
    """
    Aggregates individual model confidence values using a weighted average.
    
    Args:
        video_conf (float): Video model confidence.
        audio_conf (float): Audio model confidence.
        lip_sync_conf (float): Lip Sync model confidence.
        weights (Optional[ScoreWeights]): Configurable fusion weights.
        
    Returns:
        float: Aggregated raw confidence.
    """
    w = weights or get_default_config().weights
    
    v_conf = clamp(video_conf)
    a_conf = clamp(audio_conf)
    l_conf = clamp(lip_sync_conf)
    
    agg_conf = (w.video * v_conf) + (w.audio * a_conf) + (w.lip_sync * l_conf)
    return clamp(agg_conf)


def calibrate_fusion_result(
    video: VideoPrediction,
    audio: AudioPrediction,
    lip_sync: LipSyncPrediction,
    fused_score: float,
    weights: Optional[ScoreWeights] = None
) -> Tuple[float, float]:
    """
    Performs consensus-based score and confidence calibration.
    
    If predictions conflict (high variance between video/audio/lip sync scores),
    the aggregated confidence is penalized to reflect modality contradiction.
    If scores agree, confidence is reinforced.
    
    Args:
        video (VideoPrediction): Video prediction data.
        audio (AudioPrediction): Audio prediction data.
        lip_sync (LipSyncPrediction): Lip sync prediction data.
        fused_score (float): Weighted fused score computed.
        weights (Optional[ScoreWeights]): Configurable weights.
        
    Returns:
        Tuple[float, float]: Calibrated (fused_score, calibrated_confidence).
    """
    w = weights or get_default_config().weights
    
    # 1. Fetch raw aggregated confidence
    raw_conf = aggregate_confidence(video.confidence, audio.confidence, lip_sync.confidence, w)
    
    # 2. Compute weighted variance representing consensus level
    # Score diffs
    diff_v = video.score - fused_score
    diff_a = audio.score - fused_score
    diff_l = lip_sync.score - fused_score
    
    weighted_variance = (
        w.video * (diff_v ** 2) +
        w.audio * (diff_a ** 2) +
        w.lip_sync * (diff_l ** 2)
    )
    
    # 3. Calculate consensus penalty based on variance
    # Maximum possible variance with weights summing to 1 is 0.25 (e.g. 0.5*(-0.5)^2 + 0.5*(0.5)^2)
    # We map variance to a penalty scale. Max penalty coefficient of 2.0 scales variance to a [0.0, 0.5] penalty range.
    penalty = 2.0 * weighted_variance
    
    calibrated_confidence = raw_conf * (1.0 - penalty)
    
    # 4. Single-modality threat calibration:
    # If ANY modality (video, audio, or lip_sync) detects a strong deepfake signal (score > 0.70),
    # calibrate the fused score upward so that single-channel attacks (e.g. voice cloning or face swaps)
    # are correctly flagged as Deepfakes instead of being averaged down to Authentic.
    calibrated_score = fused_score
    max_modal_score = max(video.score, audio.score, lip_sync.score)
    if max_modal_score > 0.70:
        calibrated_score = max(fused_score, 0.40 * fused_score + 0.60 * max_modal_score)
        logger.debug(
            f"High-confidence modality threat detected ({max_modal_score:.4f}). "
            f"Calibrated score from {fused_score:.4f} to {calibrated_score:.4f}"
        )
        
    return clamp(calibrated_score), clamp(calibrated_confidence)


if __name__ == "__main__":
    print("Executing self-test for fusion/scoring.py...")
    try:
        # 1. Test clamp
        assert clamp(1.2) == 1.0
        assert clamp(-0.5) == 0.0
        assert clamp(0.5) == 0.5
        
        # 2. Test score normalization
        assert normalize_score(1.5) == 1.0
        assert normalize_score(-0.1) == 0.0
        
        # 3. Test weighted fusion (default weights: 0.5, 0.3, 0.2)
        score = compute_weighted_score(0.80, 0.40, 0.10)
        expected = 0.50 * 0.80 + 0.30 * 0.40 + 0.20 * 0.10
        print(f"Fused score: {score:.4f} (Expected: {expected:.4f})")
        assert abs(score - expected) < 1e-6
        
        # 4. Test calibration (Agreement vs Disagreement)
        # Agreement scenario (high consensus)
        v_pred1 = VideoPrediction(is_fake=True, score=0.90, confidence=0.95)
        a_pred1 = AudioPrediction(is_fake=True, score=0.92, confidence=0.90)
        l_pred1 = LipSyncPrediction(is_fake=True, score=0.88, confidence=0.85)
        
        fused1 = compute_weighted_score(v_pred1.score, a_pred1.score, l_pred1.score)
        cal_score1, cal_conf1 = calibrate_fusion_result(v_pred1, a_pred1, l_pred1, fused1)
        print(f"Agreement - Fused Score: {fused1:.4f}, Calibrated Score: {cal_score1:.4f}, Calibrated Conf: {cal_conf1:.4f}")
        # With zero/low variance, calibrated confidence should be close to raw aggregated confidence
        raw_conf1 = aggregate_confidence(v_pred1.confidence, a_pred1.confidence, l_pred1.confidence)
        assert cal_conf1 > 0.80
        assert abs(cal_conf1 - raw_conf1) < 0.05
        
        # Disagreement scenario (low consensus / conflict)
        v_pred2 = VideoPrediction(is_fake=True, score=0.95, confidence=0.95)
        a_pred2 = AudioPrediction(is_fake=False, score=0.05, confidence=0.90)
        l_pred2 = LipSyncPrediction(is_fake=False, score=0.10, confidence=0.80)
        
        fused2 = compute_weighted_score(v_pred2.score, a_pred2.score, l_pred2.score)
        cal_score2, cal_conf2 = calibrate_fusion_result(v_pred2, a_pred2, l_pred2, fused2)
        print(f"Disagreement - Fused Score: {fused2:.4f}, Calibrated Score: {cal_score2:.4f}, Calibrated Conf: {cal_conf2:.4f}")
        # Calibrated confidence should be significantly lower than raw aggregated confidence due to high variance
        raw_conf2 = aggregate_confidence(v_pred2.confidence, a_pred2.confidence, l_pred2.confidence)
        print(f"Disagreement raw confidence: {raw_conf2:.4f}, calibrated: {cal_conf2:.4f}")
        assert cal_conf2 < raw_conf2 - 0.20  # Significant drop due to modal variance
        
        print("All self-tests completed successfully.")
    except Exception as e:
        import sys
        print(f"Self-test failed with error: {e}", file=sys.stderr)
        sys.exit(1)
