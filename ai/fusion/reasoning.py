"""
reasoning.py

This module contains the rule-based explainability logic for the Authentix Fusion Engine.
It translates numeric predictions, confidence boundaries, and modal discrepancies into
structured natural language sentences explaining the engine's final decision.
"""

import logging
from pathlib import Path
import sys
from typing import List, Optional

# Ensure project root is in path for direct execution
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.fusion.config import FusionConfig, get_default_config
from ai.fusion.schemas import VideoPrediction, AudioPrediction, LipSyncPrediction

logger = logging.getLogger(__name__)


def generate_reasoning(
    video: VideoPrediction,
    audio: AudioPrediction,
    lip_sync: LipSyncPrediction,
    fused_score: float,
    config: Optional[FusionConfig] = None
) -> List[str]:
    """
    Evaluates individual model attributes and returns a collection of human-readable
    explanations for the aggregated fusion decision.
    
    Args:
        video (VideoPrediction): Video model prediction features.
        audio (AudioPrediction): Audio model prediction features.
        lip_sync (LipSyncPrediction): Lip sync model prediction features.
        fused_score (float): Computed joint deepfake probability score.
        config (Optional[FusionConfig]): Configuration object containing decision thresholds.
        
    Returns:
        List[str]: Chronological list of textual claims.
    """
    cfg = config or get_default_config()
    threshold = cfg.thresholds.decision_threshold
    
    reasons: List[str] = []
    
    # 1. Video Modality Rules
    if video.score >= threshold:
        if video.confidence >= 0.75:
            reasons.append("Visual manipulation probability is high.")
        else:
            reasons.append("Suspected visual manipulation with low detector confidence.")
    else:
        if video.confidence >= 0.75:
            reasons.append("Video appears authentic (high confidence).")
        else:
            reasons.append("Video appears authentic with low detector confidence.")
            
    # 2. Audio Modality Rules
    if audio.score >= threshold:
        if audio.confidence >= 0.75:
            reasons.append("Audio manipulation probability is high.")
        else:
            reasons.append("Suspected audio manipulation with low detector confidence.")
    else:
        if audio.confidence >= 0.75:
            reasons.append("Audio appears authentic.")
        else:
            reasons.append("Audio appears authentic with low detector confidence.")
            
    # 3. Lip Sync Modality Rules
    # Lip Sync score indicates lip-sync mismatch probability (higher score = out of sync = deepfake)
    if lip_sync.score >= threshold:
        reasons.append("Detected significant audio-video synchronization mismatch.")
    else:
        reasons.append("Audio and visual streams appear synchronized.")
        
    # 4. Modality Contradiction/Conflict Rules
    # Check if Video and Audio models directly disagree
    video_fake = video.score >= threshold
    audio_fake = audio.score >= threshold
    if video_fake != audio_fake:
        if video_fake:
            reasons.append("Modality conflict: Video indicates manipulation while Audio appears authentic.")
        else:
            reasons.append("Modality conflict: Audio indicates manipulation while Video appears authentic.")
            
    # 5. Combined Engine Summary Rule
    is_fused_fake = fused_score >= threshold
    if is_fused_fake:
        # Check if the fusion was triggered by a specific single high-threat channel
        if not video_fake and not audio_fake and lip_sync.score >= threshold:
            reasons.append("Aggregated deepfake risk is driven by lip-sync synchronization anomalies.")
        elif video_fake and audio_fake:
            reasons.append("High confidence threat: both visual and auditory streams contain signs of tampering.")
    else:
        reasons.append("Aggregated multimodal evidence indicates the media file is authentic.")
        
    return reasons


if __name__ == "__main__":
    print("Executing self-test for fusion/reasoning.py...")
    try:
        # Create test inputs simulating a high-risk multimodal threat
        video_pred = VideoPrediction(is_fake=True, score=0.96, confidence=0.92)
        audio_pred = AudioPrediction(is_fake=False, score=0.10, confidence=0.88)
        lip_pred = LipSyncPrediction(is_fake=True, score=0.85, confidence=0.80)
        
        reasons = generate_reasoning(
            video=video_pred,
            audio=audio_pred,
            lip_sync=lip_pred,
            fused_score=0.68  # Fused score indicating deepfake
        )
        
        print("Generated Reasoning Statements:")
        for r in reasons:
            print(f" - {r}")
            
        # Assert expected claims are present
        assert "Visual manipulation probability is high." in reasons
        assert "Audio appears authentic." in reasons
        assert "Detected significant audio-video synchronization mismatch." in reasons
        assert "Modality conflict: Video indicates manipulation while Audio appears authentic." in reasons
        assert len(reasons) > 0
        
        print("All self-tests completed successfully.")
    except Exception as e:
        import sys
        print(f"Self-test failed with error: {e}", file=sys.stderr)
        sys.exit(1)
