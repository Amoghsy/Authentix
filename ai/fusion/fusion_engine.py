"""
fusion_engine.py

This module contains the FusionEngine class, which orchestrates the multimodal decision
fusion pipeline of Authentix. It validates inputs, triggers score fusion, computes consensus
calibrations, maps risk levels, gathers reasoning logs, and returns the final FusionResult.
"""

import logging
from pathlib import Path
import sys
import time
from typing import Optional

# Ensure project root is in path for direct execution
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.fusion.config import FusionConfig, get_default_config, validate_config
from ai.fusion.schemas import VideoPrediction, AudioPrediction, LipSyncPrediction, FusionResult
from ai.fusion.scoring import (
    compute_weighted_score,
    calibrate_fusion_result,
    normalize_score,
    clamp
)
from ai.fusion.reasoning import generate_reasoning

logger = logging.getLogger(__name__)


class FusionValidationError(Exception):
    """Raised when input predictions fail structure or range validation checks."""
    pass


class FusionEngine:
    """
    Main coordinator for multimodal deepfake prediction fusion and classification.
    """

    def __init__(self, config: Optional[FusionConfig] = None):
        """
        Initializes the FusionEngine with custom or default configurations.
        """
        self.config = config or get_default_config()
        # Enforce validation rules on loaded configuration parameters
        validate_config(self.config)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _validate_inputs(
        self,
        video: VideoPrediction,
        audio: AudioPrediction,
        lip_sync: LipSyncPrediction
    ) -> None:
        """
        Validates prediction inputs for structure, data types, and value boundaries.
        """
        # Ensure correct dataclass instances
        if not isinstance(video, VideoPrediction):
            raise FusionValidationError(f"Invalid Video input type: {type(video).__name__}")
        if not isinstance(audio, AudioPrediction):
            raise FusionValidationError(f"Invalid Audio input type: {type(audio).__name__}")
        if not isinstance(lip_sync, LipSyncPrediction):
            raise FusionValidationError(f"Invalid LipSync input type: {type(lip_sync).__name__}")
            
        # Ensure scores are within [0.0, 1.0]
        for name, pred in [("Video", video), ("Audio", audio), ("LipSync", lip_sync)]:
            if not (0.0 <= pred.score <= 1.0):
                raise FusionValidationError(f"{name} prediction score must be in range [0.0, 1.0]. Provided: {pred.score}")
            if not (0.0 <= pred.confidence <= 1.0):
                raise FusionValidationError(f"{name} prediction confidence must be in range [0.0, 1.0]. Provided: {pred.confidence}")

    def _assign_risk_level(self, score: float) -> str:
        """
        Maps a numeric deepfake score to a categorical risk tier.
        
        Args:
            score (float): Normalized final score.
            
        Returns:
            str: Risk tier ("Low", "Medium", "High", "Critical").
        """
        rl = self.config.risk_levels
        
        if score < rl.low:
            return "Low"
        elif score < rl.medium:
            return "Medium"
        elif score < rl.high:
            return "High"
        else:
            return "Critical"

    def fuse(
        self,
        video: VideoPrediction,
        audio: AudioPrediction,
        lip_sync: LipSyncPrediction
    ) -> FusionResult:
        """
        Executes the multimodal decision fusion pipeline.
        
        Args:
            video (VideoPrediction): Video prediction results.
            audio (AudioPrediction): Audio prediction results.
            lip_sync (LipSyncPrediction): Lip sync match results.
            
        Returns:
            FusionResult: Unified assessment object.
            
        Raises:
            FusionValidationError: If predictions fail schema or boundary checks.
        """
        start_time = time.perf_counter()
        
        self.logger.info("Initializing decision fusion pipeline execution...")
        
        # 1. Validate inputs
        self._validate_inputs(video, audio, lip_sync)
        
        # 2. Fuse scores using weights
        raw_fused = compute_weighted_score(
            video_score=video.score,
            audio_score=audio.score,
            lip_sync_score=lip_sync.score,
            weights=self.config.weights
        )
        
        # 3. Perform consensus calibration (adjusts score & penalizes confidence on conflicts)
        calibrated_score, calibrated_confidence = calibrate_fusion_result(
            video=video,
            audio=audio,
            lip_sync=lip_sync,
            fused_score=raw_fused,
            weights=self.config.weights
        )
        
        # 4. Resolve binary classification decision
        is_deepfake = calibrated_score >= self.config.thresholds.decision_threshold
        prediction_label = "Deepfake" if is_deepfake else "Authentic"
        
        # 5. Map final risk level
        risk = self._assign_risk_level(calibrated_score)
        
        # 6. Generate human-readable explainability reasoning
        reasoning = generate_reasoning(
            video=video,
            audio=audio,
            lip_sync=lip_sync,
            fused_score=calibrated_score,
            config=self.config
        )
        
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        
        execution_stats = {
            "execution_time_ms": round(duration_ms, 3),
            "raw_fused_score": round(raw_fused, 4),
            "video_weight": self.config.weights.video,
            "audio_weight": self.config.weights.audio,
            "lip_sync_weight": self.config.weights.lip_sync,
            "decision_threshold": self.config.thresholds.decision_threshold
        }
        
        self.logger.info(
            f"Multimodal Fusion Complete. Prediction: {prediction_label}, "
            f"Score: {calibrated_score:.4f}, Risk: {risk}, Duration: {duration_ms:.2f}ms"
        )
        
        # 7. Package and return the final FusionResult dataclass
        return FusionResult(
            prediction=prediction_label,
            confidence=round(calibrated_confidence, 4),
            video_score=round(video.score, 4),
            audio_score=round(audio.score, 4),
            lip_sync_score=round(lip_sync.score, 4),
            fusion_score=round(calibrated_score, 4),
            risk_level=risk,
            reasoning=reasoning,
            execution_stats=execution_stats
        )


if __name__ == "__main__":
    print("Executing self-test for fusion/fusion_engine.py...")
    
    # Initialize config and engine handler
    cfg = get_default_config()
    engine = FusionEngine(cfg)
    
    # Create test scenario inputs
    v_fake = VideoPrediction(is_fake=True, score=0.96, confidence=0.95)
    a_real = AudioPrediction(is_fake=False, score=0.10, confidence=0.90)
    l_fake = LipSyncPrediction(is_fake=True, score=0.88, confidence=0.85)
    
    try:
        # Run fusion
        res = engine.fuse(video=v_fake, audio=a_real, lip_sync=l_fake)
        print(f"Fusion Result Output:\n{res.to_json()}")
        
        # Verify correctness
        assert res.prediction == "Deepfake"
        assert res.risk_level == "High"
        assert len(res.reasoning) > 0
        assert res.execution_stats["execution_time_ms"] >= 0.0
        
        # Test input validator boundaries
        try:
            bad_pred = VideoPrediction(is_fake=True, score=-0.2, confidence=0.9)
            engine.fuse(bad_pred, a_real, l_fake)
            print("Validation FAILED: Accepted negative score value.", file=sys.stderr)
            sys.exit(1)
        except FusionValidationError:
            print("Boundary validation check 1: PASSED")
            
        try:
            bad_pred = VideoPrediction(is_fake=True, score=0.9, confidence=1.5)
            engine.fuse(bad_pred, a_real, l_fake)
            print("Validation FAILED: Accepted out of range confidence score.", file=sys.stderr)
            sys.exit(1)
        except FusionValidationError:
            print("Boundary validation check 2: PASSED")
            
        print("All self-tests completed successfully.")
    except Exception as e:
        print(f"Self-test failed with error: {e}", file=sys.stderr)
        sys.exit(1)
