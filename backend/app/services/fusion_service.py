"""
fusion_service.py

This service coordinates multimodal data aggregation by passing Video, Audio, and Lip Sync
predictions to the Fusion Engine Predictor wrapper, yielding the unified FusionResult.
"""

import logging
from pathlib import Path
import sys

# Ensure project root is in path for direct execution
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.app.exceptions import ModelInferenceError
from ai.fusion.predictor import Predictor
from ai.fusion.schemas import VideoPrediction, AudioPrediction, LipSyncPrediction, FusionResult

logger = logging.getLogger("backend.services.fusion")


class FusionService:
    """Coordinates multimodal decision fusion and report registrations."""

    def __init__(self, predictor: Predictor):
        self.predictor = predictor

    async def fuse_predictions(
        self,
        video_pred: VideoPrediction,
        audio_pred: AudioPrediction,
        lips_pred: LipSyncPrediction,
        analysis_id: str
    ) -> FusionResult:
        """
        Runs scoring logic, calculates consensus calibrations, and triggers disk report writes.
        
        Args:
            video_pred (VideoPrediction): Mapped Video model predictions.
            audio_pred (AudioPrediction): Mapped Audio model predictions.
            lips_pred (LipSyncPrediction): Mapped Lip Sync model predictions.
            analysis_id (str): Unique identifier associated with this request session.
            
        Returns:
            FusionResult: Consolidated deepfake assessment details.
        """
        logger.info(f"Initiating Multimodal Fusion aggregation for ID: {analysis_id}")
        try:
            # Predictor.predict is synchronous (blocking) and generates report assets in backend/reports/{analysis_id}/
            result = self.predictor.predict(
                video=video_pred,
                audio=audio_pred,
                lip_sync=lips_pred,
                reference_id=analysis_id
            )
            
            logger.info(
                f"Multimodal Fusion aggregation completed. Unified decision = {result.prediction} "
                f"(confidence={result.confidence:.3f}, risk={result.risk_level})"
            )
            return result
        except Exception as e:
            logger.error(f"Multimodal Fusion processing failed: {e}", exc_info=True)
            raise ModelInferenceError(f"Decision aggregation process failed: {e}")


if __name__ == "__main__":
    print("Executing self-test for backend/app/services/fusion_service.py...")
    # Mock Predictor for testing
    class MockPredictor:
        def predict(self, video, audio, lip_sync, reference_id):
            return FusionResult(
                prediction="Deepfake",
                confidence=0.973,
                video_score=video.score,
                audio_score=audio.score,
                lip_sync_score=lip_sync.score,
                fusion_score=0.95,
                risk_level="High",
                reasoning=["Explanations"],
                timestamp="2026-08-05T15:35:00Z"
            )

    try:
        service = FusionService(MockPredictor())
        v_in = VideoPrediction(is_fake=True, score=0.96, confidence=0.95)
        a_in = AudioPrediction(is_fake=False, score=0.15, confidence=0.92)
        l_in = LipSyncPrediction(is_fake=True, score=0.85, confidence=0.80)
        
        import asyncio
        res = asyncio.run(service.fuse_predictions(v_in, a_in, l_in, "mock-analysis-123"))
        print(f"Fusion Result Mapped: prediction={res.prediction}, score={res.fusion_score}")
        assert res.prediction == "Deepfake"
        assert res.fusion_score == 0.95
        print("FusionService self-test: PASSED")
    except Exception as e:
        print(f"Self-test failed with error: {e}", file=sys.stderr)
        sys.exit(1)
