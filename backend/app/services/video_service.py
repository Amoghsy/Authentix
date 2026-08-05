"""
video_service.py

This service handles visual deepfake predictions by executing the cached VideoPredictor
ONNX inference pipeline on saved video uploads.
"""

import logging
from pathlib import Path
import sys

# Ensure project root is in path for direct execution
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.app.exceptions import ModelInferenceError
from backend.app.models.response_models import VideoModelResponse
from ai.video.inference.predictor import VideoPredictor
from ai.fusion.schemas import VideoPrediction

logger = logging.getLogger("backend.services.video")


class VideoService:
    """Interactions manager with the video prediction pipeline."""

    def __init__(self, predictor: VideoPredictor):
        self.predictor = predictor

    async def analyze_video(self, file_path: Path) -> VideoPrediction:
        """
        Runs the Video ONNX classifier session on the target video file.
        
        Args:
            file_path (Path): Path to the saved video file.
            
        Returns:
            VideoPrediction: Native data class matching the Fusion AI Engine input schemas.
        """
        logger.info(f"Initiating Video AI inference on: {file_path.resolve()}")
        try:
            # VideoPredictor.predict is synchronous (blocking), so we run it in a clean block
            result = self.predictor.predict(file_path)
            
            logger.info(f"Video AI inference completed: Fused classification = {result['prediction']}")
            
            # Map predictions to native VideoPrediction
            is_fake = (result["prediction"] == "Fake")
            score = result["fake_probability"]
            confidence = result["confidence"]
            
            # Pack details into metadata dictionary
            metadata = {
                "real_probability": result["real_probability"],
                "frames_processed": result["frames_processed"],
                "latency_ms": result["latency_ms"]
            }
            
            return VideoPrediction(
                is_fake=is_fake,
                score=score,
                confidence=confidence,
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Video model evaluation failed: {e}", exc_info=True)
            raise ModelInferenceError(f"Video deepfake evaluation process failed: {e}")


if __name__ == "__main__":
    print("Executing self-test for backend/app/services/video_service.py...")
    # Mock predictor for test execution
    class MockVideoPredictor:
        def predict(self, path):
            return {
                "prediction": "Fake",
                "confidence": 0.95,
                "fake_probability": 0.96,
                "real_probability": 0.04,
                "frames_processed": 32,
                "latency_ms": 150.5
            }
            
    try:
        service = VideoService(MockVideoPredictor())
        import asyncio
        pred = asyncio.run(service.analyze_video(Path("test.mp4")))
        print(f"Mapped Result: is_fake={pred.is_fake}, score={pred.score}, metadata={pred.metadata}")
        assert pred.is_fake is True
        assert pred.score == 0.96
        print("VideoService self-test: PASSED")
    except Exception as e:
        print(f"Self-test failed with error: {e}", file=sys.stderr)
        sys.exit(1)
