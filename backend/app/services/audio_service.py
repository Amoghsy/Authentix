"""
audio_service.py

This service extracts audio from video uploads and executes the cached AudioPredictor
ONNX inference pipeline to calculate audio deepfake probabilities.
"""

import logging
from pathlib import Path
import sys
from typing import Optional

# Ensure project root is in path for direct execution
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.app.config import Settings, get_settings
from backend.app.exceptions import AudioExtractionError, ModelInferenceError
from backend.app.models.response_models import AudioModelResponse
from ai.audio.inference.predictor import AudioPredictor
from ai.lip_sync.preprocessing.audio_extractor import AudioExtractor, AudioExtractionError as ExtractorError
from ai.fusion.schemas import AudioPrediction

logger = logging.getLogger("backend.services.audio")


class AudioService:
    """Manages audio demuxing and prediction pipeline operations."""

    def __init__(self, predictor: AudioPredictor, settings: Optional[Settings] = None):
        self.predictor = predictor
        self.settings = settings or get_settings()
        # Instantiate the FFmpeg audio extractor from lip_sync module
        self.extractor = AudioExtractor()

    async def analyze_audio(self, video_path: Path, analysis_id: str) -> AudioPrediction:
        """
        Extracts the audio track to temp/analysis_id/extracted.wav and runs Audio AI prediction.
        
        Args:
            video_path (Path): Path to the saved video file.
            analysis_id (str): Unique identifier for the analysis session.
            
        Returns:
            AudioPrediction: Prediction values formatted for the Fusion AI Engine input.
        """
        temp_dir = self.settings.TEMP_DIR / analysis_id
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        output_wav = temp_dir / "extracted.wav"
        
        logger.info(f"Extracting audio track from video to temp: {output_wav.resolve()}")
        try:
            self.extractor.extract_audio(video_path, output_wav)
        except ExtractorError as e:
            logger.warning(f"Audio extraction failed for {video_path}: {e}")
            raise AudioExtractionError(
                "Failed to extract audio track. Ensure the video contains a valid audio track.",
                detail=str(e)
            )
        except Exception as e:
            logger.error(f"Unexpected audio extraction error: {e}", exc_info=True)
            raise AudioExtractionError("System error during audio extraction.", detail=str(e))

        if not output_wav.exists() or output_wav.stat().st_size == 0:
            raise AudioExtractionError("Extracted audio track is empty or missing.")

        logger.info(f"Initiating Audio AI inference on: {output_wav.resolve()}")
        try:
            result = self.predictor.predict(output_wav)
            logger.info(f"Audio AI inference completed: Fused classification = {result['prediction']}")
            
            is_fake = (result["prediction"] == "Fake")
            score = result["fake_probability"]
            confidence = result["confidence"]
            
            metadata = {
                "real_probability": result["real_probability"],
                "latency_ms": result["latency_ms"]
            }
            
            return AudioPrediction(
                is_fake=is_fake,
                score=score,
                confidence=confidence,
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Audio model evaluation failed: {e}", exc_info=True)
            raise ModelInferenceError(f"Audio deepfake evaluation process failed: {e}")


if __name__ == "__main__":
    print("Executing self-test for backend/app/services/audio_service.py...")
    # Mock items for test execution
    class MockAudioPredictor:
        def predict(self, path):
            return {
                "prediction": "Real",
                "confidence": 0.92,
                "fake_probability": 0.15,
                "real_probability": 0.85,
                "latency_ms": 110.2
            }
            
    class MockExtractor:
        def extract_audio(self, video, out_wav):
            out_wav.parent.mkdir(parents=True, exist_ok=True)
            out_wav.write_text("dummy wave content")

    import tempfile
    import shutil
    import asyncio
    from typing import Optional

    try:
        temp_dir = Path(tempfile.mkdtemp())
        cfg = Settings(TEMP_DIR=temp_dir)
        
        service = AudioService(MockAudioPredictor(), cfg)
        service.extractor = MockExtractor()
        
        pred = asyncio.run(service.analyze_audio(Path("test.mp4"), "test-uuid"))
        print(f"Mapped Result: is_fake={pred.is_fake}, score={pred.score}, metadata={pred.metadata}")
        assert pred.is_fake is False
        assert pred.score == 0.15
        
        # Cleanup
        shutil.rmtree(temp_dir)
        print("AudioService self-test: PASSED")
    except Exception as e:
        print(f"Self-test failed with error: {e}", file=sys.stderr)
        sys.exit(1)
