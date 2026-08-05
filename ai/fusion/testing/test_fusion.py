"""
test_fusion.py

This module contains comprehensive unit tests for the Authentix Multimodal Fusion Engine.
It uses standard unittest framework to assert scoring weights, calibrations, risk tiers,
explainability claims, boundary constraints, and type validations.
"""

from pathlib import Path
import sys
import unittest

# Ensure project root is in path for direct execution
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.fusion.config import get_default_config
from ai.fusion.schemas import VideoPrediction, AudioPrediction, LipSyncPrediction, FusionResult
from ai.fusion.fusion_engine import FusionEngine, FusionValidationError
from ai.fusion.predictor import predict_multimodal


class TestMultimodalFusion(unittest.TestCase):
    """
    Unit test cases mapping to production threat vectors and validation boundaries.
    """

    def setUp(self) -> None:
        """
        Initializes configuration and FusionEngine objects before each test.
        """
        self.config = get_default_config()
        self.engine = FusionEngine(self.config)

    def test_perfect_real(self) -> None:
        """
        Scenario: All modalities report low deepfake probability with high confidence.
        Expected: Authentic prediction, Low Risk level, and positive validation reasoning.
        """
        v = VideoPrediction(is_fake=False, score=0.04, confidence=0.95)
        a = AudioPrediction(is_fake=False, score=0.06, confidence=0.92)
        l = LipSyncPrediction(is_fake=False, score=0.10, confidence=0.90)
        
        result = self.engine.fuse(v, a, l)
        
        self.assertEqual(result.prediction, "Authentic")
        self.assertEqual(result.risk_level, "Low")
        self.assertGreater(result.confidence, 0.85)
        self.assertIn("Audio appears authentic.", result.reasoning)
        self.assertIn("Aggregated multimodal evidence indicates the media file is authentic.", result.reasoning)

    def test_perfect_fake(self) -> None:
        """
        Scenario: All modalities report high deepfake probability with high confidence.
        Expected: Deepfake prediction, Critical Risk level, and clear threat warnings in reasoning.
        """
        v = VideoPrediction(is_fake=True, score=0.98, confidence=0.96)
        a = AudioPrediction(is_fake=True, score=0.96, confidence=0.94)
        l = LipSyncPrediction(is_fake=True, score=0.92, confidence=0.90)
        
        result = self.engine.fuse(v, a, l)
        
        self.assertEqual(result.prediction, "Deepfake")
        self.assertEqual(result.risk_level, "Critical")
        self.assertGreater(result.confidence, 0.85)
        self.assertIn("Visual manipulation probability is high.", result.reasoning)
        self.assertIn("High confidence threat: both visual and auditory streams contain signs of tampering.", result.reasoning)

    def test_video_only_fake(self) -> None:
        """
        Scenario: Video AI reports deepfake manipulation, Audio/Lip Sync report authentic states.
        Expected: High-confidence video threat triggers calibration boost to classify final as Deepfake.
        """
        # High confidence video threat
        v = VideoPrediction(is_fake=True, score=0.97, confidence=0.95)
        a = AudioPrediction(is_fake=False, score=0.08, confidence=0.90)
        l = LipSyncPrediction(is_fake=False, score=0.12, confidence=0.88)
        
        result = self.engine.fuse(v, a, l)
        
        # Raw fused score would be 0.50*0.97 + 0.30*0.08 + 0.20*0.12 = 0.485 + 0.024 + 0.024 = 0.533
        # High-confidence video threat (score > 0.95, conf > 0.85) boosts it:
        # Fused = 0.7 * 0.533 + 0.3 * 0.97 = 0.3731 + 0.2910 = 0.6641
        self.assertEqual(result.prediction, "Deepfake")
        self.assertEqual(result.risk_level, "High")
        self.assertIn("Visual manipulation probability is high.", result.reasoning)
        self.assertIn("Modality conflict: Video indicates manipulation while Audio appears authentic.", result.reasoning)

    def test_audio_only_fake(self) -> None:
        """
        Scenario: Audio AI reports deepfake manipulation, Video/Lip Sync report authentic states.
        Expected: High-confidence audio threat triggers calibration boost to classify final as Deepfake.
        """
        # High confidence audio threat
        v = VideoPrediction(is_fake=False, score=0.05, confidence=0.92)
        a = AudioPrediction(is_fake=True, score=0.96, confidence=0.94)
        l = LipSyncPrediction(is_fake=False, score=0.15, confidence=0.85)
        
        result = self.engine.fuse(v, a, l)
        
        # Raw fused: 0.50*0.05 + 0.30*0.96 + 0.20*0.15 = 0.025 + 0.288 + 0.03 = 0.343
        # Boosted score: 0.7 * 0.343 + 0.3 * 0.96 = 0.2401 + 0.2880 = 0.5281 (classified as Deepfake)
        self.assertEqual(result.prediction, "Deepfake")
        self.assertEqual(result.risk_level, "Medium")
        self.assertIn("Audio manipulation probability is high.", result.reasoning)
        self.assertIn("Modality conflict: Audio indicates manipulation while Video appears authentic.", result.reasoning)

    def test_lip_sync_failure(self) -> None:
        """
        Scenario: Lip Sync is out-of-sync, but Video and Audio appear authentic.
        Expected: Fused score does not exceed decision threshold, but reasoning flags sync anomaly.
        """
        v = VideoPrediction(is_fake=False, score=0.10, confidence=0.95)
        a = AudioPrediction(is_fake=False, score=0.08, confidence=0.92)
        l = LipSyncPrediction(is_fake=True, score=0.85, confidence=0.80)
        
        result = self.engine.fuse(v, a, l)
        
        # Raw fused: 0.50*0.10 + 0.30*0.08 + 0.20*0.85 = 0.05 + 0.024 + 0.170 = 0.244 (Authentic, Low risk)
        self.assertEqual(result.prediction, "Authentic")
        self.assertEqual(result.risk_level, "Low")
        self.assertIn("Detected significant audio-video synchronization mismatch.", result.reasoning)

    def test_missing_inputs(self) -> None:
        """
        Scenario: Non-prediction objects or None arguments are passed to the engine.
        Expected: FusionValidationError raised.
        """
        v = VideoPrediction(is_fake=True, score=0.95, confidence=0.90)
        a = AudioPrediction(is_fake=False, score=0.10, confidence=0.85)
        
        with self.assertRaises(FusionValidationError):
            # Missing Lip Sync
            self.engine.fuse(v, a, None)  # type: ignore

        with self.assertRaises(FusionValidationError):
            # Passing raw dict instead of dataclass
            self.engine.fuse(v, a, {"score": 0.5, "confidence": 0.8})  # type: ignore

    def test_low_confidence(self) -> None:
        """
        Scenario: Low confidence detectors (e.g. 0.20) are aggregated.
        Expected: Decision confidence is heavily penalized or resolved low.
        """
        v = VideoPrediction(is_fake=True, score=0.90, confidence=0.20)
        a = AudioPrediction(is_fake=False, score=0.10, confidence=0.20)
        l = LipSyncPrediction(is_fake=False, score=0.10, confidence=0.20)
        
        result = self.engine.fuse(v, a, l)
        
        self.assertLess(result.confidence, 0.30)
        self.assertIn("Suspected visual manipulation with low detector confidence.", result.reasoning)

    def test_boundary_thresholds(self) -> None:
        """
        Scenario: Input scores are calibrated exactly at configuration boundary thresholds.
        Expected: Stable classification resolving cleanly according to boundary logic.
        """
        # Score resolves exactly to 0.30 (Low boundary threshold)
        v = VideoPrediction(is_fake=False, score=0.30, confidence=0.90)
        a = AudioPrediction(is_fake=False, score=0.30, confidence=0.90)
        l = LipSyncPrediction(is_fake=False, score=0.30, confidence=0.90)
        
        result = self.engine.fuse(v, a, l)
        # Score 0.30 is NOT < 0.30, so it maps to "Medium" risk
        self.assertEqual(result.risk_level, "Medium")
        self.assertEqual(result.prediction, "Authentic")  # 0.30 < 0.50 decision threshold


if __name__ == "__main__":
    unittest.main()
