"""
schemas.py

This module contains dataclasses defining the data models used across the Fusion Engine.
It defines schemas for individual model predictions (Video, Audio, Lip Sync) and the
aggregated output (FusionResult), supporting direct dictionary and JSON serialization.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import json
from typing import List, Dict, Any, Optional


class JSONSerializableMixin:
    """
    Mixin providing standard helper methods to serialize dataclasses to dict or JSON format.
    """
    def to_dict(self) -> Dict[str, Any]:
        """
        Recursively converts the dataclass instance to a standard Python dictionary.
        """
        return asdict(self)

    def to_json(self, indent: Optional[int] = 4) -> str:
        """
        Serializes the dataclass instance to a formatted JSON string.
        """
        return json.dumps(self.to_dict(), indent=indent)


@dataclass(frozen=True)
class VideoPrediction(JSONSerializableMixin):
    """
    Prediction model output from the Video AI deepfake classifier.
    """
    is_fake: bool
    score: float                      # Probability of being fake [0.0, 1.0]
    confidence: float = 1.0           # Classifier confidence score [0.0, 1.0]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AudioPrediction(JSONSerializableMixin):
    """
    Prediction model output from the Audio AI deepfake classifier.
    """
    is_fake: bool
    score: float                      # Probability of being fake [0.0, 1.0]
    confidence: float = 1.0           # Classifier confidence score [0.0, 1.0]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LipSyncPrediction(JSONSerializableMixin):
    """
    Prediction model output from the SyncNet visual-audio match analyzer.
    """
    is_fake: bool
    score: float                      # Probability of being fake / out-of-sync [0.0, 1.0]
    confidence: float = 1.0           # Classifier confidence score [0.0, 1.0]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FusionResult(JSONSerializableMixin):
    """
    Consolidated assessment output produced by the Fusion Engine.
    """
    prediction: str                   # "Deepfake" or "Authentic"
    confidence: float                 # Aggregated confidence score
    video_score: float                # Video score consumed
    audio_score: float                # Audio score consumed
    lip_sync_score: float             # Lip Sync score consumed
    fusion_score: float               # Aggregated weighted fusion score
    risk_level: str                   # "Low", "Medium", "High", "Critical"
    reasoning: List[str]              # Human-readable explanations
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    execution_stats: Dict[str, Any] = field(default_factory=dict)


if __name__ == "__main__":
    print("Executing self-test for fusion/schemas.py...")
    try:
        # Create test inputs
        video = VideoPrediction(is_fake=True, score=0.92, confidence=0.95)
        audio = AudioPrediction(is_fake=False, score=0.15, confidence=0.88)
        lip = LipSyncPrediction(is_fake=True, score=0.80, confidence=0.90)
        
        # Verify dictionary serialization
        video_dict = video.to_dict()
        print(f"Video Dict: {video_dict}")
        assert video_dict["is_fake"] is True
        assert video_dict["score"] == 0.92
        assert video_dict["confidence"] == 0.95
        
        # Verify JSON serialization
        video_json = video.to_json()
        print(f"Video JSON:\n{video_json}")
        assert '"is_fake": true' in video_json
        
        # Create mock FusionResult
        res = FusionResult(
            prediction="Deepfake",
            confidence=0.915,
            video_score=0.92,
            audio_score=0.15,
            lip_sync_score=0.80,
            fusion_score=0.665,
            risk_level="High",
            reasoning=[
                "Visual manipulation probability is high.",
                "Detected significant audio-video synchronization mismatch."
            ]
        )
        res_json = res.to_json()
        print(f"Fusion Result JSON:\n{res_json}")
        assert '"prediction": "Deepfake"' in res_json
        assert '"risk_level": "High"' in res_json
        assert len(res.reasoning) == 2
        
        print("All self-tests completed successfully.")
    except Exception as e:
        import sys
        print(f"Self-test failed with error: {e}", file=sys.stderr)
        sys.exit(1)
