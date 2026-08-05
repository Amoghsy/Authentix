"""
response_models.py

This module contains the API output response models for the Authentix FastAPI backend.
All classes inherit from Pydantic's BaseModel for strict validation and serialization.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class VideoModelResponse(BaseModel):
    """Output values schema for Video AI predictions."""
    is_fake: bool
    score: float
    confidence: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AudioModelResponse(BaseModel):
    """Output values schema for Audio AI predictions."""
    is_fake: bool
    score: float
    confidence: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LipSyncModelResponse(BaseModel):
    """Output values schema for Lip Sync predictions."""
    is_fake: bool
    score: float
    confidence: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FusionModelResponse(BaseModel):
    """Output values schema for Fusion Engine result summaries."""
    prediction: str
    confidence: float
    video_score: float
    audio_score: float
    lip_sync_score: float
    fusion_score: float
    risk_level: str
    reasoning: List[str]
    timestamp: str
    execution_stats: Dict[str, Any] = Field(default_factory=dict)


class AnalysisResponse(BaseModel):
    """
    Consolidated response schema returned by the POST /api/analyze endpoint.
    """
    analysis_id: str
    prediction: str
    confidence: float
    risk_level: str
    processing_time_ms: float
    video: VideoModelResponse
    audio: AudioModelResponse
    lip_sync: LipSyncModelResponse
    fusion: FusionModelResponse
    reasoning: List[str]


class HealthResponse(BaseModel):
    """Response schema returned by the GET /health endpoint."""
    status: str
    device: str
    gpu_available: bool
    onnx_runtime_version: str
    models_loaded: Dict[str, bool]


class VersionResponse(BaseModel):
    """Response schema returned by the GET /api/version endpoint."""
    backend_version: str
    api_version: str
    model_versions: Dict[str, str]


if __name__ == "__main__":
    print("Executing self-test for backend/app/models/response_models.py...")
    try:
        # Create sub-models
        v_res = VideoModelResponse(is_fake=True, score=0.96, confidence=0.95)
        a_res = AudioModelResponse(is_fake=False, score=0.15, confidence=0.92)
        l_res = LipSyncModelResponse(is_fake=True, score=0.85, confidence=0.80)
        
        f_res = FusionModelResponse(
            prediction="Deepfake",
            confidence=0.973,
            video_score=0.96,
            audio_score=0.15,
            lip_sync_score=0.85,
            fusion_score=0.95,
            risk_level="High",
            reasoning=["Explanations"],
            timestamp="2026-08-05T15:30:00Z"
        )
        
        # Assemble composite response
        resp = AnalysisResponse(
            analysis_id="uuid-test-1234",
            prediction="Deepfake",
            confidence=0.973,
            risk_level="High",
            processing_time_ms=842.5,
            video=v_res,
            audio=a_res,
            lip_sync=l_res,
            fusion=f_res,
            reasoning=["Explanations"]
        )
        
        print(f"Instantiated AnalysisResponse successfully. ID: {resp.analysis_id}")
        assert resp.prediction == "Deepfake"
        assert resp.video.score == 0.96
        
        print("All self-tests completed successfully: PASSED")
    except Exception as e:
        import sys
        print(f"Self-test failed with error: {e}", file=sys.stderr)
        sys.exit(1)
