"""
detection_schema.py
───────────────────
Pydantic models for the Authentix deepfake detection API.

All existing schemas are preserved for backward compatibility with the
existing route handlers.  New nested schemas support the required
hybrid output format.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


# ===========================================================================
# Request models (unchanged)
# ===========================================================================

class VideoUploadResponse(BaseModel):
    video_id: str
    path: str


class PreprocessRequest(BaseModel):
    path: str


class FeatureExtractionRequest(BaseModel):
    frames: List[Any]
    audio_path: str


class PredictionRequest(BaseModel):
    features: Dict[str, Dict[str, float]]


class FusionRequest(BaseModel):
    model_scores: Dict[str, float]
    heuristic_scores: Dict[str, float]


class ExplanationRequest(BaseModel):
    features: Dict[str, Dict[str, float]]
    scores: Dict[str, Any]


# ===========================================================================
# Feature schemas (unchanged)
# ===========================================================================

class VideoFeatures(BaseModel):
    variance: float
    flicker: float
    blur: float


class AudioFeatures(BaseModel):
    mfcc_mean: float
    pitch_var: float
    energy: float


class FeatureResponse(BaseModel):
    video: VideoFeatures
    audio: AudioFeatures
    pretrained_video_score: Optional[float] = None
    pretrained_audio_score: Optional[float] = None


# ===========================================================================
# Model prediction schema (unchanged — flat keys for route handlers)
# ===========================================================================

class ModelPredictionResponse(BaseModel):
    pretrained_video_score: float
    custom_video_score: float
    video_model_score: float
    pretrained_audio_score: float
    custom_audio_score: float
    audio_model_score: float
    model_score: float
    model_used: str


# ===========================================================================
# Heuristic schema (unchanged)
# ===========================================================================

class HeuristicResponse(BaseModel):
    video_heuristic_score: float
    audio_heuristic_score: float
    heuristic_score: float
    issues: List[str]


# ===========================================================================
# Fusion schema (unchanged — flat, for route /fusion)
# ===========================================================================

class FusionResponse(BaseModel):
    final_score: float
    status: str
    pretrained_video_score: float
    custom_video_score: float
    video_model_score: float
    pretrained_audio_score: float
    custom_audio_score: float
    audio_model_score: float
    model_score: float
    video_heuristic_score: float
    audio_heuristic_score: float
    heuristic_score: float


# ===========================================================================
# Score breakdown (updated — includes new signals)
# ===========================================================================

class ScoreBreakdown(BaseModel):
    video_model_score: float
    audio_model_score: float
    model_score: float
    video_heuristic_score: float
    audio_heuristic_score: float
    heuristic_score: float
    final_score: float


# ===========================================================================
# NEW — Signal breakdown block
# ===========================================================================

class BreakdownBlock(BaseModel):
    """Per-signal score breakdown for maximum transparency."""
    cnn: float = 0.0
    temporal: float = 0.0
    fft: float = 0.0
    deepface: float = 0.0
    landmark: float = 0.0
    lip_sync: float = 0.0


# ===========================================================================
# Explanation schema (unchanged)
# ===========================================================================

class ExplanationResponse(BaseModel):
    explanation: List[str]
    attribution: str


# ===========================================================================
# Nested blocks matching the required output specification
# ===========================================================================

class ModelPredictionBlock(BaseModel):
    """Nested block surfaced in FinalDetectionResponse.model_prediction."""
    video_model_score: float
    audio_model_score: float
    model_score: float


class HeuristicAnalysisBlock(BaseModel):
    """Nested block surfaced in FinalDetectionResponse.heuristic_analysis."""
    video_heur: float
    audio_heur: float
    heuristic_score: float


class FusionBlock(BaseModel):
    """Nested block surfaced in FinalDetectionResponse.fusion."""
    final_score: float


# ===========================================================================
# Final detection response (updated to include breakdown)
# ===========================================================================

class FinalDetectionResponse(BaseModel):
    """
    Top-level API response for POST /detect.

    Required output format:
    {
        "label":             str,
        "confidence":        float,
        "video_model_score": float,
        "audio_model_score": float,
        "model_score":       float,
        "heuristic_score":   float,
        "final_score":       float,
        "breakdown": {
            "cnn":       float,
            "temporal":  float,
            "fft":       float,
            "deepface":  float,
            "landmark":  float,
            "lip_sync":  float,
        },
        "explanation": List[str],
    }
    """
    # ── Primary fields ─────────────────────────────────────────────────────
    video_id:   str
    label:      str
    confidence: float

    video_model_score: float
    audio_model_score: float
    model_score:       float
    heuristic_score:   float
    final_score:       float

    # ── Signal breakdown ───────────────────────────────────────────────────
    breakdown: BreakdownBlock

    # ── Explanation ────────────────────────────────────────────────────────
    explanation: List[str]
    attribution: str

    # ── Required nested blocks (kept for structured consumers) ─────────────
    model_prediction:   ModelPredictionBlock
    heuristic_analysis: HeuristicAnalysisBlock
    fusion:             FusionBlock

    # ── Legacy flat fields (preserved for backward compat) ─────────────────
    scores:   ScoreBreakdown
    issues:   List[str]
    features: FeatureResponse
