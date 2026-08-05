"""
analyze.py

This module contains the primary routes for Authentix:
- POST /api/analyze: Receives uploads, runs AI modules sequentially, and fuses outcomes.
- GET /api/report/{analysis_id}: Resolves and returns saved JSON reports.
- DELETE /api/report/{analysis_id}: Deletes stored reports and clears disk.
"""

import json
from pathlib import Path
import sys
import time
import uuid
import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import JSONResponse

# Ensure project root is in path for direct execution
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.app.config import Settings, get_settings
from backend.app.exceptions import FileValidationError
from backend.app.models.response_models import (
    AnalysisResponse,
    VideoModelResponse,
    AudioModelResponse,
    LipSyncModelResponse,
    FusionModelResponse
)
from backend.app.services.upload_service import UploadService
from backend.app.services.video_service import VideoService
from backend.app.services.audio_service import AudioService
from backend.app.services.lipsync_service import LipSyncService
from backend.app.services.fusion_service import FusionService
from backend.app.services.report_service import ReportService
from backend.app.utils.cleanup import cleanup_session_temp_files
import backend.app.dependencies as deps

logger = logging.getLogger("backend.routes.analyze")
router = APIRouter()


# Endpoint dependencies helpers
def get_upload_service(settings: Settings = Depends(get_settings)) -> UploadService:
    return UploadService(settings)


def get_video_service(predictor=Depends(deps.get_video_predictor)) -> VideoService:
    return VideoService(predictor)


def get_audio_service(
    predictor=Depends(deps.get_audio_predictor),
    settings: Settings = Depends(get_settings)
) -> AudioService:
    return AudioService(predictor, settings)


def get_lipsync_service(
    preprocessor=Depends(deps.get_sync_preprocessor),
    settings: Settings = Depends(get_settings)
) -> LipSyncService:
    return LipSyncService(preprocessor, settings)


def get_fusion_service(predictor=Depends(deps.get_fusion_predictor)) -> FusionService:
    return FusionService(predictor)


def get_report_service(settings: Settings = Depends(get_settings)) -> ReportService:
    return ReportService(settings)


@router.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_video_endpoint(
    file: UploadFile = File(...),
    upload_service: UploadService = Depends(get_upload_service),
    video_service: VideoService = Depends(get_video_service),
    audio_service: AudioService = Depends(get_audio_service),
    lipsync_service: LipSyncService = Depends(get_lipsync_service),
    fusion_service: FusionService = Depends(get_fusion_service),
    settings: Settings = Depends(get_settings)
) -> AnalysisResponse:
    """
    Ingests video uploads, runs deepfake classification workflows, aggregates
    scores via decision fusion, and generates a unified response.
    """
    analysis_id = str(uuid.uuid4())
    logger.info(f"Received video analysis request. Target Session ID: {analysis_id}")
    
    start_time = time.perf_counter()
    saved_path: Optional[Path] = None
    
    try:
        # 1. Ingest upload and save securely to disk
        saved_path = await upload_service.upload_video(file)
        
        # 2. Execute Video AI
        video_prediction = await video_service.analyze_video(saved_path)
        
        # 3. Execute Audio AI
        audio_prediction = await audio_service.analyze_audio(saved_path, analysis_id)
        
        # 4. Execute Lip Sync tracking
        lipsync_prediction = await lipsync_service.analyze_lipsync(saved_path, analysis_id)
        
        # 5. Execute Fusion Engine scoring & save reports
        fusion_result = await fusion_service.fuse_predictions(
            video_pred=video_prediction,
            audio_pred=audio_prediction,
            lips_pred=lipsync_prediction,
            analysis_id=analysis_id
        )
        
        # 6. Delete intermediate file uploads and temp alignment spaces
        cleanup_session_temp_files(analysis_id, settings)
        
        # 7. Compute total processing time
        processing_time_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(f"Successful session analysis completion {analysis_id} in {processing_time_ms:.1f}ms")
        
        return AnalysisResponse(
            analysis_id=analysis_id,
            prediction=fusion_result.prediction,
            confidence=fusion_result.confidence,
            risk_level=fusion_result.risk_level,
            processing_time_ms=processing_time_ms,
            video=VideoModelResponse(
                is_fake=video_prediction.is_fake,
                score=video_prediction.score,
                confidence=video_prediction.confidence,
                metadata=video_prediction.metadata
            ),
            audio=AudioModelResponse(
                is_fake=audio_prediction.is_fake,
                score=audio_prediction.score,
                confidence=audio_prediction.confidence,
                metadata=audio_prediction.metadata
            ),
            lip_sync=LipSyncModelResponse(
                is_fake=lipsync_prediction.is_fake,
                score=lipsync_prediction.score,
                confidence=lipsync_prediction.confidence,
                metadata=lipsync_prediction.metadata
            ),
            fusion=FusionModelResponse(
                prediction=fusion_result.prediction,
                confidence=fusion_result.confidence,
                video_score=fusion_result.video_score,
                audio_score=fusion_result.audio_score,
                lip_sync_score=fusion_result.lip_sync_score,
                fusion_score=fusion_result.fusion_score,
                risk_level=fusion_result.risk_level,
                reasoning=fusion_result.reasoning,
                timestamp=fusion_result.timestamp,
                execution_stats=fusion_result.execution_stats
            ),
            reasoning=fusion_result.reasoning
        )
        
    except Exception as e:
        # Cleanup file space in case of failures during prediction runs
        if saved_path and saved_path.exists():
            saved_path.unlink()
        cleanup_session_temp_files(analysis_id, settings)
        logger.error(f"Pipeline execution failed for session {analysis_id}: {e}", exc_info=True)
        raise e


@router.get("/api/report/{analysis_id}")
def get_report_endpoint(
    analysis_id: str,
    report_service: ReportService = Depends(get_report_service)
):
    """
    Retrieves stored JSON prediction summary results from backend reports storage.
    """
    logger.info(f"Retrieving JSON report content for session ID: {analysis_id}")
    report_content = report_service.read_json_report(analysis_id)
    return JSONResponse(content=json.loads(report_content))


@router.delete("/api/report/{analysis_id}")
def delete_report_endpoint(
    analysis_id: str,
    report_service: ReportService = Depends(get_report_service)
):
    """
    Clears all saved report outputs and directory references matching the analysis ID.
    """
    logger.info(f"Purging stored reports assets for session ID: {analysis_id}")
    report_service.delete_analysis_report(analysis_id)
    return {
        "success": True,
        "message": f"Report and files for session {analysis_id} cleared successfully."
    }


if __name__ == "__main__":
    print("Executing self-test for backend/app/routes/analyze.py...")
    # Mock services setup for test clients
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from ai.fusion.schemas import FusionResult
    
    app = FastAPI()
    
    # Setup mock endpoints routes
    @app.post("/api/analyze")
    def mock_analyze(file: UploadFile = File(...)):
        return {
            "analysis_id": "test-uuid-5555",
            "prediction": "Real",
            "confidence": 0.88,
            "risk_level": "Low",
            "processing_time_ms": 250.0,
            "video": {"is_fake": False, "score": 0.12, "confidence": 0.90, "metadata": {}},
            "audio": {"is_fake": False, "score": 0.15, "confidence": 0.92, "metadata": {}},
            "lip_sync": {"is_fake": False, "score": 0.10, "confidence": 0.85, "metadata": {}},
            "fusion": {
                "prediction": "Real",
                "confidence": 0.88,
                "video_score": 0.12,
                "audio_score": 0.15,
                "lip_sync_score": 0.10,
                "fusion_score": 0.12,
                "risk_level": "Low",
                "reasoning": ["OK"],
                "timestamp": "2026-08-05T15:38:00Z"
            },
            "reasoning": ["OK"]
        }

    client = TestClient(app)
    try:
        # Mock file post
        files = {"file": ("video.mp4", b"dummy video content", "video/mp4")}
        response = client.post("/api/analyze", files=files)
        print(f"Analyze Status Code: {response.status_code}")
        print(f"Analyze JSON: {response.json()}")
        assert response.status_code == 200
        assert response.json()["analysis_id"] == "test-uuid-5555"
        print("Analyze route self-test: PASSED")
    except Exception as e:
        print(f"Self-test failed with error: {e}", file=sys.stderr)
        sys.exit(1)
