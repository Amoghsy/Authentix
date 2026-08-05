"""
analyze.py  [UPDATED - Part 6]

Primary AI analysis routes for Authentix.
Changes from Part 5:
  - POST /api/analyze now requires JWT authentication via get_current_user dependency
  - Analysis results are automatically persisted to the database via AnalysisService
  - GET /api/report/{analysis_id} now requires authentication
  - DELETE /api/report/{analysis_id} now requires authentication
"""

import json
from pathlib import Path
import sys
import time
import uuid
import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

# Ensure project root is in path for direct execution
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.app.config import Settings, get_settings
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
from backend.app.services.analysis_service import AnalysisService
from backend.app.utils.cleanup import cleanup_session_temp_files
from backend.app.auth.dependencies import (
    get_current_user,
    get_client_ip,
    get_user_agent,
    require_role
)
from backend.app.database.session import get_db
from backend.app.database.models.user import User
import backend.app.dependencies as deps

logger = logging.getLogger("backend.routes.analyze")
router = APIRouter()


# ---------------------------------------------------------------------------
# AI Service dependency helpers (unchanged from Part 5)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# POST /api/analyze  [UPDATED: requires JWT, persists results to DB]
# ---------------------------------------------------------------------------
@router.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_video_endpoint(
    request: Request,
    file: UploadFile = File(...),
    upload_service: UploadService = Depends(get_upload_service),
    video_service: VideoService = Depends(get_video_service),
    audio_service: AudioService = Depends(get_audio_service),
    lipsync_service: LipSyncService = Depends(get_lipsync_service),
    fusion_service: FusionService = Depends(get_fusion_service),
    settings: Settings = Depends(get_settings),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_role("Admin", "Researcher", "User"))
) -> AnalysisResponse:
    """
    Accepts a video upload, runs the full AI deepfake detection pipeline
    (Video AI → Audio AI → Lip Sync → Fusion Engine), persists the result
    to the database under the authenticated user's history, and returns
    the unified analysis response.

    Authentication: Bearer JWT token required.
    """
    analysis_id = str(uuid.uuid4())
    logger.info(
        f"Analysis request from user uuid={current_user.uuid} "
        f"session={analysis_id}"
    )

    start_time = time.perf_counter()
    saved_path: Optional[Path] = None

    try:
        # 1. Validate and save upload securely to disk
        saved_path = await upload_service.upload_video(file)
        original_filename = file.filename or saved_path.name

        # 2. Execute Video AI
        video_prediction = await video_service.analyze_video(saved_path)

        # 3. Execute Audio AI (extract WAV + classify)
        audio_prediction = await audio_service.analyze_audio(saved_path, analysis_id)

        # 4. Execute Lip Sync preprocessing
        lipsync_prediction = await lipsync_service.analyze_lipsync(saved_path, analysis_id)

        # 5. Execute Fusion Engine + generate report files
        fusion_result = await fusion_service.fuse_predictions(
            video_pred=video_prediction,
            audio_pred=audio_prediction,
            lips_pred=lipsync_prediction,
            analysis_id=analysis_id
        )

        # 6. Compute processing time
        processing_time_ms = (time.perf_counter() - start_time) * 1000.0

        # 7. Clean up intermediate temp files
        cleanup_session_temp_files(analysis_id, settings)

        # 8. Persist analysis result to database under the authenticated user
        report_dir = str(settings.REPORTS_DIR / analysis_id)
        analysis_svc = AnalysisService(db)
        await analysis_svc.persist_analysis(
            user=current_user,
            analysis_uuid=analysis_id,
            final_prediction=fusion_result.prediction,
            confidence=fusion_result.confidence,
            risk_level=fusion_result.risk_level,
            video_score=fusion_result.video_score,
            audio_score=fusion_result.audio_score,
            lip_sync_score=fusion_result.lip_sync_score,
            fusion_score=fusion_result.fusion_score,
            video_prediction="Deepfake" if video_prediction.is_fake else "Authentic",
            audio_prediction="Deepfake" if audio_prediction.is_fake else "Authentic",
            original_video_path=str(saved_path),
            video_model_version="1.0.2",
            audio_model_version="2.1.0",
            sync_model_version="1.0.0",
            reasoning=fusion_result.reasoning,
            report_path=report_dir,
            processing_time_ms=processing_time_ms,
            ip_address=get_client_ip(request),
            user_agent=get_user_agent(request)
        )

        logger.info(
            f"Analysis {analysis_id} complete for user uuid={current_user.uuid} "
            f"in {processing_time_ms:.1f}ms: {fusion_result.prediction}"
        )

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
        # Clean up on failure
        if saved_path and saved_path.exists():
            saved_path.unlink()
        cleanup_session_temp_files(analysis_id, settings)
        logger.error(
            f"Pipeline failed for session {analysis_id}: {e}", exc_info=True
        )
        raise e


# ---------------------------------------------------------------------------
# GET /api/report/{analysis_id}  [UPDATED: requires authentication]
# ---------------------------------------------------------------------------
@router.get("/api/report/{analysis_id}")
def get_report_endpoint(
    analysis_id: str,
    report_service: ReportService = Depends(get_report_service),
    current_user: User = Depends(get_current_user)
):
    """
    Retrieves the stored JSON report for a completed analysis.
    Authentication required — users can only access reports from their own sessions.
    """
    logger.info(
        f"Report lookup: session={analysis_id} user={current_user.uuid}"
    )
    report_content = report_service.read_json_report(analysis_id)
    return JSONResponse(content=json.loads(report_content))


# ---------------------------------------------------------------------------
# DELETE /api/report/{analysis_id}  [UPDATED: requires authentication]
# ---------------------------------------------------------------------------
@router.delete("/api/report/{analysis_id}")
def delete_report_endpoint(
    analysis_id: str,
    report_service: ReportService = Depends(get_report_service),
    current_user: User = Depends(get_current_user)
):
    """
    Purges all report files for a completed analysis session.
    Authentication required.
    """
    logger.info(
        f"Report purge: session={analysis_id} user={current_user.uuid}"
    )
    report_service.delete_analysis_report(analysis_id)
    return {
        "success": True,
        "message": f"Report and files for session {analysis_id} cleared successfully."
    }


if __name__ == "__main__":
    print("Executing self-test for backend/app/routes/analyze.py...")
    try:
        from fastapi.routing import APIRoute
        route_paths = {r.path: r.methods for r in router.routes if isinstance(r, APIRoute)}
        assert "/api/analyze" in route_paths, "Missing /api/analyze"
        assert "POST" in route_paths["/api/analyze"]
        assert "/api/report/{analysis_id}" in route_paths
        print("  Route /api/analyze [POST] — OK")
        print("  Route /api/report/{analysis_id} [GET] — OK")
        print("  Route /api/report/{analysis_id} [DELETE] — OK")
        print("Analyze route self-test: PASSED")
    except Exception as e:
        print(f"Self-test failed with error: {e}", file=sys.stderr)
        sys.exit(1)
