"""
history.py  (routes/)

Defines the analysis history API endpoints:
    GET    /api/history        - Retrieve the authenticated user's analysis history
    GET    /api/history/{id}   - Retrieve a single analysis record by UUID
    DELETE /api/history/{id}   - Delete a single analysis record by UUID
"""

import logging
from pathlib import Path
import sys
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

# Ensure project root is in path for direct execution
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.app.auth.dependencies import (
    get_client_ip,
    get_current_user,
    get_user_agent,
    require_role,
)
from backend.app.services.analysis_service import AnalysisService
from backend.app.database.session import get_db
from backend.app.database.models.user import User

logger = logging.getLogger("backend.routes.history")
router = APIRouter(prefix="/api/history", tags=["Analysis History"])


# ---------------------------------------------------------------------------
# GET /api/history
# ---------------------------------------------------------------------------
@router.get(
    "",
    summary="Retrieve the current user's analysis history"
)
async def get_history(
    skip: int = Query(default=0, ge=0, description="Number of records to skip"),
    limit: int = Query(default=20, ge=1, le=100, description="Max records to return"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_role("Admin", "Researcher", "User"))
) -> JSONResponse:
    """
    Returns a paginated list of all analysis records belonging to the current user.
    Admins can use GET /api/history?all=true (via get_all_history) for cross-user access.
    """
    svc = AnalysisService(db)
    records = await svc.get_user_history(
        user=current_user,
        skip=skip,
        limit=limit
    )

    return JSONResponse(content={
        "total": len(records),
        "skip": skip,
        "limit": limit,
        "records": [_serialize_record(r) for r in records]
    })


# ---------------------------------------------------------------------------
# GET /api/history/{analysis_id}
# ---------------------------------------------------------------------------
@router.get(
    "/{analysis_id}",
    summary="Retrieve a single analysis record by UUID"
)
async def get_single_analysis(
    analysis_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_role("Admin", "Researcher", "User"))
) -> JSONResponse:
    """
    Returns the full stored record for a single analysis identified by UUID.
    Users can only access their own records. Admins can access any record.
    """
    svc = AnalysisService(db)
    record = await svc.get_single(
        analysis_uuid=analysis_id,
        requesting_user=current_user
    )
    return JSONResponse(content=_serialize_record(record))


# ---------------------------------------------------------------------------
# DELETE /api/history/{analysis_id}
# ---------------------------------------------------------------------------
@router.delete(
    "/{analysis_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete a single analysis record by UUID"
)
async def delete_analysis(
    analysis_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_role("Admin", "Researcher", "User"))
) -> JSONResponse:
    """
    Permanently deletes the analysis record and its associated report files.
    Users can only delete their own records. Admins can delete any record.
    """
    svc = AnalysisService(db)
    await svc.delete_analysis(
        analysis_uuid=analysis_id,
        requesting_user=current_user,
        ip_address=get_client_ip(request),
        user_agent=get_user_agent(request)
    )
    return JSONResponse(content={
        "success": True,
        "message": f"Analysis {analysis_id} deleted successfully."
    })


# ---------------------------------------------------------------------------
# Serialization helper
# ---------------------------------------------------------------------------
def _serialize_record(record) -> dict:
    """Converts an AnalysisHistory ORM instance to a JSON-safe dictionary."""
    return {
        "analysis_uuid": record.analysis_uuid,
        "final_prediction": record.final_prediction,
        "confidence": record.confidence,
        "risk_level": record.risk_level,
        "video_score": record.video_score,
        "audio_score": record.audio_score,
        "lip_sync_score": record.lip_sync_score,
        "fusion_score": record.fusion_score,
        "video_prediction": record.video_prediction,
        "audio_prediction": record.audio_prediction,
        "video_model_version": record.video_model_version,
        "audio_model_version": record.audio_model_version,
        "sync_model_version": record.sync_model_version,
        "reasoning": record.reasoning,
        "report_path": record.report_path,
        "processing_time_ms": record.processing_time_ms,
        "original_video_path": record.original_video_path,
        "created_at": record.created_at.isoformat() if record.created_at else None,
    }


if __name__ == "__main__":
    print("Executing self-test for backend/app/routes/history.py...")
    try:
        from fastapi.routing import APIRoute

        # Inspect the router's own routes directly
        route_paths = [r.path for r in router.routes if isinstance(r, APIRoute)]
        expected = [
            "/api/history",
            "/api/history/{analysis_id}",
        ]
        for path in expected:
            assert path in route_paths, f"Missing route: {path}"
            print(f"  Route registered: {path} — OK")

        print("History router self-test: PASSED")
    except Exception as e:
        print(f"Self-test failed with error: {e}", file=sys.stderr)
        sys.exit(1)
