"""
analysis_service.py

Handles analysis history business logic.
Bridges the AI pipeline outputs and database persistence layer.
Stores every completed analysis for reproducibility and user history.
"""

import logging
from pathlib import Path
import sys
from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

# Ensure project root is in path for direct execution
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.app.repositories.analysis_repository import AnalysisRepository
from backend.app.repositories.audit_repository import AuditRepository
from backend.app.database.models.analysis import AnalysisHistory
from backend.app.database.models.user import User

logger = logging.getLogger("backend.services.analysis")


class AnalysisService:
    """
    Coordinates analysis result persistence, history retrieval,
    and per-user scoped deletions.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.analysis_repo = AnalysisRepository(db)
        self.audit_repo = AuditRepository(db)

    async def persist_analysis(
        self,
        user: User,
        analysis_uuid: str,
        final_prediction: str,
        confidence: float,
        risk_level: str,
        video_score: Optional[float] = None,
        audio_score: Optional[float] = None,
        lip_sync_score: Optional[float] = None,
        fusion_score: Optional[float] = None,
        video_prediction: Optional[str] = None,
        audio_prediction: Optional[str] = None,
        original_video_path: Optional[str] = None,
        extracted_audio_path: Optional[str] = None,
        video_model_version: Optional[str] = None,
        audio_model_version: Optional[str] = None,
        sync_model_version: Optional[str] = None,
        reasoning: Optional[list] = None,
        report_path: Optional[str] = None,
        processing_time_ms: Optional[float] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> AnalysisHistory:
        """
        Persist a completed AI analysis result to the database for the authenticated user.
        Also writes an audit log entry for traceability.

        Returns:
            The saved AnalysisHistory ORM instance.
        """
        # Wrap reasoning list in a dict for JSON column storage
        reasoning_payload = {"claims": reasoning} if reasoning else None

        record = await self.analysis_repo.create(
            user_id=user.id,
            analysis_uuid=analysis_uuid,
            final_prediction=final_prediction,
            confidence=confidence,
            risk_level=risk_level,
            video_score=video_score,
            audio_score=audio_score,
            lip_sync_score=lip_sync_score,
            fusion_score=fusion_score,
            video_prediction=video_prediction,
            audio_prediction=audio_prediction,
            original_video_path=original_video_path,
            extracted_audio_path=extracted_audio_path,
            video_model_version=video_model_version,
            audio_model_version=audio_model_version,
            sync_model_version=sync_model_version,
            reasoning=reasoning_payload,
            report_path=report_path,
            processing_time_ms=processing_time_ms
        )

        # Audit log the analysis event
        await self.audit_repo.log(
            action="analyze_video",
            user_id=user.id,
            ip_address=ip_address,
            user_agent=user_agent
        )

        logger.info(
            f"Analysis {analysis_uuid} persisted for user uuid={user.uuid}: "
            f"{final_prediction} (confidence={confidence:.2f})"
        )
        return record

    async def get_user_history(
        self,
        user: User,
        skip: int = 0,
        limit: int = 50
    ) -> list[AnalysisHistory]:
        """
        Retrieve the paginated analysis history for the authenticated user.
        Admins can view all analyses using get_all_history().
        """
        return await self.analysis_repo.get_by_user(
            user_id=user.id,
            skip=skip,
            limit=limit
        )

    async def get_single(
        self,
        analysis_uuid: str,
        requesting_user: User
    ) -> AnalysisHistory:
        """
        Retrieve a single analysis record by UUID, enforcing ownership.

        Raises:
            HTTPException 404: If the record doesn't exist.
            HTTPException 403: If the requesting user doesn't own the record
                               (unless Admin).
        """
        record = await self.analysis_repo.get_by_uuid(analysis_uuid)
        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Analysis {analysis_uuid} not found."
            )
        # Ownership check: non-admins can only access their own records
        if requesting_user.role != "Admin" and record.user_id != requesting_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to access this analysis."
            )
        return record

    async def delete_analysis(
        self,
        analysis_uuid: str,
        requesting_user: User,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> None:
        """
        Delete an analysis record, enforcing ownership.

        Raises:
            HTTPException 404: If the record doesn't exist or ownership check fails.
        """
        deleted = await self.analysis_repo.delete_by_uuid(
            analysis_uuid=analysis_uuid,
            user_id=requesting_user.id
        )
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Analysis {analysis_uuid} not found or access denied."
            )

        await self.audit_repo.log(
            action="delete_analysis",
            user_id=requesting_user.id,
            ip_address=ip_address,
            user_agent=user_agent
        )
        logger.info(
            f"Analysis {analysis_uuid} deleted by user uuid={requesting_user.uuid}"
        )

    async def get_all_history(
        self,
        requesting_user: User,
        skip: int = 0,
        limit: int = 50
    ) -> list[AnalysisHistory]:
        """
        Retrieve all analysis records across all users. Admin-only.

        Raises:
            HTTPException 403: If the requesting user is not an Admin.
        """
        if requesting_user.role != "Admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required to view all analyses."
            )
        return await self.analysis_repo.get_all(skip=skip, limit=limit)


if __name__ == "__main__":
    print("Executing self-test for backend/app/services/analysis_service.py...")
    try:
        import inspect
        methods = [m for m in dir(AnalysisService) if not m.startswith("_")]
        required = [
            "persist_analysis", "get_user_history",
            "get_single", "delete_analysis", "get_all_history"
        ]
        for m in required:
            assert m in methods, f"Missing method: {m}"
            sig = inspect.signature(getattr(AnalysisService, m))
            print(f"  Method '{m}' signature: {sig} — OK")
        print("AnalysisService structural self-test: PASSED")
    except Exception as e:
        print(f"Self-test failed with error: {e}", file=sys.stderr)
        sys.exit(1)
