"""
analysis_repository.py

Data access abstraction for all AnalysisHistory table operations.
Handles creation, retrieval, and deletion of analysis records per user.
"""

import logging
from pathlib import Path
import sys
from typing import Optional

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

# Ensure project root is in path for direct execution
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.app.database.models.analysis import AnalysisHistory

logger = logging.getLogger("backend.repositories.analysis")


class AnalysisRepository:
    """
    Encapsulates all database access operations for the AnalysisHistory entity.
    Accepts an AsyncSession injected per-request via FastAPI dependency injection.
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(
        self,
        user_id: int,
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
        reasoning: Optional[dict] = None,
        report_path: Optional[str] = None,
        processing_time_ms: Optional[float] = None
    ) -> AnalysisHistory:
        """Persist a completed analysis result to the database."""
        record = AnalysisHistory(
            user_id=user_id,
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
            reasoning=reasoning,
            report_path=report_path,
            processing_time_ms=processing_time_ms
        )
        self.db.add(record)
        await self.db.flush()
        await self.db.refresh(record)
        logger.info(
            f"Persisted analysis {analysis_uuid}: {final_prediction} "
            f"(confidence={confidence:.2f}) for user_id={user_id}"
        )
        return record

    async def get_by_uuid(self, analysis_uuid: str) -> Optional[AnalysisHistory]:
        """Fetch a single analysis record by its unique UUID."""
        result = await self.db.execute(
            select(AnalysisHistory).where(AnalysisHistory.analysis_uuid == analysis_uuid)
        )
        return result.scalar_one_or_none()

    async def get_by_user(
        self,
        user_id: int,
        skip: int = 0,
        limit: int = 50
    ) -> list[AnalysisHistory]:
        """Fetch all analysis records belonging to a specific user, paginated."""
        result = await self.db.execute(
            select(AnalysisHistory)
            .where(AnalysisHistory.user_id == user_id)
            .order_by(AnalysisHistory.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_all(self, skip: int = 0, limit: int = 50) -> list[AnalysisHistory]:
        """Fetch all analysis records across all users. Admin-only operation."""
        result = await self.db.execute(
            select(AnalysisHistory)
            .order_by(AnalysisHistory.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def delete_by_uuid(self, analysis_uuid: str, user_id: int) -> bool:
        """
        Delete an analysis record by UUID, scoped to the owning user.
        Returns True if a row was deleted, False if not found or ownership mismatch.
        """
        result = await self.db.execute(
            delete(AnalysisHistory)
            .where(
                AnalysisHistory.analysis_uuid == analysis_uuid,
                AnalysisHistory.user_id == user_id
            )
            .returning(AnalysisHistory.id)
        )
        deleted = result.scalar_one_or_none()
        if deleted:
            logger.info(f"Deleted analysis {analysis_uuid} for user_id={user_id}")
            return True
        return False


if __name__ == "__main__":
    print("Executing self-test for backend/app/repositories/analysis_repository.py...")
    try:
        import inspect
        repo_methods = [m for m in dir(AnalysisRepository) if not m.startswith("_")]
        required = ["create", "get_by_uuid", "get_by_user", "get_all", "delete_by_uuid"]
        for method in required:
            assert method in repo_methods, f"Missing method: {method}"
            sig = inspect.signature(getattr(AnalysisRepository, method))
            print(f"  Method '{method}' signature: {sig} — OK")
        print("AnalysisRepository structural self-test: PASSED")
    except Exception as e:
        print(f"Self-test failed with error: {e}", file=sys.stderr)
        sys.exit(1)
