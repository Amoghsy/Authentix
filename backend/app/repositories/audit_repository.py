"""
audit_repository.py

Data access abstraction for all AuditLog table operations.
Handles creation and admin-level retrieval of system audit events.
"""

import logging
from pathlib import Path
import sys
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# Ensure project root is in path for direct execution
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.app.database.models.audit_log import AuditLog

logger = logging.getLogger("backend.repositories.audit")


class AuditRepository:
    """
    Encapsulates all database access operations for the AuditLog entity.
    Accepts an AsyncSession injected per-request via FastAPI dependency injection.
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def log(
        self,
        action: str,
        user_id: Optional[int] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> AuditLog:
        """
        Persist a new audit log event to the database.

        Args:
            action:     Short action label (e.g. 'login', 'analyze_video', 'delete_report').
            user_id:    The ID of the user who triggered the action. Nullable for anon events.
            ip_address: The originating request IP address.
            user_agent: The originating request User-Agent header string.

        Returns:
            The saved AuditLog ORM instance.
        """
        entry = AuditLog(
            action=action,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent
        )
        self.db.add(entry)
        await self.db.flush()
        logger.debug(
            f"Audit: action='{action}' user_id={user_id} ip={ip_address}"
        )
        return entry

    async def get_by_user(
        self,
        user_id: int,
        skip: int = 0,
        limit: int = 100
    ) -> list[AuditLog]:
        """Fetch audit log entries for a specific user, paginated. Admin-only."""
        result = await self.db.execute(
            select(AuditLog)
            .where(AuditLog.user_id == user_id)
            .order_by(AuditLog.timestamp.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_all(self, skip: int = 0, limit: int = 100) -> list[AuditLog]:
        """Fetch all audit log entries across all users, paginated. Admin-only."""
        result = await self.db.execute(
            select(AuditLog)
            .order_by(AuditLog.timestamp.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())


if __name__ == "__main__":
    print("Executing self-test for backend/app/repositories/audit_repository.py...")
    try:
        import inspect
        repo_methods = [m for m in dir(AuditRepository) if not m.startswith("_")]
        required = ["log", "get_by_user", "get_all"]
        for method in required:
            assert method in repo_methods, f"Missing method: {method}"
            sig = inspect.signature(getattr(AuditRepository, method))
            print(f"  Method '{method}' signature: {sig} — OK")
        print("AuditRepository structural self-test: PASSED")
    except Exception as e:
        print(f"Self-test failed with error: {e}", file=sys.stderr)
        sys.exit(1)
