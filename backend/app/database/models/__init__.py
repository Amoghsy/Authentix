"""
__init__.py

Aggregates all ORM database model classes so that Alembic's autogenerate
migration feature can discover all mapped tables from a single import.
"""

from backend.app.database.models.user import User
from backend.app.database.models.analysis import AnalysisHistory
from backend.app.database.models.refresh_token import RefreshToken
from backend.app.database.models.audit_log import AuditLog

__all__ = ["User", "AnalysisHistory", "RefreshToken", "AuditLog"]
