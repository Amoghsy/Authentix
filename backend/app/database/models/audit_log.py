"""
audit_log.py

This module contains the SQLAlchemy ORM model mapping class for the "audit_logs"
table. It records key system events (logins, analyses, deletes) with originating
IP address and user agent strings for security monitoring.
"""

from datetime import datetime
from pathlib import Path
import sys
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

# Ensure project root is in path for direct execution
project_root = str(Path(__file__).resolve().parents[4])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.app.database.base import Base


class AuditLog(Base):
    """
    SQLAlchemy ORM model for recording system audit events for security monitoring
    and compliance tracing.
    """
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    ip_address: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)   # Supports IPv6
    user_agent: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=func.now()
    )


if __name__ == "__main__":
    print("Executing self-test for backend/app/database/models/audit_log.py...")
    try:
        log = AuditLog(
            user_id=1,
            action="login",
            ip_address="192.168.1.100",
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        print(f"Instantiated AuditLog: action={log.action}, ip={log.ip_address}")
        assert log.action == "login"
        assert log.ip_address == "192.168.1.100"
        print("All self-tests completed successfully: PASSED")
    except Exception as e:
        print(f"Self-test failed with error: {e}", file=sys.stderr)
        sys.exit(1)
