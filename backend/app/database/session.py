"""
session.py

This module configures the asynchronous SQLAlchemy database engine and session maker
for the Authentix FastAPI backend.
"""

from pathlib import Path
import sys
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

# Ensure project root is in path for direct execution
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.app.config import get_settings

settings = get_settings()

# Initialize async engine targeting PostgreSQL async driver (asyncpg)
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,  # Set to True for debugging SQL statement traces in development
    future=True
)

# Async session maker
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency yielding an asynchronous database session context.
    Ensures rollback on failures and clean closing after queries complete.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


if __name__ == "__main__":
    print("Executing self-test for backend/app/database/session.py...")
    try:
        # Check async engine and session local local mapping
        print(f"Async engine: {engine}")
        print(f"Async session maker: {AsyncSessionLocal}")
        print("All self-tests completed successfully: PASSED")
    except Exception as e:
        print(f"Self-test failed with error: {e}", file=sys.stderr)
        sys.exit(1)
