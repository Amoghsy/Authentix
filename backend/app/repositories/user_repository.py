"""
user_repository.py

Data access abstraction for all User table operations.
All database queries for users are isolated here and never leak into services.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

# Ensure project root is in path for direct execution
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.app.database.models.user import User

logger = logging.getLogger("backend.repositories.user")


class UserRepository:
    """
    Encapsulates all database access operations for the User entity.
    Accepts an AsyncSession injected per-request via FastAPI dependency injection.
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_id(self, user_id: int) -> Optional[User]:
        """Fetch a user by their integer primary key."""
        result = await self.db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    async def get_by_uuid(self, uuid: str) -> Optional[User]:
        """Fetch a user by their string UUID (used in JWT sub claim)."""
        result = await self.db.execute(select(User).where(User.uuid == uuid))
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> Optional[User]:
        """Fetch a user by email address. Used during login and registration checks."""
        result = await self.db.execute(
            select(User).where(User.email == email.lower().strip())
        )
        return result.scalar_one_or_none()

    async def create(
        self,
        email: str,
        password_hash: str,
        name: Optional[str] = None,
        role: str = "User"
    ) -> User:
        """Create and persist a new user record."""
        import uuid as uuid_lib
        user = User(
            uuid=str(uuid_lib.uuid4()),
            email=email.lower().strip(),
            password_hash=password_hash,
            name=name,
            role=role,
            is_active=True,
            is_verified=False
        )
        self.db.add(user)
        await self.db.flush()   # Flush to get the generated ID without committing
        await self.db.refresh(user)
        logger.info(f"Created new user: {user.email} (uuid={user.uuid})")
        return user

    async def update_last_login(self, user_id: int) -> None:
        """Stamp the last_login timestamp to the current UTC time."""
        await self.db.execute(
            update(User)
            .where(User.id == user_id)
            .values(last_login=datetime.now(timezone.utc))
        )

    async def update_profile(
        self,
        user_id: int,
        name: Optional[str] = None,
        password_hash: Optional[str] = None
    ) -> Optional[User]:
        """Update mutable profile fields for a user."""
        values: dict = {"updated_at": datetime.now(timezone.utc)}
        if name is not None:
            values["name"] = name
        if password_hash is not None:
            values["password_hash"] = password_hash

        await self.db.execute(
            update(User).where(User.id == user_id).values(**values)
        )
        return await self.get_by_id(user_id)

    async def deactivate(self, user_id: int) -> None:
        """Soft-delete a user by setting is_active=False."""
        await self.db.execute(
            update(User).where(User.id == user_id).values(is_active=False)
        )
        logger.info(f"Deactivated user id={user_id}")

    async def get_all(self, skip: int = 0, limit: int = 50) -> list[User]:
        """Fetch a paginated list of all users. Admin-only operation."""
        result = await self.db.execute(
            select(User).offset(skip).limit(limit).order_by(User.created_at.desc())
        )
        return list(result.scalars().all())


if __name__ == "__main__":
    print("Executing self-test for backend/app/repositories/user_repository.py...")
    try:
        # Structural test: check method signatures without a live DB connection
        import inspect
        repo_methods = [m for m in dir(UserRepository) if not m.startswith("_")]
        required = ["get_by_id", "get_by_uuid", "get_by_email", "create",
                    "update_last_login", "update_profile", "deactivate", "get_all"]
        for method in required:
            assert method in repo_methods, f"Missing method: {method}"
            sig = inspect.signature(getattr(UserRepository, method))
            print(f"  Method '{method}' signature: {sig} — OK")
        print("UserRepository structural self-test: PASSED")
    except Exception as e:
        print(f"Self-test failed with error: {e}", file=sys.stderr)
        sys.exit(1)
