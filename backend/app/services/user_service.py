"""
user_service.py

Handles user profile management business logic.
Exposes current user retrieval and profile update operations.
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

from backend.app.auth.security import hash_password, verify_password
from backend.app.auth.schemas import UpdateProfileRequest, ProfileResponse
from backend.app.repositories.user_repository import UserRepository
from backend.app.repositories.audit_repository import AuditRepository
from backend.app.database.models.user import User

logger = logging.getLogger("backend.services.user")


class UserService:
    """
    Coordinates all user profile read and update workflows.
    Never exposes password hashes in response payloads.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.user_repo = UserRepository(db)
        self.audit_repo = AuditRepository(db)

    async def get_current_user(self, user_uuid: str) -> User:
        """
        Fetch the currently authenticated user by their JWT sub (UUID).

        Raises:
            HTTPException 404: If the user no longer exists.
            HTTPException 403: If the account has been deactivated.
        """
        user = await self.user_repo.get_by_uuid(user_uuid)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User account not found."
            )
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account has been deactivated."
            )
        return user

    async def update_profile(
        self,
        user: User,
        payload: UpdateProfileRequest,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> ProfileResponse:
        """
        Apply profile updates (name and/or password change) for the current user.

        Rules:
            - Updating name requires no extra verification.
            - Updating password requires current_password to be verified first.

        Raises:
            HTTPException 400: If new_password is provided without current_password.
            HTTPException 401: If current_password does not match the stored hash.
        """
        new_hash: Optional[str] = None

        # Handle password change request
        if payload.new_password:
            if not payload.current_password:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="current_password is required when setting a new password."
                )
            if not verify_password(payload.current_password, user.password_hash):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Current password is incorrect."
                )
            new_hash = hash_password(payload.new_password)

        # Apply updates
        updated_user = await self.user_repo.update_profile(
            user_id=user.id,
            name=payload.name,
            password_hash=new_hash
        )

        # Audit log
        await self.audit_repo.log(
            action="update_profile",
            user_id=user.id,
            ip_address=ip_address,
            user_agent=user_agent
        )

        logger.info(f"Profile updated for user uuid={user.uuid}")
        return ProfileResponse.model_validate(updated_user)

    async def get_all_users(
        self,
        requesting_user: User,
        skip: int = 0,
        limit: int = 50
    ) -> list[User]:
        """
        Fetch all users. Restricted to Admin role only.

        Raises:
            HTTPException 403: If the requesting user is not an Admin.
        """
        if requesting_user.role != "Admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required."
            )
        return await self.user_repo.get_all(skip=skip, limit=limit)


if __name__ == "__main__":
    print("Executing self-test for backend/app/services/user_service.py...")
    try:
        import inspect
        methods = [m for m in dir(UserService) if not m.startswith("_")]
        required = ["get_current_user", "update_profile", "get_all_users"]
        for m in required:
            assert m in methods, f"Missing method: {m}"
            sig = inspect.signature(getattr(UserService, m))
            print(f"  Method '{m}' signature: {sig} — OK")
        print("UserService structural self-test: PASSED")
    except Exception as e:
        print(f"Self-test failed with error: {e}", file=sys.stderr)
        sys.exit(1)
