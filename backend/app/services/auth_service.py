"""
auth_service.py

Handles authentication business logic: registration, login, token refresh, and logout.
All password hashing, JWT minting, and token revocation happen here.
This service is completely decoupled from the AI inference pipeline.
"""

import logging
from datetime import datetime, timezone
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
from backend.app.auth.tokens import (
    create_access_token,
    create_refresh_token,
    decode_token,
    ExpiredSignatureError,
    InvalidTokenError
)
from backend.app.auth.schemas import RegisterRequest, LoginRequest, TokenResponse
from backend.app.repositories.user_repository import UserRepository
from backend.app.repositories.audit_repository import AuditRepository
from backend.app.database.models.refresh_token import RefreshToken
from sqlalchemy import select, update

logger = logging.getLogger("backend.services.auth")


class AuthService:
    """
    Coordinates all authentication workflows:
    registration, credential validation, JWT issuance, refresh, and revocation.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.user_repo = UserRepository(db)
        self.audit_repo = AuditRepository(db)

    async def register(
        self,
        payload: RegisterRequest,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> TokenResponse:
        """
        Register a new user account, hash the password, persist to DB,
        and immediately return a token pair on success.

        Raises:
            HTTPException 409: If the email address is already registered.
        """
        # 1. Check for duplicate email
        existing = await self.user_repo.get_by_email(payload.email)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An account with this email address already exists."
            )

        # 2. Hash password and create user record
        pw_hash = hash_password(payload.password)
        user = await self.user_repo.create(
            email=payload.email,
            password_hash=pw_hash,
            name=payload.name,
            role="User"
        )

        # 3. Issue access + refresh token pair
        access_token = create_access_token(user.uuid, user.role)
        refresh_token_str, expires_at = create_refresh_token(user.uuid, user.role)

        # 4. Persist refresh token
        await self._store_refresh_token(user.id, refresh_token_str, expires_at)

        # 5. Audit log
        await self.audit_repo.log(
            action="register",
            user_id=user.id,
            ip_address=ip_address,
            user_agent=user_agent
        )

        logger.info(f"New user registered: {user.email} (uuid={user.uuid})")
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token_str
        )

    async def login(
        self,
        payload: LoginRequest,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> TokenResponse:
        """
        Validate credentials, stamp last_login, and return a fresh token pair.

        Raises:
            HTTPException 401: If credentials are invalid or account is inactive.
        """
        # 1. Look up user by email
        user = await self.user_repo.get_by_email(payload.email)
        if not user or not verify_password(payload.password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password.",
                headers={"WWW-Authenticate": "Bearer"}
            )

        # 2. Enforce active account check
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="This account has been deactivated.",
                headers={"WWW-Authenticate": "Bearer"}
            )

        # 3. Issue token pair
        access_token = create_access_token(user.uuid, user.role)
        refresh_token_str, expires_at = create_refresh_token(user.uuid, user.role)

        # 4. Persist refresh token and update last_login
        await self._store_refresh_token(user.id, refresh_token_str, expires_at)
        await self.user_repo.update_last_login(user.id)

        # 5. Audit log
        await self.audit_repo.log(
            action="login",
            user_id=user.id,
            ip_address=ip_address,
            user_agent=user_agent
        )

        logger.info(f"User logged in: {user.email}")
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token_str
        )

    async def refresh(self, refresh_token_str: str) -> TokenResponse:
        """
        Validate a refresh token, revoke it, and issue a new token pair (rotation).

        Raises:
            HTTPException 401: If the refresh token is expired, revoked, or invalid.
        """
        # 1. Decode and validate JWT signature
        try:
            payload = decode_token(refresh_token_str)
        except (ExpiredSignatureError, InvalidTokenError):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token is expired or invalid.",
                headers={"WWW-Authenticate": "Bearer"}
            )

        if payload.token_type != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type. Expected a refresh token."
            )

        # 2. Check token exists in DB and is not revoked
        result = await self.db.execute(
            select(RefreshToken).where(
                RefreshToken.token == refresh_token_str,
                RefreshToken.revoked == False  # noqa: E712
            )
        )
        db_token = result.scalar_one_or_none()
        if not db_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token has been revoked or does not exist."
            )

        # 3. Revoke the used token (rotation)
        await self.db.execute(
            update(RefreshToken)
            .where(RefreshToken.id == db_token.id)
            .values(revoked=True)
        )

        # 4. Fetch user and issue new pair
        user = await self.user_repo.get_by_uuid(payload.sub)
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account not found or inactive."
            )

        new_access = create_access_token(user.uuid, user.role)
        new_refresh, new_expires = create_refresh_token(user.uuid, user.role)
        await self._store_refresh_token(user.id, new_refresh, new_expires)

        logger.info(f"Token rotated for user uuid={user.uuid}")
        return TokenResponse(access_token=new_access, refresh_token=new_refresh)

    async def logout(self, refresh_token_str: str) -> None:
        """
        Revoke the provided refresh token to invalidate the session.
        """
        await self.db.execute(
            update(RefreshToken)
            .where(RefreshToken.token == refresh_token_str)
            .values(revoked=True)
        )
        logger.info("Refresh token revoked on logout.")

    async def _store_refresh_token(
        self,
        user_id: int,
        token: str,
        expires_at: datetime
    ) -> None:
        """Persist a new refresh token record to the database."""
        db_token = RefreshToken(
            user_id=user_id,
            token=token,
            expires_at=expires_at,
            revoked=False
        )
        self.db.add(db_token)
        await self.db.flush()


if __name__ == "__main__":
    print("Executing self-test for backend/app/services/auth_service.py...")
    try:
        import inspect
        methods = [m for m in dir(AuthService) if not m.startswith("_")]
        required = ["register", "login", "refresh", "logout"]
        for m in required:
            assert m in methods, f"Missing method: {m}"
            sig = inspect.signature(getattr(AuthService, m))
            print(f"  Method '{m}' signature: {sig} — OK")
        print("AuthService structural self-test: PASSED")
    except Exception as e:
        print(f"Self-test failed with error: {e}", file=sys.stderr)
        sys.exit(1)
