"""
dependencies.py  (auth/)

FastAPI injectable dependencies for:
    - Bearer token extraction and JWT verification
    - Current user resolution from DB
    - Role-Based Access Control (RBAC) guards

Usage in route functions:
    current_user: User = Depends(get_current_user)
    admin_user:   User = Depends(require_role("Admin"))
"""

import logging
from pathlib import Path
import sys
from typing import Callable

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

# Ensure project root is in path for direct execution
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.app.auth.tokens import decode_token, ExpiredSignatureError, InvalidTokenError
from backend.app.database.session import get_db
from backend.app.database.models.user import User
from backend.app.repositories.user_repository import UserRepository

logger = logging.getLogger("backend.auth.dependencies")

# HTTP Bearer scheme - auto-extracts the Authorization: Bearer <token> header
_bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Primary FastAPI dependency that extracts the Bearer token from the
    Authorization header, verifies the JWT signature, and resolves the
    authenticated User from the database.

    Raises:
        HTTPException 401: If no token is provided, token is expired,
                           token is invalid, or user is not found / inactive.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header is missing. Please provide a Bearer token.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    token = credentials.credentials

    # Decode and validate JWT
    try:
        payload = decode_token(token)
    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access token has expired. Please refresh your session.",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid access token.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Reject refresh tokens presented as access tokens
    if payload.token_type != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type. Expected an access token.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Resolve user from DB using UUID from token sub claim
    user_repo = UserRepository(db)
    user = await user_repo.get_by_uuid(payload.sub)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User associated with this token no longer exists.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This account has been deactivated."
        )

    return user


def require_role(*allowed_roles: str) -> Callable:
    """
    RBAC role guard factory. Returns a FastAPI dependency that enforces
    that the current user has one of the specified roles.

    Usage:
        @router.get("/admin/users")
        async def list_users(user: User = Depends(require_role("Admin"))):
            ...

        @router.get("/analyses")
        async def list_own(user: User = Depends(require_role("Admin", "Researcher", "User"))):
            ...

    Raises:
        HTTPException 403: If the user's role is not in the allowed set.
    """
    async def _role_checker(
        current_user: User = Depends(get_current_user)
    ) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Access denied. Required role(s): {', '.join(allowed_roles)}. "
                    f"Your role: {current_user.role}."
                )
            )
        return current_user
    return _role_checker


def get_client_ip(request: Request) -> str:
    """
    Extracts the client IP from the request, respecting X-Forwarded-For
    proxy headers for deployments behind load balancers or reverse proxies.
    """
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def get_user_agent(request: Request) -> str:
    """Extracts the User-Agent header string from the incoming request."""
    return request.headers.get("User-Agent", "unknown")


if __name__ == "__main__":
    print("Executing self-test for backend/app/auth/dependencies.py...")
    try:
        import inspect

        # Verify get_current_user signature
        sig = inspect.signature(get_current_user)
        assert "credentials" in sig.parameters
        assert "db" in sig.parameters
        print(f"  get_current_user signature: {sig} — OK")

        # Verify require_role returns a callable
        guard = require_role("Admin", "Researcher")
        assert callable(guard), "require_role must return a callable"
        print(f"  require_role('Admin', 'Researcher') returns callable — OK")

        # Verify get_client_ip and get_user_agent are present
        assert callable(get_client_ip)
        assert callable(get_user_agent)
        print("  get_client_ip — OK")
        print("  get_user_agent — OK")

        print("All self-tests completed successfully: PASSED")
    except Exception as e:
        print(f"Self-test failed with error: {e}", file=sys.stderr)
        sys.exit(1)
