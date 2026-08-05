"""
router.py  (auth/)

Defines all authentication API endpoints:
    POST   /api/auth/register  - Create a new user account
    POST   /api/auth/login     - Authenticate and receive tokens
    POST   /api/auth/refresh   - Rotate refresh token for a new access token
    POST   /api/auth/logout    - Revoke the current refresh token
    GET    /api/auth/me        - Get current user profile
    PUT    /api/auth/profile   - Update current user profile / password
"""

import logging
from pathlib import Path
import sys

from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

# Ensure project root is in path for direct execution
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.app.auth.schemas import (
    LoginRequest,
    MessageResponse,
    ProfileResponse,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UpdateProfileRequest,
    UserResponse,
)
from backend.app.auth.dependencies import (
    get_client_ip,
    get_current_user,
    get_user_agent,
)
from backend.app.services.auth_service import AuthService
from backend.app.services.user_service import UserService
from backend.app.database.session import get_db
from backend.app.database.models.user import User

logger = logging.getLogger("backend.auth.router")
router = APIRouter(prefix="/api/auth", tags=["Authentication"])


# ---------------------------------------------------------------------------
# POST /api/auth/register
# ---------------------------------------------------------------------------
@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account"
)
async def register(
    payload: RegisterRequest,
    request: Request,
    db: AsyncSession = Depends(get_db)
) -> TokenResponse:
    """
    Creates a new user account with the provided email and password.
    Returns a JWT access + refresh token pair on success.
    """
    svc = AuthService(db)
    return await svc.register(
        payload=payload,
        ip_address=get_client_ip(request),
        user_agent=get_user_agent(request)
    )


# ---------------------------------------------------------------------------
# POST /api/auth/login
# ---------------------------------------------------------------------------
@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login with email and password"
)
async def login(
    payload: LoginRequest,
    request: Request,
    db: AsyncSession = Depends(get_db)
) -> TokenResponse:
    """
    Authenticates a user by email and password.
    Returns a JWT access + refresh token pair on success.
    """
    svc = AuthService(db)
    return await svc.login(
        payload=payload,
        ip_address=get_client_ip(request),
        user_agent=get_user_agent(request)
    )


# ---------------------------------------------------------------------------
# POST /api/auth/refresh
# ---------------------------------------------------------------------------
@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Rotate refresh token for a new access token"
)
async def refresh_token(
    payload: RefreshRequest,
    db: AsyncSession = Depends(get_db)
) -> TokenResponse:
    """
    Validates the provided refresh token, revokes it (rotation),
    and issues a fresh access + refresh token pair.
    """
    svc = AuthService(db)
    return await svc.refresh(payload.refresh_token)


# ---------------------------------------------------------------------------
# POST /api/auth/logout
# ---------------------------------------------------------------------------
@router.post(
    "/logout",
    response_model=MessageResponse,
    summary="Revoke the current session refresh token"
)
async def logout(
    payload: RefreshRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> MessageResponse:
    """
    Revokes the provided refresh token, invalidating the current session.
    Requires a valid access token to confirm identity before logout.
    """
    svc = AuthService(db)
    await svc.logout(payload.refresh_token)
    logger.info(f"User logged out: uuid={current_user.uuid}")
    return MessageResponse(message="Logged out successfully.")


# ---------------------------------------------------------------------------
# GET /api/auth/me
# ---------------------------------------------------------------------------
@router.get(
    "/me",
    response_model=ProfileResponse,
    summary="Get the current authenticated user's profile"
)
async def get_me(
    current_user: User = Depends(get_current_user)
) -> ProfileResponse:
    """
    Returns the full profile of the currently authenticated user.
    """
    return ProfileResponse.model_validate(current_user)


# ---------------------------------------------------------------------------
# PUT /api/auth/profile
# ---------------------------------------------------------------------------
@router.put(
    "/profile",
    response_model=ProfileResponse,
    summary="Update the current user's profile or password"
)
async def update_profile(
    payload: UpdateProfileRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> ProfileResponse:
    """
    Updates the current user's display name and/or password.
    Password changes require the current_password field to be provided.
    """
    svc = UserService(db)
    return await svc.update_profile(
        user=current_user,
        payload=payload,
        ip_address=get_client_ip(request),
        user_agent=get_user_agent(request)
    )


if __name__ == "__main__":
    print("Executing self-test for backend/app/auth/router.py...")
    try:
        from fastapi.routing import APIRoute

        # Inspect the router's own routes directly
        route_paths = [r.path for r in router.routes if isinstance(r, APIRoute)]
        expected = [
            "/api/auth/register",
            "/api/auth/login",
            "/api/auth/refresh",
            "/api/auth/logout",
            "/api/auth/me",
            "/api/auth/profile"
        ]
        for path in expected:
            assert path in route_paths, f"Missing route: {path}"
            print(f"  Route registered: {path} — OK")

        print("Auth router self-test: PASSED")
    except Exception as e:
        print(f"Self-test failed with error: {e}", file=sys.stderr)
        sys.exit(1)
