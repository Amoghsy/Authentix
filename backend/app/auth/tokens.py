"""
tokens.py

Provides JWT access token and refresh token creation and verification utilities
for the Authentix authentication layer.

Token Lifecycle:
    Access Token  - Short-lived (15 minutes). Contains user UUID and role.
    Refresh Token - Long-lived (7 days). Stored in database for revocation support.

Note: This module is named tokens.py (not jwt.py) to avoid shadowing the PyJWT
      standard library package.
"""

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

# Ensure project root is in path for direct execution
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Ensure PyJWT is loaded from site-packages, not a local jwt.py shadow.
# Insert the venv site-packages path ahead of any local directories.
_site_packages = str(Path(__file__).resolve().parents[4] / ".venv" / "Lib" / "site-packages")
if _site_packages not in sys.path:
    sys.path.insert(1, _site_packages)

# Remove auth directory from path temporarily to avoid jwt.py shadowing PyJWT
_auth_dir = str(Path(__file__).resolve().parent)
_removed = False
if _auth_dir in sys.path:
    sys.path.remove(_auth_dir)
    _removed = True

import jwt as _jwt_lib
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError

# Restore auth directory if it was removed
if _removed:
    sys.path.append(_auth_dir)

from backend.app.config import get_settings

logger = logging.getLogger("backend.auth.tokens")



class TokenPayload:
    """Structured token payload parsed from a decoded JWT."""

    def __init__(self, sub: str, role: str, token_type: str, exp: datetime):
        self.sub = sub                   # User UUID
        self.role = role                 # User role: Admin | Researcher | User
        self.token_type = token_type     # "access" | "refresh"
        self.exp = exp                   # Expiry datetime


def create_access_token(user_uuid: str, role: str) -> str:
    """
    Creates a short-lived JWT access token for a given user.

    Args:
        user_uuid: The unique identifier for the user.
        role: The user's assigned RBAC role.

    Returns:
        A signed JWT access token string.
    """
    settings = get_settings()
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    payload = {
        "sub": user_uuid,
        "role": role,
        "type": "access",
        "iat": now,
        "exp": expire
    }

    token = _jwt_lib.encode(
        payload,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )
    logger.debug(f"Created access token for user UUID: {user_uuid}")
    return token


def create_refresh_token(user_uuid: str, role: str) -> tuple[str, datetime]:
    """
    Creates a long-lived JWT refresh token for a given user.

    Args:
        user_uuid: The unique identifier for the user.
        role: The user's assigned RBAC role.

    Returns:
        A tuple of (signed JWT refresh token string, expiry datetime).
    """
    settings = get_settings()
    now = datetime.now(timezone.utc)
    expire = now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

    payload = {
        "sub": user_uuid,
        "role": role,
        "type": "refresh",
        "iat": now,
        "exp": expire
    }

    token = _jwt_lib.encode(
        payload,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )
    logger.debug(f"Created refresh token for user UUID: {user_uuid}")
    return token, expire


def decode_token(token: str) -> TokenPayload:
    """
    Decodes and validates a JWT token.

    Args:
        token: The JWT string to decode.

    Returns:
        A TokenPayload instance if the token is valid.

    Raises:
        ExpiredSignatureError: If the token has expired.
        InvalidTokenError: If the token is malformed or signature is invalid.
    """
    settings = get_settings()
    payload = _jwt_lib.decode(
        token,
        settings.JWT_SECRET_KEY,
        algorithms=[settings.JWT_ALGORITHM]
    )
    return TokenPayload(
        sub=payload["sub"],
        role=payload["role"],
        token_type=payload["type"],
        exp=datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
    )


if __name__ == "__main__":
    print("Executing self-test for backend/app/auth/tokens.py...")
    try:
        # Test access token creation and decoding
        access_tok = create_access_token("user-uuid-abc123", "Researcher")
        print(f"Access token generated (first 40 chars): {access_tok[:40]}...")

        decoded = decode_token(access_tok)
        assert decoded.sub == "user-uuid-abc123"
        assert decoded.role == "Researcher"
        assert decoded.token_type == "access"
        print("Access token encode/decode: PASSED")

        # Test refresh token creation and decoding
        refresh_tok, expire_dt = create_refresh_token("user-uuid-abc123", "Researcher")
        decoded_refresh = decode_token(refresh_tok)
        assert decoded_refresh.sub == "user-uuid-abc123"
        assert decoded_refresh.token_type == "refresh"
        assert decoded_refresh.exp > datetime.now(timezone.utc)
        print("Refresh token encode/decode: PASSED")

        # Test expired token raises ExpiredSignatureError
        settings = get_settings()
        expired_payload = {
            "sub": "user-uuid-abc123",
            "role": "User",
            "type": "access",
            "iat": datetime.now(timezone.utc) - timedelta(hours=1),
            "exp": datetime.now(timezone.utc) - timedelta(seconds=1)
        }
        expired_tok = _jwt_lib.encode(
            expired_payload,
            settings.JWT_SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM
        )
        try:
            decode_token(expired_tok)
            print("Expired token test: FAILED (should have raised ExpiredSignatureError)")
            sys.exit(1)
        except ExpiredSignatureError:
            print("Expired token correctly raises ExpiredSignatureError: PASSED")

        print("All self-tests completed successfully: PASSED")

    except Exception as e:
        print(f"Self-test failed with error: {e}", file=sys.stderr)
        sys.exit(1)
