"""
schemas.py

Pydantic V2 request and response models for the Authentix authentication layer.
Covers: registration, login, token responses, profile updates, and current user output.
"""

import sys
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    """Payload for new user registration."""
    name: Optional[str] = Field(None, max_length=100, examples=["Alice Smith"])
    email: EmailStr = Field(..., examples=["alice@authentix.ai"])
    password: str = Field(..., min_length=8, max_length=128, examples=["Secure@Pass123"])

    @field_validator("password")
    @classmethod
    def password_must_be_strong(cls, v: str) -> str:
        errors = []
        if not any(c.isupper() for c in v):
            errors.append("one uppercase letter")
        if not any(c.islower() for c in v):
            errors.append("one lowercase letter")
        if not any(c.isdigit() for c in v):
            errors.append("one digit")
        if not any(c in "!@#$%^&*()_+-=[]{}|;':\",./<>?" for c in v):
            errors.append("one special character")
        if errors:
            raise ValueError(f"Password must contain: {', '.join(errors)}.")
        return v


class LoginRequest(BaseModel):
    """Payload for user login with email and password."""
    email: EmailStr = Field(..., examples=["alice@authentix.ai"])
    password: str = Field(..., min_length=1, examples=["Secure@Pass123"])


class RefreshRequest(BaseModel):
    """Payload for refresh token rotation."""
    refresh_token: str = Field(..., description="Valid non-revoked refresh token.")


class UpdateProfileRequest(BaseModel):
    """Payload for updating the current user's profile fields."""
    name: Optional[str] = Field(None, max_length=100, examples=["Alice Smith"])
    current_password: Optional[str] = Field(
        None, description="Required when changing password."
    )
    new_password: Optional[str] = Field(
        None, min_length=8, max_length=128,
        description="New password. Requires current_password to be provided."
    )

    @field_validator("new_password")
    @classmethod
    def new_password_must_be_strong(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        errors = []
        if not any(c.isupper() for c in v):
            errors.append("one uppercase letter")
        if not any(c.islower() for c in v):
            errors.append("one lowercase letter")
        if not any(c.isdigit() for c in v):
            errors.append("one digit")
        if not any(c in "!@#$%^&*()_+-=[]{}|;':\",./<>?" for c in v):
            errors.append("one special character")
        if errors:
            raise ValueError(f"New password must contain: {', '.join(errors)}.")
        return v


# ---------------------------------------------------------------------------
# Response Schemas
# ---------------------------------------------------------------------------

class TokenResponse(BaseModel):
    """Response payload after a successful login or token refresh."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    """
    Public representation of a user returned from the API.
    Never exposes the password_hash field.
    """
    uuid: str
    name: Optional[str]
    email: str
    role: str
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login: Optional[datetime]

    model_config = {"from_attributes": True}


class ProfileResponse(BaseModel):
    """Response payload for the /api/auth/me and /api/auth/profile endpoints."""
    uuid: str
    name: Optional[str]
    email: str
    role: str
    is_active: bool
    is_verified: bool
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime]

    model_config = {"from_attributes": True}


class MessageResponse(BaseModel):
    """Generic success message response."""
    message: str
    success: bool = True


if __name__ == "__main__":
    print("Executing self-test for backend/app/auth/schemas.py...")
    try:
        # Test RegisterRequest validation
        reg = RegisterRequest(
            name="Alice Smith",
            email="alice@authentix.ai",
            password="Secure@Pass123"
        )
        assert reg.email == "alice@authentix.ai"
        print("RegisterRequest validation: PASSED")

        # Test weak password rejection
        try:
            RegisterRequest(email="bob@test.ai", password="weakpass")
            print("Weak password test: FAILED (should have raised)")
            sys.exit(1)
        except Exception:
            print("Weak password rejection: PASSED")

        # Test LoginRequest
        login = LoginRequest(email="alice@authentix.ai", password="Secure@Pass123")
        assert login.email == "alice@authentix.ai"
        print("LoginRequest validation: PASSED")

        # Test TokenResponse
        tok = TokenResponse(access_token="acc.tok.xyz", refresh_token="ref.tok.xyz")
        assert tok.token_type == "bearer"
        print("TokenResponse validation: PASSED")

        # Test MessageResponse
        msg = MessageResponse(message="Logout successful.")
        assert msg.success is True
        print("MessageResponse validation: PASSED")

        print("All self-tests completed successfully: PASSED")

    except Exception as e:
        print(f"Self-test failed with error: {e}", file=sys.stderr)
        sys.exit(1)
