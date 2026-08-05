"""
security.py

Provides bcrypt password hashing and verification utilities for the Authentix
authentication layer. All password operations go through this module exclusively.
"""

import logging
from pathlib import Path
import sys

import bcrypt

logger = logging.getLogger("backend.auth.security")


def hash_password(plain_password: str) -> str:
    """
    Hashes a plain-text password using bcrypt with a random salt.

    Args:
        plain_password: The raw password string to hash.

    Returns:
        A bcrypt-hashed password string (UTF-8 encoded).
    """
    password_bytes = plain_password.encode("utf-8")
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verifies a plain-text password against a stored bcrypt hash.

    Args:
        plain_password: The raw password string to check.
        hashed_password: The stored bcrypt hash to compare against.

    Returns:
        True if the password matches, False otherwise.
    """
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8")
        )
    except Exception as e:
        logger.warning(f"Password verification error: {e}")
        return False


def validate_password_strength(password: str) -> tuple[bool, str]:
    """
    Validates that a password meets minimum security requirements.

    Rules:
        - Minimum 8 characters
        - At least one uppercase letter
        - At least one lowercase letter
        - At least one digit
        - At least one special character

    Args:
        password: The plain-text password to validate.

    Returns:
        A tuple of (is_valid: bool, error_message: str).
        error_message is empty if the password is valid.
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter."
    if not any(c.islower() for c in password):
        return False, "Password must contain at least one lowercase letter."
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one digit."
    if not any(c in "!@#$%^&*()_+-=[]{}|;':\",./<>?" for c in password):
        return False, "Password must contain at least one special character."
    return True, ""


if __name__ == "__main__":
    print("Executing self-test for backend/app/auth/security.py...")
    try:
        # Test password hashing
        raw = "SecurePass@123"
        hashed = hash_password(raw)
        print(f"Hashed password: {hashed[:30]}...")
        assert hashed != raw, "Hash must differ from plain text"

        # Test verification - correct password
        assert verify_password(raw, hashed) is True, "Correct password verification failed"
        print("Correct password verification: PASSED")

        # Test verification - wrong password
        assert verify_password("WrongPass@999", hashed) is False, "Wrong password should fail"
        print("Wrong password verification: PASSED")

        # Test password strength
        ok, msg = validate_password_strength("SecurePass@123")
        assert ok is True and msg == "", f"Strength check failed: {msg}"
        print("Password strength check (strong): PASSED")

        bad, bad_msg = validate_password_strength("weak")
        assert bad is False and bad_msg != ""
        print(f"Password strength check (weak): PASSED ({bad_msg})")

        print("All self-tests completed successfully: PASSED")
    except Exception as e:
        print(f"Self-test failed with error: {e}", file=sys.stderr)
        sys.exit(1)
