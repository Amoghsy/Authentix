"""
base.py

This module defines the declarative base class for all database models
used in the Authentix FastAPI backend.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """
    Base class for all SQLAlchemy database models.
    Leverages SQLAlchemy 2.0 type mapping features.
    """
    pass


if __name__ == "__main__":
    print("Executing self-test for backend/app/database/base.py...")
    try:
        # Check instance properties
        b = Base()
        print(f"Instantiated SQLAlchemy Base class: {b}")
        print("All self-tests completed successfully: PASSED")
    except Exception as e:
        import sys
        print(f"Self-test failed with error: {e}", file=sys.stderr)
        sys.exit(1)
