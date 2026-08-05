"""
request_models.py

This module contains the API input request models for the Authentix FastAPI backend.
All classes inherit from Pydantic's BaseModel for strict validation.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    """
    Optional analysis configuration override inputs (can be passed alongside form data).
    """
    settings_override: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dynamic settings overrides for thresholds or execution devices."
    )


if __name__ == "__main__":
    print("Executing self-test for backend/app/models/request_models.py...")
    try:
        req = AnalysisRequest(settings_override={"device": "cpu"})
        print(f"Instantiated AnalysisRequest: {req.settings_override}")
        assert req.settings_override["device"] == "cpu"
        print("All self-tests completed successfully: PASSED")
    except Exception as e:
        import sys
        print(f"Self-test failed with error: {e}", file=sys.stderr)
        sys.exit(1)
