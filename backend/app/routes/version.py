"""
version.py

This module defines the /api/version route for retrieving backend, API, and AI model version logs.
"""

import logging
from pathlib import Path
import sys
from fastapi import APIRouter, Depends

# Ensure project root is in path for direct execution
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.app.config import Settings, get_settings
from backend.app.models.response_models import VersionResponse

logger = logging.getLogger("backend.routes.version")
router = APIRouter()


@router.get("/api/version", response_model=VersionResponse)
def get_version_info(settings: Settings = Depends(get_settings)) -> VersionResponse:
    """
    Returns the current semantic version information for the server API and AI weights configurations.
    """
    logger.info("Ingesting version lookup request...")
    
    # Static or configuration-derived model versions mapping
    model_versions = {
        "video_onnx_model": "1.0.2",
        "audio_onnx_model": "2.1.0",
        "lipsync_sfd_model": "1.0.0",
        "lipsync_face_mesh": "1.0.0"
    }
    
    return VersionResponse(
        backend_version=settings.VERSION,
        api_version="1.0.0",
        model_versions=model_versions
    )


if __name__ == "__main__":
    print("Executing self-test for backend/app/routes/version.py...")
    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    
    try:
        response = client.get("/api/version")
        print(f"Version Response Status Code: {response.status_code}")
        print(f"Version Response JSON: {response.json()}")
        assert response.status_code == 200
        assert response.json()["backend_version"] == "1.0.0"
        print("Version route self-test: PASSED")
    except Exception as e:
        print(f"Self-test failed with error: {e}", file=sys.stderr)
        sys.exit(1)
