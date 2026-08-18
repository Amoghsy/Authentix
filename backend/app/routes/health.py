"""
health.py

This module defines the /health route for checking API status, hardware compatibility,
ONNX Runtime versions, and loaded model singletons status.
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
from backend.app.models.response_models import HealthResponse
import backend.app.dependencies as deps

logger = logging.getLogger("backend.routes.health")
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """
    Returns the backend service health status and Hugging Face AI service integration status.
    """
    logger.info("Ingesting health check lookup request...")
    
    models_state = {
        "fusion": deps._fusion_predictor is not None,
        "hf_client": deps._hf_client is not None
    }
    
    return HealthResponse(
        status="healthy",
        device=settings.DEVICE,
        gpu_available=False,
        onnx_runtime_version="External HF Service",
        models_loaded=models_state
    )



if __name__ == "__main__":
    print("Executing self-test for backend/app/routes/health.py...")
    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    
    try:
        response = client.get("/health")
        print(f"Health Response Status Code: {response.status_code}")
        print(f"Health Response JSON: {response.json()}")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        print("Health route self-test: PASSED")
    except Exception as e:
        print(f"Self-test failed with error: {e}", file=sys.stderr)
        sys.exit(1)
