"""
middleware.py

This module contains middleware configurations for the Authentix FastAPI server.
It sets up CORS allowances and logs execution latency for every incoming HTTP request.
"""

import logging
from pathlib import Path
import sys
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

# Ensure project root is in path for direct execution
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logger = logging.getLogger("backend.middleware")


class LatencyLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that measures and records HTTP request execution latency.
    """
    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()
        
        # Process the request
        try:
            response = await call_next(request)
        except Exception as e:
            # Latency logging even in case of unhandled server exception
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(
                f"HTTP {request.method} {request.url.path} - FAILED with exception - "
                f"Latency: {latency_ms:.2f}ms"
            )
            raise e
            
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        response.headers["X-Process-Time-Ms"] = f"{latency_ms:.2f}"
        
        # Log latency of request
        logger.info(
            f"HTTP {request.method} {request.url.path} - "
            f"Status: {response.status_code} - Latency: {latency_ms:.2f}ms"
        )
        
        return response


def register_middleware(app: FastAPI) -> None:
    """
    Registers the CORS policy and latency measuring middleware onto the FastAPI app instance.
    """
    from backend.app.config import get_settings
    settings = get_settings()
    
    # 1. Setup CORS middleware
    allow_origins = settings.CORS_ORIGINS
    allow_credentials = True
    if "*" in allow_origins or (len(allow_origins) == 1 and allow_origins[0] == "*"):
        allow_credentials = False
        
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 2. Setup custom Latency logging middleware
    app.add_middleware(LatencyLoggingMiddleware)


if __name__ == "__main__":
    print("Executing self-test for backend/app/middleware.py...")
    try:
        # Create a mock FastAPI app to test registrations
        app = FastAPI()
        register_middleware(app)
        print("FastAPI Middleware registration: PASSED")
        print("All self-tests completed successfully: PASSED")
    except Exception as e:
        print(f"Self-test failed with error: {e}", file=sys.stderr)
        sys.exit(1)
