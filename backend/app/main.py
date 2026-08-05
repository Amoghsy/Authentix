"""
main.py

This is the primary entry point for the Authentix FastAPI backend.
It coordinates application lifespans (pre-loading model sessions), logging setups,
middlewares registration, route structures, and unified exceptions handlers.
"""

from contextlib import asynccontextmanager
import logging
from pathlib import Path
import sys

from fastapi import FastAPI

# Ensure project root is in path for direct execution
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.app.config import get_settings, setup_app_directories, validate_settings
from backend.app.logging_config import configure_logging
from backend.app.exceptions import register_exception_handlers
from backend.app.middleware import register_middleware
from backend.app.dependencies import warm_up_services
from backend.app.routes import health, version, analyze

logger = logging.getLogger("backend.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown lifecycles.
    Eagerly loads and warms up AI model sessions.
    """
    settings = get_settings()
    
    # 1. Setup centralized logging formatters
    configure_logging(settings)
    logger.info("Starting up Authentix FastAPI backend services...")
    
    try:
        # 2. Build local uploads, reports, static, and temp directories if missing
        setup_app_directories(settings)
        
        # 3. Assert target ONNX and Landmark model assets exist in workspace
        validate_settings(settings)
        
        # 4. Cache and warm up singleton predictors sessions
        warm_up_services()
        
        logger.info("Authentix FastAPI backend startup tasks completed successfully. Service is ready.")
        yield
    except Exception as e:
        logger.critical(f"FastAPI backend failed to start cleanly: {e}", exc_info=True)
        raise e
    finally:
        logger.info("Shutting down Authentix FastAPI backend services...")


# Initialize FastAPI app instance
app = FastAPI(
    title="Authentix FastAPI Orchestrator",
    description="Production-grade Multimodal Fusion Deepfake Detection API service.",
    version="1.0.0",
    lifespan=lifespan
)

# Bind exception handlers
register_exception_handlers(app)

# Bind CORS policy and latency loggers
register_middleware(app)

# Include endpoint routes
app.include_router(health.router)
app.include_router(version.router)
app.include_router(analyze.router)


if __name__ == "__main__":
    import uvicorn
    print("Starting production uvicorn server command interface...")
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)
