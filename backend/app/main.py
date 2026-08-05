"""
main.py  [UPDATED - Part 6]

Primary entry point for the Authentix FastAPI backend.
Changes from Part 5:
  - Database engine startup: creates all tables on first boot (dev mode).
  - Registers auth router (/api/auth/*)
  - Registers history router (/api/history/*)
  - DB session lifecycle is managed by get_db dependency (per-request).
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
from backend.app.routes import health, version, analyze, history
from backend.app.auth import router as auth_router_module

logger = logging.getLogger("backend.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown lifecycles.
    Order:
        1. Configure logging
        2. Setup workspace directories
        3. Validate model file assets
        4. Initialize database tables (create_all for dev convenience)
        5. Warm up AI ONNX model singleton sessions
    """
    settings = get_settings()

    # 1. Logging setup
    configure_logging(settings)
    logger.info("Starting up Authentix FastAPI backend...")

    try:
        # 2. Workspace directories
        setup_app_directories(settings)

        # 3. Validate ONNX model files exist
        validate_settings(settings)

        # 4. Create database tables if they don't exist yet (dev mode)
        #    In production, use Alembic migrations instead.
        from backend.app.database.session import engine
        from backend.app.database.base import Base
        import backend.app.database.models  # noqa: F401 — registers all ORM models

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables verified / created.")

        # 5. Warm up AI model sessions
        warm_up_services()

        logger.info("Authentix backend startup complete. Service is ready.")
        yield

    except Exception as e:
        logger.critical(f"Startup failed: {e}", exc_info=True)
        raise e

    finally:
        # Dispose the engine connection pool on shutdown
        from backend.app.database.session import engine as _engine
        await _engine.dispose()
        logger.info("Database engine disposed. Authentix backend shut down cleanly.")


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Authentix API",
    description=(
        "Production-grade Multimodal Fusion Deepfake Detection API. "
        "Supports user authentication, JWT authorization, AI inference, "
        "and persistent analysis history."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# Exception handlers
register_exception_handlers(app)

# CORS + timing middleware
register_middleware(app)

# ---------------------------------------------------------------------------
# Route Registration
# ---------------------------------------------------------------------------
# System routes (Part 5)
app.include_router(health.router)
app.include_router(version.router)
app.include_router(analyze.router)

# Auth routes (Part 6)
app.include_router(auth_router_module.router)

# History routes (Part 6)
app.include_router(history.router)


if __name__ == "__main__":
    import uvicorn
    print("Starting Authentix backend with uvicorn...")
    uvicorn.run(
        "backend.app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
