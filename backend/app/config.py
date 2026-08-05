"""
config.py

This module manages backend settings and folder mappings for the Authentix FastAPI server.
It defines allowed video formats, upload size constraints, directories setup, and AI
model locations.
"""

from functools import lru_cache
import os
from pathlib import Path
from typing import Set
from pydantic import BaseModel, Field, ConfigDict


class Settings(BaseModel):
    """
    Application-wide configuration parameters and constants.
    """
    PROJECT_NAME: str = "Authentix API"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    
    # Path Resolution
    # APP_ROOT matches .../backend/app/
    APP_ROOT: Path = Field(default_factory=lambda: Path(__file__).resolve().parent)
    # BACKEND_ROOT matches .../backend/
    BACKEND_ROOT: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1])
    # PROJECT_ROOT matches .../Authentix/
    PROJECT_ROOT: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])
    
    # Runtime Directories
    UPLOADS_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "uploads")
    TEMP_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "temp")
    REPORTS_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "reports")
    STATIC_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "static")
    
    # Upload Constraints
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100 Megabytes
    ALLOWED_FORMATS: Set[str] = {"mp4", "avi", "mov", "mkv"}
    ALLOWED_MIME_TYPES: Set[str] = {
        "video/mp4",
        "video/x-msvideo",
        "video/quicktime",
        "video/x-matroska",
        "video/webm"
    }
    
    # Model Configurations
    DEVICE: str = Field(default_factory=lambda: os.getenv("AUTHENTIX_DEVICE", "cuda"))
    
    # Database Configurations
    DATABASE_URL: str = Field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/authentix"
        )
    )
    
    # Security & Authentication Configurations
    JWT_SECRET_KEY: str = Field(
        default_factory=lambda: os.getenv(
            "JWT_SECRET_KEY", "authentix_jwt_secret_key_change_me_in_production_extremely_long_key_1234567890"
        )
    )
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Absolute paths to completed AI models inside workspace
    VIDEO_MODEL_PATH: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "ai" / "video" / "exports" / "deepfake_model.onnx"
    )
    AUDIO_MODEL_PATH: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "exports" / "audio_model.onnx"
    )
    FACE_LANDMARKER_PATH: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "ai" / "lip_sync" / "checkpoints" / "face_landmarker.task"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached settings instance (singleton pattern).
    """
    return Settings()


def setup_app_directories(settings: Settings) -> None:
    """
    Creates runtime directories for uploads, reports, temp files, and static files if missing.
    """
    dirs = [
        settings.UPLOADS_DIR,
        settings.TEMP_DIR,
        settings.REPORTS_DIR,
        settings.STATIC_DIR
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def validate_settings(settings: Settings) -> None:
    """
    Validates model paths and folder settings on startup.
    """
    # Verify Video model
    if not settings.VIDEO_MODEL_PATH.exists():
        raise FileNotFoundError(f"Video AI Model not found at: {settings.VIDEO_MODEL_PATH.resolve()}")
    
    # Verify Audio model
    if not settings.AUDIO_MODEL_PATH.exists():
        raise FileNotFoundError(f"Audio AI Model not found at: {settings.AUDIO_MODEL_PATH.resolve()}")

    # Verify Face Landmarker task model
    if not settings.FACE_LANDMARKER_PATH.exists():
        raise FileNotFoundError(f"Face Landmarker model not found at: {settings.FACE_LANDMARKER_PATH.resolve()}")


if __name__ == "__main__":
    print("Executing self-test for backend/app/config.py...")
    try:
        cfg = get_settings()
        print(f"Loaded Settings: {cfg.PROJECT_NAME} v{cfg.VERSION}")
        print(f"Project Root: {cfg.PROJECT_ROOT}")
        print(f"Uploads Dir: {cfg.UPLOADS_DIR}")
        print(f"Video Model: {cfg.VIDEO_MODEL_PATH}")
        print(f"Audio Model: {cfg.AUDIO_MODEL_PATH}")
        
        print("Setting up directories...")
        setup_app_directories(cfg)
        
        print("Validating settings assets...")
        validate_settings(cfg)
        
        print("All self-tests completed successfully: PASSED")
    except Exception as e:
        import sys
        print(f"Self-test failed with error: {e}", file=sys.stderr)
        sys.exit(1)
