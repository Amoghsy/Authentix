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
from dotenv import load_dotenv

# Load .env file from the project root automatically (if it exists).
# Variables already set in the environment take precedence over .env values.
_env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=_env_path, override=False)


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
    
    # CORS Origins Configurations
    CORS_ORIGINS: list[str] = Field(
        default_factory=lambda: [
            origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,https://authentix-deepfake.vercel.app").split(",") if origin.strip()
        ]
    )
    
    # Database Configurations
    DATABASE_URL: str = Field(
        default_factory=lambda: (
            os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/authentix")
            .replace("postgres://", "postgresql+asyncpg://", 1)
            .replace("postgresql://", "postgresql+asyncpg://", 1)
            if os.getenv("DATABASE_URL") else "postgresql+asyncpg://postgres:postgres@localhost:5432/authentix"
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
    
    # External Hugging Face AI Inference Service Configurations
    HF_INFERENCE_URL: str = Field(
        default_factory=lambda: os.getenv("HF_INFERENCE_URL", "http://localhost:7860")
    )
    HF_INFERENCE_API_KEY: str = Field(
        default_factory=lambda: os.getenv("HF_INFERENCE_API_KEY", "")
    )
    HF_INFERENCE_TIMEOUT: float = Field(
        default_factory=lambda: float(os.getenv("HF_INFERENCE_TIMEOUT", "120.0"))
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
    Validates backend configuration settings on startup.
    Ensures runtime directories exist and logs the configured HF Inference URL.
    """
    setup_app_directories(settings)
    if not settings.HF_INFERENCE_URL:
        raise ValueError("HF_INFERENCE_URL must be configured.")



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
