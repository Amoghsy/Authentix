"""
logging_config.py

This module contains centralized logging configurations for the Authentix FastAPI server.
It sets up standard formatting, console outputs (stdout), and log file outputs.
"""

import logging
from pathlib import Path
import sys
from typing import Optional

# Ensure project root is in path for direct execution
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.app.config import Settings, get_settings


def configure_logging(settings: Optional[Settings] = None) -> None:
    """
    Configures the root logger to write formatting logs to sys.stdout and a persistent log file.
    
    Args:
        settings (Optional[Settings]): Settings instance.
    """
    cfg = settings or get_settings()
    
    # Logs directory under backend/logs/
    logs_dir = cfg.BACKEND_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "backend.log"
    
    # Configure formatter
    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(log_format, datefmt=date_format)
    
    # Retrieve root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear pre-existing handlers to prevent duplicated logs
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    # File Handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    
    # Console Stream Handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    
    # Configure third-party loggers to avoid verbosity
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("multipart").setLevel(logging.WARNING)
    
    logging.info(f"Initialized backend application logging. Logs destination file: {log_file.resolve()}")


if __name__ == "__main__":
    print("Executing self-test for backend/app/logging_config.py...")
    try:
        configure_logging()
        logger = logging.getLogger("backend_test")
        logger.info("Self-test logging statement: PASSED")
        
        # Verify log file creation
        cfg = get_settings()
        log_file = cfg.BACKEND_ROOT / "logs" / "backend.log"
        assert log_file.exists()
        print("Log file check: PASSED")
        
        print("All self-tests completed successfully: PASSED")
    except Exception as e:
        print(f"Self-test failed with error: {e}", file=sys.stderr)
        sys.exit(1)
