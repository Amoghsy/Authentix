"""
cleanup.py

This module contains cleanup utilities for the Authentix FastAPI backend.
It manages deleting temporary file uploads, local WAV files, and report directories.
"""

import logging
from pathlib import Path
import shutil
import sys
import time

from typing import Optional

# Ensure project root is in path for direct execution
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.app.config import Settings, get_settings

logger = logging.getLogger("backend.utils.cleanup")


def safe_delete_file(file_path: Path) -> None:
    """
    Safely deletes a file from disk if it exists, logging exceptions.
    """
    path = Path(file_path)
    try:
        if path.exists() and path.is_file():
            path.unlink()
            logger.info(f"Cleaned up file: {path.resolve()}")
    except Exception as e:
        logger.error(f"Failed to delete file {path.resolve()}: {e}")


def safe_delete_directory(dir_path: Path) -> None:
    """
    Safely deletes a directory and all of its contents if it exists.
    """
    path = Path(dir_path)
    try:
        if path.exists() and path.is_dir():
            shutil.rmtree(path)
            logger.info(f"Cleaned up directory: {path.resolve()}")
    except Exception as e:
        logger.error(f"Failed to delete directory {path.resolve()}: {e}")


def cleanup_session_temp_files(analysis_id: str, settings: Optional[Settings] = None) -> None:
    """
    Cleans up all intermediate temporary files generated for a specific analysis ID
    while keeping the final generated reports intact.
    
    Deletes:
      - Uploaded video file under uploads/
      - Temp directory under temp/analysis_id
      - Preprocessed lip sync items under ai/lip_sync/dataset/processed/analysis_id
    """
    cfg = settings or get_settings()
    
    # 1. Clear temp workspace folders
    temp_dir = cfg.TEMP_DIR / analysis_id
    safe_delete_directory(temp_dir)
    
    # 2. Clear preprocessed Lip Sync folders (from sync_preprocessor output)
    lip_sync_dataset_dir = cfg.PROJECT_ROOT / "ai" / "lip_sync" / "dataset" / "processed" / analysis_id
    safe_delete_directory(lip_sync_dataset_dir)
    
    # 3. Clear uploads directory files associated with this analysis id
    # Since filename is generated as uuid.ext, we search for files matching analysis_id in UPLOADS_DIR
    try:
        if cfg.UPLOADS_DIR.exists():
            for file_path in cfg.UPLOADS_DIR.glob(f"{analysis_id}.*"):
                safe_delete_file(file_path)
    except Exception as e:
        logger.error(f"Failed to scan and clean uploads for analysis {analysis_id}: {e}")


def cleanup_entire_session(analysis_id: str, settings: Optional[Settings] = None) -> None:
    """
    Performs full cleanup of a session, including both temporary files and the final reports.
    Typically called during DELETE /api/report/{analysis_id}.
    """
    cfg = settings or get_settings()
    
    # Clean temp files
    cleanup_session_temp_files(analysis_id, cfg)
    
    # Clean reports folder
    reports_dir = cfg.REPORTS_DIR / analysis_id
    safe_delete_directory(reports_dir)


if __name__ == "__main__":
    print("Executing self-test for backend/app/utils/cleanup.py...")
    import tempfile
    
    try:
        # Create temp settings for mock testing
        temp_base = Path(tempfile.mkdtemp())
        
        cfg = Settings(
            UPLOADS_DIR=temp_base / "uploads",
            TEMP_DIR=temp_base / "temp",
            REPORTS_DIR=temp_base / "reports",
            STATIC_DIR=temp_base / "static"
        )
        
        # Initialize directories
        cfg.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        cfg.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        
        analysis_id = "test-session-uuid"
        
        # Place test file uploads and temp dirs
        mock_upload = cfg.UPLOADS_DIR / f"{analysis_id}.mp4"
        mock_upload.write_text("dummy video")
        
        mock_temp = cfg.TEMP_DIR / analysis_id
        mock_temp.mkdir(parents=True, exist_ok=True)
        (mock_temp / "extracted.wav").write_text("dummy wave")
        
        mock_report = cfg.REPORTS_DIR / analysis_id
        mock_report.mkdir(parents=True, exist_ok=True)
        (mock_report / "report.json").write_text("dummy json")
        
        # Test temp file cleanup (reports should stay)
        print("Running temp cleanup...")
        cleanup_session_temp_files(analysis_id, cfg)
        
        assert not mock_upload.exists()
        assert not mock_temp.exists()
        assert mock_report.exists()
        print("Temp cleanup checks: PASSED")
        
        # Test full cleanup (reports should go)
        print("Running full cleanup...")
        cleanup_entire_session(analysis_id, cfg)
        assert not mock_report.exists()
        print("Full cleanup checks: PASSED")
        
        # Cleanup root temp
        shutil.rmtree(temp_base)
        print("All self-tests completed successfully: PASSED")
        
    except Exception as e:
        print(f"Self-test failed with error: {e}", file=sys.stderr)
        sys.exit(1)
