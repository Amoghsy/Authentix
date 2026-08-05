"""
exceptions.py

This module contains custom exceptions and unified error handlers for the Authentix FastAPI server.
It maps model failures, file validations, and processing issues to standard HTTP JSON responses.
"""

import logging
from pathlib import Path
import sys
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Ensure project root is in path for direct execution
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logger = logging.getLogger("backend.exceptions")


class AuthentixException(Exception):
    """Base exception for all domain errors within Authentix."""
    def __init__(self, message: str, detail: Optional[Any] = None):
        super().__init__(message)
        self.message = message
        self.detail = detail


class FileValidationError(AuthentixException):
    """Raised when file extension, MIME type, size, or path traversal checks fail."""
    pass


class VideoProcessingError(AuthentixException):
    """Raised when frame extracts, codec parsing, or video formats fail."""
    pass


class AudioExtractionError(AuthentixException):
    """Raised when demuxing audio tracks or peak normalizations fail (e.g. missing audio tracks)."""
    pass


class ModelInferenceError(AuthentixException):
    """Raised when ONNX runtime, hardware (GPU), or predictor evaluations fail."""
    pass


class ReportNotFoundError(AuthentixException):
    """Raised when looking up an analysis ID that does not exist in the database."""
    pass


def register_exception_handlers(app: FastAPI) -> None:
    """
    Binds custom exceptions to status codes and registers them on the FastAPI app instance.
    """
    
    @app.exception_handler(AuthentixException)
    async def domain_exception_handler(request: Request, exc: AuthentixException):
        # Resolve status code mapping
        status_code = 500
        if isinstance(exc, FileValidationError):
            status_code = 400
        elif isinstance(exc, ReportNotFoundError):
            status_code = 404
        elif isinstance(exc, (VideoProcessingError, AudioExtractionError)):
            status_code = 400
            
        logger.error(
            f"API Domain Error caught: {exc.__class__.__name__} - {exc.message}. Details: {exc.detail}"
        )
        
        return JSONResponse(
            status_code=status_code,
            content={
                "success": False,
                "error": {
                    "type": exc.__class__.__name__,
                    "message": exc.message,
                    "detail": exc.detail
                }
            }
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.critical(f"Unhandled system error caught: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "type": exc.__class__.__name__,
                    "message": "An unexpected system error occurred while processing your request.",
                    "detail": str(exc)
                }
            }
        )


if __name__ == "__main__":
    print("Executing self-test for backend/app/exceptions.py...")
    try:
        # Create a mock FastAPI app to test registrations
        app = FastAPI()
        register_exception_handlers(app)
        print("FastAPI Exception handler registration: PASSED")
        
        # Test class instantiations
        ex1 = FileValidationError("Invalid file extension", detail="Allowed: .mp4")
        assert ex1.message == "Invalid file extension"
        assert ex1.detail == "Allowed: .mp4"
        
        ex2 = ModelInferenceError("ONNX initialization failed")
        assert ex2.message == "ONNX initialization failed"
        
        print("All self-tests completed successfully: PASSED")
    except Exception as e:
        print(f"Self-test failed with error: {e}", file=sys.stderr)
        sys.exit(1)
