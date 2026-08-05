"""
validation.py

This module contains file size, extension, and content type validation utilities
for the Authentix FastAPI backend.
"""

import logging
from pathlib import Path
import sys
from typing import Set

# Ensure project root is in path for direct execution
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.app.exceptions import FileValidationError

logger = logging.getLogger("backend.utils.validation")


def validate_file_size(file_size: int, max_size: int) -> None:
    """
    Checks that the uploaded file size does not exceed the maximum allowed limit.
    
    Raises:
        FileValidationError: If the file size is too large or invalid.
    """
    if file_size <= 0:
        raise FileValidationError("Invalid file size: file is empty.")
        
    if file_size > max_size:
        max_mb = max_size / (1024 * 1024)
        size_mb = file_size / (1024 * 1024)
        raise FileValidationError(
            f"File size exceeds limit of {max_mb:.1f} MB (Uploaded: {size_mb:.1f} MB)."
        )


def validate_file_format(
    filename: str,
    content_type: str,
    allowed_formats: Set[str],
    allowed_mime_types: Set[str]
) -> None:
    """
    Validates the filename extension and Content-Type MIME type headers.
    
    Raises:
        FileValidationError: If either the extension or MIME type is not allowed.
    """
    if not filename:
        raise FileValidationError("Missing uploaded filename.")
        
    # Extract extension
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext not in allowed_formats:
        raise FileValidationError(
            f"Unsupported file format: '.{ext}'. Supported formats: {', '.join(allowed_formats)}"
        )
        
    # Check Content-Type header
    mime = content_type.lower().strip()
    if mime not in allowed_mime_types:
        raise FileValidationError(
            f"Unsupported MIME type: '{mime}'. Supported MIME types: {', '.join(allowed_mime_types)}"
        )


if __name__ == "__main__":
    print("Executing self-test for backend/app/utils/validation.py...")
    try:
        # 1. Test size checks
        validate_file_size(10 * 1024 * 1024, max_size=50 * 1024 * 1024)
        
        try:
            validate_file_size(60 * 1024 * 1024, max_size=50 * 1024 * 1024)
            print("Validation FAILED: Accepted oversized file.", file=sys.stderr)
            sys.exit(1)
        except FileValidationError:
            print("Size bounds check 1: PASSED")
            
        try:
            validate_file_size(0, max_size=50 * 1024 * 1024)
            print("Validation FAILED: Accepted empty file.", file=sys.stderr)
            sys.exit(1)
        except FileValidationError:
            print("Size bounds check 2: PASSED")
            
        # 2. Test format checks
        formats = {"mp4", "mov"}
        mimes = {"video/mp4", "video/quicktime"}
        
        validate_file_format("video.mp4", "video/mp4", formats, mimes)
        print("Valid format check: PASSED")
        
        try:
            validate_file_format("script.sh", "text/x-shellscript", formats, mimes)
            print("Validation FAILED: Accepted blacklisted shell script.", file=sys.stderr)
            sys.exit(1)
        except FileValidationError:
            print("Invalid format check 1: PASSED")
            
        try:
            validate_file_format("fake_video.mp4", "application/octet-stream", formats, mimes)
            print("Validation FAILED: Accepted incorrect MIME type wrapper.", file=sys.stderr)
            sys.exit(1)
        except FileValidationError:
            print("Invalid format check 2: PASSED")
            
        print("All self-tests completed successfully: PASSED")
    except Exception as e:
        print(f"Self-test failed with error: {e}", file=sys.stderr)
        sys.exit(1)
