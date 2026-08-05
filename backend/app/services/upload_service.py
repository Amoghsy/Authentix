"""
upload_service.py

This service handles multipart video uploads, performs content validation checks,
and saves the uploaded file to disk using secure UUID-based filenames.
"""

import logging
from pathlib import Path
import sys
from typing import Optional
from fastapi import UploadFile

# Ensure project root is in path for direct execution
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.app.config import Settings, get_settings
from backend.app.exceptions import FileValidationError
from backend.app.utils.file_utils import generate_secure_filename, save_upload_file, is_safe_path
from backend.app.utils.validation import validate_file_size, validate_file_format

logger = logging.getLogger("backend.services.upload")


class UploadService:
    """Handles uploading files securely to the uploads directory."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

    async def upload_video(self, file: UploadFile) -> Path:
        """
        Validates content-type headers, size limits, format tags, and saves to uploads/.
        
        Args:
            file (UploadFile): Uploaded file from route.
            
        Returns:
            Path: The resolved absolute path to the uploaded file.
            
        Raises:
            FileValidationError: If size, extension, or MIME checks fail.
        """
        if not file or not file.filename:
            raise FileValidationError("Invalid upload request: no file provided.")

        # Read content type and validate headers
        # Strip codec parameters (e.g. 'video/mp4; codecs="avc1"') → 'video/mp4'
        raw_content_type = file.content_type or ""
        content_type = raw_content_type.split(";")[0].strip().lower()
        filename = file.filename
        
        logger.info(f"Validating file upload: Name={filename}, MIME={content_type}")
        
        # Validate format
        validate_file_format(
            filename=filename,
            content_type=content_type,
            allowed_formats=self.settings.ALLOWED_FORMATS,
            allowed_mime_types=self.settings.ALLOWED_MIME_TYPES
        )

        # Read file content to determine size, then seek back.
        # UploadFile.seek() only accepts a single position argument (no whence).
        try:
            content = await file.read()
            size = len(content)
            await file.seek(0)  # Reset pointer to start for downstream reads
        except Exception as e:
            logger.error(f"Failed to resolve uploaded file size: {e}")
            raise FileValidationError(f"Failed to read file size: {e}")

        # Validate size
        validate_file_size(size, self.settings.MAX_UPLOAD_SIZE)

        # Generate safe unique filename
        secure_name = generate_secure_filename(filename)
        destination_path = self.settings.UPLOADS_DIR / secure_name

        # Prevent traversal
        if not is_safe_path(self.settings.UPLOADS_DIR, destination_path):
            raise FileValidationError("Directory traversal attempt detected.")

        # Save to disk
        logger.info(f"Saving upload to: {destination_path.resolve()}")
        await save_upload_file(file, destination_path)
        logger.info("File upload saved successfully.")

        return destination_path.resolve()


if __name__ == "__main__":
    print("Executing self-test for backend/app/services/upload_service.py...")
    from typing import Optional
    # Setup simple mock objects to test
    class MockUploadFile:
        def __init__(self, filename: str, content_type: str, data: bytes):
            self.filename = filename
            self.content_type = content_type
            self.data = data
            self.pointer = 0

        async def read(self, chunk_size: int = -1):
            if self.pointer >= len(self.data):
                return b""
            chunk = self.data[self.pointer:self.pointer + chunk_size]
            self.pointer += len(chunk)
            return chunk

        async def seek(self, offset: int, whence: int = 0):
            if whence == 2:
                self.pointer = len(self.data)
            else:
                self.pointer = offset

        async def tell(self):
            return self.pointer

    import tempfile
    import asyncio

    async def run_test():
        temp_dir = Path(tempfile.mkdtemp())
        cfg = Settings(UPLOADS_DIR=temp_dir)
        service = UploadService(cfg)
        
        # Test valid upload
        mock_file = MockUploadFile("my_video.mp4", "video/mp4", b"dummy mp4 video bytes")
        path = await service.upload_video(mock_file)
        print(f"Uploaded Path: {path}")
        assert path.exists()
        assert path.parent == temp_dir.resolve()
        
        # Cleanup
        path.unlink()
        temp_dir.rmdir()
        print("UploadService self-test: PASSED")

    asyncio.run(run_test())
