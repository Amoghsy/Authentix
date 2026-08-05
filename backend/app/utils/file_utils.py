"""
file_utils.py

This module contains file handling utility functions for the Authentix backend.
It implements secure path checks, secure filename generation, and chunked upload writes.
"""

import logging
from pathlib import Path
import shutil
import sys
import uuid
from fastapi import UploadFile

# Ensure project root is in path for direct execution
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logger = logging.getLogger("backend.utils.file")


def is_safe_path(base_dir: Path, target_path: Path) -> bool:
    """
    Prevents path traversal vulnerabilities by verifying that the target path
    is strictly resolved inside the base directory.
    """
    try:
        resolved_base = Path(base_dir).resolve()
        resolved_target = Path(target_path).resolve()
        return resolved_base in resolved_target.parents or resolved_base == resolved_target
    except Exception as e:
        logger.error(f"Path safety evaluation failed for {target_path}: {e}")
        return False


def generate_secure_filename(original_filename: str) -> str:
    """
    Generates a secure, unique filename to prevent filename collisions and injections.
    Retains the original file extension.
    """
    ext = Path(original_filename).suffix.lower()
    # Sanitize extension to standard alphanumeric tags
    if len(ext) > 5 or not ext[1:].isalnum():
        ext = ".mp4"  # Default fallback safe extension
        
    unique_id = uuid.uuid4().hex
    return f"{unique_id}{ext}"


async def save_upload_file(upload_file: UploadFile, destination: Path) -> int:
    """
    Saves an uploaded FastAPI file in chunks to a target path to conserve memory.
    
    Args:
        upload_file (UploadFile): Ingested upload file instance.
        destination (Path): Target file path.
        
    Returns:
        int: Total size in bytes written to disk.
    """
    total_bytes = 0
    chunk_size = 1024 * 1024  # 1MB chunks
    
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(destination, "wb") as f:
            while True:
                chunk = await upload_file.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                total_bytes += len(chunk)
        return total_bytes
    except Exception as e:
        # Cleanup incomplete file on exception
        if destination.exists():
            destination.unlink()
        logger.error(f"Failed to save uploaded file to {destination}: {e}")
        raise IOError(f"File write operation failed: {e}")
    finally:
        # Reset file pointer for subsequent reads if needed
        await upload_file.seek(0)


if __name__ == "__main__":
    print("Executing self-test for backend/app/utils/file_utils.py...")
    import tempfile
    
    try:
        # Test secure filename
        name1 = generate_secure_filename("my_video.MP4")
        print(f"Generated filename: {name1}")
        assert name1.endswith(".mp4")
        assert len(name1) > 10
        
        name2 = generate_secure_filename("traversal_attempt.PHP?test=123")
        print(f"Fallback filename: {name2}")
        assert name2.endswith(".mp4")  # Safe fallback suffix
        
        # Test path traversal safety
        temp_base = Path(tempfile.mkdtemp())
        safe_targ = temp_base / "child" / "file.txt"
        unsafe_targ = temp_base / ".." / "outside_dir" / "hack.txt"
        
        assert is_safe_path(temp_base, safe_targ) is True
        assert is_safe_path(temp_base, unsafe_targ) is False
        
        # Cleanup
        temp_base.rmdir()
        print("All self-tests completed successfully: PASSED")
    except Exception as e:
        print(f"Self-test failed with error: {e}", file=sys.stderr)
        sys.exit(1)
