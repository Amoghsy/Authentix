"""Helper utilities for video deepfake standalone inference.

This module implements:
- Video file validation checks.
- Metadata extraction (resolution, duration, FPS, frame counts).
- Timer context manager utility.
- Results text serialization formats (JSON, CSV).
- Standardized error handlers.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Set, Tuple, Union

import cv2


class Timer:
    """Context manager to measure execution durations in milliseconds."""

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000.0


def validate_video_file(file_path: Union[str, Path]) -> Tuple[bool, str]:
    """Validates existence, format, and size of target video files.

    Args:
        file_path: Path of the video file.

    Returns:
        Tuple[bool, str]:
            - bool: True if valid, False if invalid.
            - str: Validation details or error explanation message.
    """
    path = Path(file_path)
    supported_extensions: Set[str] = {".mp4", ".avi", ".mov", ".mkv"}

    # 1. Check path exists
    if not path.exists():
        return False, "File does not exist"

    # 2. Check is actual file
    if not path.is_file():
        return False, "Path is not a file"

    # 3. Check extension
    if path.suffix.lower() not in supported_extensions:
        return False, f"Unsupported video extension: {path.suffix}"

    # 4. Check file size
    try:
        size = path.stat().st_size
        if size == 0:
            return False, "Empty file (0 bytes)"
    except Exception as e:
        return False, f"Failed to read file stats: {e}"

    return True, "Valid"


def extract_video_metadata(file_path: Union[str, Path]) -> Dict[str, Union[str, float, int]]:
    """Reads metadata properties of the target video recording using OpenCV.

    Args:
        file_path: Path to the target video.

    Returns:
        Dict: Metadata dictionary (resolution_width, resolution_height, fps, total_frames, duration_seconds, file_size_bytes).
    """
    path = Path(file_path)
    metadata = {
        "file_name": path.name,
        "file_size_bytes": path.stat().st_size,
        "resolution_width": 0,
        "resolution_height": 0,
        "fps": 0.0,
        "total_frames": 0,
        "duration_seconds": 0.0
    }

    try:
        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            duration = total_frames / fps if fps > 0 else 0.0

            metadata["resolution_width"] = width
            metadata["resolution_height"] = height
            metadata["fps"] = round(fps, 2)
            metadata["total_frames"] = total_frames
            metadata["duration_seconds"] = round(duration, 2)
            
            cap.release()
    except Exception as e:
        logging.getLogger("video_inference.utils").warning(f"Could not extract metadata from {path.name}: {e}")

    return metadata


def serialize_prediction(
    result: Dict,
    file_path: Union[str, Path],
    output_format: str = "json"
) -> str:
    """Serializes prediction outputs to target text formats (JSON or CSV).

    Args:
        result: Prediction metrics dictionary.
        file_path: Original video path.
        output_format: Output type ('json' or 'csv').

    Returns:
        str: Serialized result string.
    """
    path = Path(file_path)
    data = {
        "file_path": str(path.resolve()),
        "file_name": path.name,
        **result
    }

    if output_format.lower() == "csv":
        headers = ["file_path", "file_name", "prediction", "confidence", "fake_probability", "real_probability", "frames_processed", "latency_ms"]
        row = [str(data.get(h, "")) for h in headers]
        return ",".join(row)

    return json.dumps(data, indent=4)


def handle_inference_error(error: Exception, file_path: Union[str, Path]) -> Dict:
    """Constructs a standard prediction error response matching expected outputs.

    Args:
        error: Caught execution Exception.
        file_path: Original video path.

    Returns:
        Dict: Standardized error dictionary.
    """
    path = Path(file_path)
    return {
        "file_path": str(path.resolve()),
        "file_name": path.name,
        "prediction": "Unknown",
        "confidence": 0.0,
        "fake_probability": 0.0,
        "real_probability": 0.0,
        "frames_processed": 0,
        "latency_ms": 0.0,
        "error_message": str(error)
    }
