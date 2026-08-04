"""Helper utilities for standalone audio inference.

This module implements:
- Audio file validation checks.
- Metadata extraction (sample rate, channels, duration).
- Timer context manager utility.
- Results text serialization formats (JSON, CSV).
- Standardized error handlers.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Set, Tuple, Union

import soundfile as sf
import torchaudio


class Timer:
    """Context manager to measure execution durations in milliseconds."""

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000.0


def validate_audio_file(file_path: Union[str, Path]) -> Tuple[bool, str]:
    """Validates existence, format, and content metrics of target audio files.

    Args:
        file_path: Path of the audio file.

    Returns:
        Tuple[bool, str]:
            - bool: True if valid, False if invalid.
            - str: Validation details or error explanation message.
    """
    path = Path(file_path)
    supported_extensions: Set[str] = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}

    # 1. Check path exists
    if not path.exists():
        return False, "File does not exist"

    # 2. Check is actual file
    if not path.is_file():
        return False, "Path is not a file"

    # 3. Check extension
    if path.suffix.lower() not in supported_extensions:
        return False, f"Unsupported audio extension: {path.suffix}"

    # 4. Check file size
    try:
        size = path.stat().st_size
        if size == 0:
            return False, "Empty file (0 bytes)"
    except Exception as e:
        return False, f"Failed to read file stats: {e}"

    return True, "Valid"


def extract_audio_metadata(file_path: Union[str, Path]) -> Dict[str, Union[str, float, int]]:
    """Reads metadata properties of the target audio recording.

    Args:
        file_path: Path to the target audio.

    Returns:
        Dict: Metadata dictionary (sample_rate, channels, duration_seconds, file_size_bytes).
    """
    path = Path(file_path)
    metadata = {
        "file_name": path.name,
        "file_size_bytes": path.stat().st_size,
        "sample_rate": 0,
        "channels": 0,
        "duration_seconds": 0.0
    }

    try:
        try:
            info = torchaudio.info(str(path))
            metadata["sample_rate"] = info.sample_rate
            metadata["channels"] = info.num_channels
            metadata["duration_seconds"] = info.num_frames / info.sample_rate if info.sample_rate > 0 else 0.0
        except Exception:
            # Fallback to soundfile loading
            info_sf = sf.info(str(path))
            metadata["sample_rate"] = info_sf.samplerate
            metadata["channels"] = info_sf.channels
            metadata["duration_seconds"] = info_sf.duration
    except Exception as e:
        # Warn but return partial stats
        logging.getLogger("audio_inference.utils").warning(f"Could not extract metadata from {path.name}: {e}")

    return metadata


def serialize_prediction(
    result: Dict,
    file_path: Union[str, Path],
    output_format: str = "json"
) -> str:
    """Serializes prediction outputs to target text formats (JSON or CSV).

    Args:
        result: Prediction metrics dictionary.
        file_path: Original audio path.
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
        # Format as simple CSV values matching headers
        headers = ["file_path", "file_name", "prediction", "confidence", "fake_probability", "real_probability", "latency_ms"]
        row = [str(data.get(h, "")) for h in headers]
        return ",".join(row)

    return json.dumps(data, indent=4)


def handle_inference_error(error: Exception, file_path: Union[str, Path]) -> Dict:
    """Constructs a standard prediction error response matching expected outputs.

    Args:
        error: Caught execution Exception.
        file_path: Original audio path.

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
        "latency_ms": 0.0,
        "error_message": str(error)
    }
