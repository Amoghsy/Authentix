"""
utils.py

This module contains support utilities for the Authentix Fusion Engine.
It implements timezone-aware UTC timestamps, JSON serializers for paths and datetimes,
reusable range validators, custom exceptions, and logging bootstrapping helpers.
"""

from datetime import datetime, timezone
import json
from logging import Logger, getLogger
from pathlib import Path
import sys
from typing import Any, Dict, Optional


class FusionError(Exception):
    """Base exception class for all errors in the Fusion Engine."""
    pass


class InvalidDataError(FusionError):
    """Raised when score or confidence inputs violate constraints."""
    pass


class CustomJSONEncoder(json.JSONEncoder):
    """
    JSON encoder extension that handles serializations for Path and datetime instances.
    """
    def default(self, obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def get_utc_timestamp() -> str:
    """
    Generates a current ISO 8601 UTC timestamp string.
    
    Returns:
        str: Timestamp string, e.g. '2026-08-05T15:15:00.123456+00:00'.
    """
    return datetime.now(timezone.utc).isoformat()


def validate_score_and_confidence(score: float, confidence: float, name: str) -> None:
    """
    Validates that a metric and its confidence value are floats within [0.0, 1.0].
    
    Args:
        score (float): Probability score.
        confidence (float): Classifier confidence.
        name (str): Modality name tag for error messages.
        
    Raises:
        InvalidDataError: If boundaries are violated.
    """
    try:
        val_score = float(score)
        val_conf = float(confidence)
    except (TypeError, ValueError) as e:
        raise InvalidDataError(f"{name} inputs must be numeric values. Error: {e}")
        
    if not (0.0 <= val_score <= 1.0):
        raise InvalidDataError(f"{name} score must be in range [0.0, 1.0]. Provided: {score}")
    if not (0.0 <= val_conf <= 1.0):
        raise InvalidDataError(f"{name} confidence must be in range [0.0, 1.0]. Provided: {confidence}")


def write_json_file(data: Dict[str, Any], file_path: Path, indent: int = 4) -> None:
    """
    Saves a dictionary as a JSON file, resolving Paths and datetimes.
    
    Args:
        data (Dict[str, Any]): Dictionary payload.
        file_path (Path): Path to output file.
        indent (int): Indentation level.
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, cls=CustomJSONEncoder, indent=indent)
    except Exception as e:
        raise FusionError(f"Failed to write JSON file to {file_path}. Error: {e}")


def read_json_file(file_path: Path) -> Dict[str, Any]:
    """
    Reads a dictionary payload from a JSON file.
    
    Args:
        file_path (Path): Path to input file.
        
    Returns:
        Dict[str, Any]: Parsed JSON data.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise FusionError(f"Failed to load JSON file from {file_path}. Error: {e}")


def get_module_logger(name: str) -> Logger:
    """
    Retrieves a logger configured for a specific module inside Fusion.
    """
    return getLogger(f"fusion.{name}")


if __name__ == "__main__":
    print("Executing self-test for fusion/utils.py...")
    import tempfile
    
    try:
        # 1. Test UTC Timestamp
        ts = get_utc_timestamp()
        print(f"Generated Timestamp: {ts}")
        assert "+00:00" in ts or "Z" in ts or ts.endswith("00")
        
        # 2. Test boundary validation
        validate_score_and_confidence(0.95, 0.90, "TestModality")
        
        try:
            validate_score_and_confidence(-0.5, 0.90, "TestModality")
            print("Validation FAILED: Accepted negative score value.", file=sys.stderr)
            sys.exit(1)
        except InvalidDataError:
            print("Boundary check 1: PASSED")
            
        try:
            validate_score_and_confidence(0.5, 1.20, "TestModality")
            print("Validation FAILED: Accepted out of range confidence score.", file=sys.stderr)
            sys.exit(1)
        except InvalidDataError:
            print("Boundary check 2: PASSED")
            
        # 3. Test Custom JSON Serialization
        temp_dir = Path(tempfile.mkdtemp())
        test_file = temp_dir / "test_output.json"
        
        payload = {
            "path_value": Path("/foo/bar"),
            "datetime_value": datetime(2026, 8, 5, 12, 0, 0, tzinfo=timezone.utc),
            "score": 0.85
        }
        
        write_json_file(payload, test_file)
        print("JSON write: PASSED")
        
        loaded = read_json_file(test_file)
        print(f"Loaded JSON payload: {loaded}")
        assert Path(loaded["path_value"]) == Path("/foo/bar")
        assert "2026-08-05T12:00:00" in loaded["datetime_value"]
        assert loaded["score"] == 0.85
        
        # Cleanup
        test_file.unlink()
        temp_dir.rmdir()
        print("JSON read & cleanup: PASSED")
        
        # 4. Test Logger
        log = get_module_logger("test")
        assert log.name == "fusion.test"
        print("Logger acquisition: PASSED")
        
        print("All self-tests completed successfully.")
        
    except Exception as e:
        print(f"Self-test failed with error: {e}", file=sys.stderr)
        sys.exit(1)
