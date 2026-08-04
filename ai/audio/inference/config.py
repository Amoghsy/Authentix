"""Configuration module for the Authentix Audio AI Production Inference Engine.

This module defines the InferenceConfig dataclass containing paths to the exported
ONNX model, confidence classification boundaries, and device/logging properties.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union


@dataclass(frozen=True)
class InferenceConfig:
    """Production configuration settings for standalone audio inference execution."""

    # Project structure root paths
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[3]
    )

    # Model configuration
    model_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[3] / "exports" / "audio_model.onnx"
    )
    backbone_name: str = "MelodyMachine/Deepfake-audio-detection-V2"
    
    # Classification decision boundary
    confidence_threshold: float = 0.5

    # Device selection ('cuda' or 'cpu')
    device: str = "cuda"

    # Logging config
    log_file: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "logs" / "audio_inference.log"
    )
    log_level: int = logging.INFO

    # Outputs
    output_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "inference" / "results"
    )

    def validate(self) -> None:
        """Validates configuration parameters and checks ONNX model existence.

        Raises:
            FileNotFoundError: If the ONNX model file does not exist.
            ValueError: If the confidence threshold is out of range.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Exported ONNX model not found at: {self.model_path}")

        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError(f"Confidence threshold must be between 0.0 and 1.0. Found: {self.confidence_threshold}")

    def create_directories(self) -> None:
        """Creates target directories for logs and results outputs."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)


def setup_logger(name: str, log_file: Union[str, Path], level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a file and console logger for the inference engine.

    Args:
        name: Name of the logger.
        log_file: Path to the log file location.
        level: Logger level.

    Returns:
        logging.Logger: The configured Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if setup is called multiple times
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
        )

        # File Handler
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
