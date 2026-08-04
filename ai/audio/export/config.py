"""Configuration module for the Authentix Audio AI Model Export Pipeline.

This module defines configuration classes (using dataclasses) for model checkpoints,
ONNX export settings, logging parameters, opset versions, and dynamic axes shapes.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Union


@dataclass(frozen=True)
class LoggingConfig:
    """Logger configurations for model export and verification."""

    log_file: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "logs" / "model_export.log"
    )
    log_level: int = logging.INFO


@dataclass(frozen=True)
class ONNXConfig:
    """ONNX runtime and export configurations."""

    opset_version: int = 17  # Standard opset version supporting audio transformers
    
    # Define dynamic axes to allow variable batch size and sequence lengths during inference
    dynamic_axes: Dict[str, Dict[int, str]] = field(
        default_factory=lambda: {
            "input_values": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"}
        }
    )


@dataclass(frozen=True)
class ExportConfig:
    """Top-level configuration settings for model export execution."""

    # Sub-configuration groups
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    onnx: ONNXConfig = field(default_factory=ONNXConfig)

    # Input paths
    checkpoint_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "checkpoints" / "best_model.pth"
    )
    backbone_name: str = "MelodyMachine/Deepfake-audio-detection-V2"

    # Output paths
    output_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[3] / "exports"
    )
    model_name: str = "audio_model.onnx"
    export_report_name: str = "export_report.json"
    verification_report_name: str = "verification_report.json"

    # Evaluation verification tolerances
    abs_tolerance: float = 1e-4
    rel_tolerance: float = 1e-3

    @property
    def onnx_model_path(self) -> Path:
        """Returns the full target path to the exported ONNX model."""
        return self.output_dir / self.model_name

    @property
    def export_report_path(self) -> Path:
        """Returns the full target path to the export JSON report."""
        return self.output_dir / self.export_report_name

    @property
    def verification_report_path(self) -> Path:
        """Returns the full target path to the verification JSON report."""
        return self.output_dir / self.verification_report_name

    def validate(self) -> None:
        """Validates input path requirements.

        Raises:
            FileNotFoundError: If the source checkpoint weights file does not exist.
        """
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Trained model checkpoint not found at: {self.checkpoint_path}")

    def create_directories(self) -> None:
        """Creates target directories for export outputs and logging files."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logging.log_file.parent.mkdir(parents=True, exist_ok=True)


def setup_logger(name: str, log_file: Union[str, Path], level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a file and console logger for the export pipeline.

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
