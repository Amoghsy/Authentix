"""Configuration module for the Authentix Audio AI Testing & Evaluation Pipeline.

This module defines configuration classes (using dataclasses) for testing paths,
output report structures, logging levels, and hardware configurations.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Set, Union


@dataclass(frozen=True)
class LoggingConfig:
    """Logger configurations for testing console and file logging."""

    log_file: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "logs" / "audio_testing.log"
    )
    log_level: int = logging.INFO


@dataclass(frozen=True)
class ReportConfig:
    """Configuration settings for visual and tabular metric reports."""

    output_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "testing" / "results"
    )
    results_json: str = "test_results.json"
    classification_report: str = "classification_report.txt"
    confusion_matrix_img: str = "confusion_matrix.png"
    roc_curve_img: str = "roc_curve.png"
    precision_recall_curve_img: str = "precision_recall_curve.png"
    predictions_csv: str = "predictions.csv"
    evaluation_report_md: str = "evaluation_report.md"


@dataclass(frozen=True)
class DeviceConfig:
    """Hardware resource settings for testing runs."""

    device: str = "cuda"  # Will fallback to cpu dynamically in script if CUDA is unavailable
    mixed_precision: bool = True
    dataloader_num_workers: int = 2
    pin_memory: bool = True


@dataclass(frozen=True)
class TestingConfig:
    """Top-level configuration class for managing testing execution parameters."""

    # Sub-configuration groups
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)

    # Backbone path or identifier
    backbone_name: str = "MelodyMachine/Deepfake-audio-detection-V2"

    # Dataset paths
    manifest_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "dataset" / "dataset_manifest.csv"
    )
    label_mapping_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "dataset" / "label_mapping.json"
    )
    checkpoint_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "checkpoints" / "best_model.pth"
    )

    batch_size: int = 16

    def validate(self) -> None:
        """Validates configuration paths and parameter bounds.

        Raises:
            ValueError: If parameters are invalid.
            FileNotFoundError: If manifest or checkpoint path is missing.
        """
        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be a positive integer. Found: {self.batch_size}")

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Dataset manifest not found at: {self.manifest_path}")

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at: {self.checkpoint_path}")

    def create_directories(self) -> None:
        """Creates output directories for logs and reports."""
        self.report.output_dir.mkdir(parents=True, exist_ok=True)
        self.logging.log_file.parent.mkdir(parents=True, exist_ok=True)


def setup_logger(name: str, log_file: Union[str, Path], level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a file and console logger for the testing pipeline.

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
