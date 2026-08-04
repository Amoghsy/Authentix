"""Configuration management for the Authentix Audio Preprocessing Pipeline.

This module provides dataclass-based configuration settings, including path structures,
audio parameters, dataset splits, logging setup, directory creation utilities, and
configuration validation to ensure correct pipeline execution.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Union


@dataclass(frozen=True)
class AudioPreprocessingConfig:
    """Enterprise-grade configuration settings for the Audio Preprocessing Pipeline.

    All paths are stored as Path objects. Audio parameters specify targets for Wav2Vec2.
    """

    # --- Project & Directory Structure Paths ---
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[3]
    )

    # Base audio pipeline directories
    audio_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1]
    )

    dataset_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "dataset"
    )

    # Input (Raw) and Output (Processed/Split) Audio Paths
    raw_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "dataset" / "deepfake audio"
    )
    processed_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "dataset" / "processed"
    )
    train_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "dataset" / "train"
    )
    validation_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "dataset" / "validation"
    )
    test_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "dataset" / "test"
    )
    logs_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "logs"
    )

    # --- Preprocessing Artifacts Paths ---
    manifest_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "dataset" / "dataset_manifest.csv"
    )
    statistics_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "dataset" / "dataset_statistics.json"
    )
    summary_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "dataset" / "dataset_summary.md"
    )
    label_mapping_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "dataset" / "label_mapping.json"
    )
    preprocessing_report_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "dataset" / "preprocessing_report.md"
    )
    conversion_report_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "dataset" / "conversion_report.json"
    )

    # --- Audio Quality & Formatting Specs ---
    sample_rate: int = 16000  # Target sample rate (16 kHz for Wav2Vec2)
    target_channels: int = 1  # 1 = Mono, 2 = Stereo
    bit_depth: int = 16  # 16-bit PCM WAV
    audio_extensions: Set[str] = field(
        default_factory=lambda: {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
    )
    min_duration_seconds: float = 0.1  # Rejects files shorter than this
    clipping_threshold: float = 0.999  # Absolute maximum amplitude threshold to flag clipping

    # --- Dataset Partition Ratios ---
    train_split: float = 0.80
    validation_split: float = 0.10
    test_split: float = 0.10
    random_seed: int = 42

    # --- Process Configuration ---
    num_workers: int = 4  # Default number of workers for multiprocessing

    def validate(self) -> None:
        """Validates configuration parameter values and split configurations.

        Raises:
            ValueError: If splits do not sum to 1.0, values are negative, or format is invalid.
        """
        # Split ratio validation
        total_split = self.train_split + self.validation_split + self.test_split
        if not abs(total_split - 1.0) < 1e-6:
            raise ValueError(
                f"Split ratios (train: {self.train_split}, validation: {self.validation_split}, "
                f"test: {self.test_split}) must sum to 1.0. Found: {total_split}"
            )

        if self.train_split < 0 or self.validation_split < 0 or self.test_split < 0:
            raise ValueError("Split ratios cannot be negative values.")

        # Audio spec validation
        if self.sample_rate <= 0:
            raise ValueError(f"Sample rate must be a positive integer. Found: {self.sample_rate}")

        if self.target_channels <= 0:
            raise ValueError(f"Target channels must be a positive integer. Found: {self.target_channels}")

        if self.min_duration_seconds <= 0:
            raise ValueError(f"Minimum duration must be a positive number. Found: {self.min_duration_seconds}")

        if not (0.0 < self.clipping_threshold <= 1.0):
            raise ValueError(f"Clipping threshold must be between 0.0 and 1.0. Found: {self.clipping_threshold}")

        # Extensions check
        for ext in self.audio_extensions:
            if not ext.startswith("."):
                raise ValueError(f"Audio extensions must start with a dot. Invalid extension: '{ext}'")

    def create_directories(self) -> None:
        """Creates all required pipeline directories if they do not exist."""
        directories_to_create = [
            self.processed_dir,
            self.train_dir / "real",
            self.train_dir / "fake",
            self.validation_dir / "real",
            self.validation_dir / "fake",
            self.test_dir / "real",
            self.test_dir / "fake",
            self.logs_dir,
        ]
        for directory in directories_to_create:
            directory.mkdir(parents=True, exist_ok=True)


def setup_logger(name: str, log_file: Union[str, Path], level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a file and console logger for the preprocessing pipeline.

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
