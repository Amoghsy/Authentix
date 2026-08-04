"""Configuration module for the Authentix Video AI Production Inference Engine.

This module defines configuration classes (using dataclasses) for video loading,
ONNX model options, logging setup, threshold bounds, and visualizer outputs.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Union


@dataclass(frozen=True)
class LoggingConfig:
    """Logger configurations for video inference execution."""

    log_file: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "logs" / "video_inference.log"
    )
    log_level: int = logging.INFO


@dataclass(frozen=True)
class ONNXConfig:
    """ONNX Runtime hardware acceleration configurations."""

    device: str = "cuda"  # Dynamic fallback to CPU in runner if unavailable or fails
    intra_op_num_threads: int = 0
    inter_op_num_threads: int = 0


@dataclass(frozen=True)
class PredictionConfig:
    """Model inference and frame processing configurations."""

    confidence_threshold: float = 0.5
    
    # Process every N-th frame of the video
    frame_sampling_rate: int = 1
    
    # Maximum number of frames to extract and process per video
    max_frames: int = 32
    
    # Model input crop resolution (Height, Width)
    input_size: Tuple[int, int] = (224, 224)


@dataclass(frozen=True)
class VisualizationConfig:
    """Visual prediction diagnostic plot configurations."""

    enable_plots: bool = True
    output_viz_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "inference" / "results" / "visualizations"
    )


@dataclass(frozen=True)
class InferenceConfig:
    """Top-level configuration management wrapper for video deepfake inference."""

    # Sub-configuration groups
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    onnx: ONNXConfig = field(default_factory=ONNXConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    # Input paths
    model_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "exports" / "deepfake_model.onnx"
    )

    # Output paths
    output_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "inference" / "results"
    )
    predictions_csv: str = "predictions.csv"
    summary_json: str = "summary.json"

    def validate(self) -> None:
        """Validates inference configuration paths and parameters.

        Raises:
            FileNotFoundError: If the ONNX model path does not exist.
            ValueError: If configurations are out of valid bounds.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Exported Video ONNX model not found at: {self.model_path}")

        if not (0.0 <= self.prediction.confidence_threshold <= 1.0):
            raise ValueError(
                f"Confidence threshold must be between 0.0 and 1.0. Found: {self.prediction.confidence_threshold}"
            )

        if self.prediction.max_frames <= 0:
            raise ValueError(f"max_frames must be a positive integer. Found: {self.prediction.max_frames}")

        if self.prediction.frame_sampling_rate <= 0:
            raise ValueError(
                f"frame_sampling_rate must be a positive integer. Found: {self.prediction.frame_sampling_rate}"
            )

    def create_directories(self) -> None:
        """Initializes logging, reports, and visualization folders on disk."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.visualization.output_viz_dir.mkdir(parents=True, exist_ok=True)
        self.logging.log_file.parent.mkdir(parents=True, exist_ok=True)


def setup_logger(name: str, log_file: Union[str, Path], level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a file and console logger for the video inference engine.

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
