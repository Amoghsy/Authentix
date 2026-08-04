"""Configuration module for the Authentix Audio AI Training Pipeline.

This module defines configuration classes (using dataclasses) for all hyperparameters,
model options, optimization routines, learning rate scheduling, checkpoints, TensorBoard logging,
and logging options required for audio deepfake classification training.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union


@dataclass(frozen=True)
class ModelConfig:
    """Configuration options for the HuggingFace audio model backbone."""

    # Name or local path to the HuggingFace pretrained backbone
    backbone_name: str = "MelodyMachine/Deepfake-audio-detection-V2"
    num_labels: int = 2
    # Classifier dropout probability
    classifier_dropout: float = 0.1
    # Number of encoder layers to unfreeze in Phase 2
    unfreeze_layers_count: int = 2


@dataclass(frozen=True)
class OptimizerConfig:
    """Hyperparameters for model optimization using AdamW."""

    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8


@dataclass(frozen=True)
class SchedulerConfig:
    """Hyperparameters for learning rate scheduling."""

    # Options: 'cosine', 'linear', 'step', None
    scheduler_type: str = "cosine"
    # Minimum learning rate for cosine annealing
    min_lr: float = 1e-6
    # Epoch count step interval (if using step scheduler)
    step_size: int = 5
    # Decay rate (if using step scheduler)
    gamma: float = 0.5


@dataclass(frozen=True)
class CheckpointConfig:
    """Configuration settings for checkpoint creation and model tracking."""

    checkpoint_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "checkpoints"
    )
    save_every_epoch: bool = True
    # Metric used to decide which checkpoint is the "best" ('loss', 'accuracy', 'f1')
    monitor_metric: str = "val_loss"
    # Mode ('min' for loss, 'max' for accuracy/f1)
    monitor_mode: str = "min"


@dataclass(frozen=True)
class TensorBoardConfig:
    """Configuration settings for TensorBoard metrics visualization."""

    log_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "logs" / "tensorboard"
    )
    # Step interval to log scalar metrics (e.g. loss) during training
    log_step_interval: int = 10


@dataclass(frozen=True)
class LoggingConfig:
    """Logger configurations for training console and file logging."""

    log_file: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "logs" / "audio_training.log"
    )
    log_level: int = logging.INFO


@dataclass(frozen=True)
class TrainingConfig:
    """Top-level configuration class managing the training pipeline execution configurations."""

    # Sub-configuration groups
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    tensorboard: TensorBoardConfig = field(default_factory=TensorBoardConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Dataset paths
    manifest_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "dataset" / "dataset_manifest.csv"
    )
    label_mapping_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "dataset" / "label_mapping.json"
    )

    # Staged learning phase configurations
    # Phase 1: Classifier Training (Backbone completely frozen)
    epochs_phase1: int = 5
    lr_phase1: float = 1e-3

    # Phase 2: Unfreeze last N encoder layers + Fine-tune
    epochs_phase2: int = 10
    lr_phase2: float = 5e-5

    # Phase 3: Fine-tune entire model
    epochs_phase3: int = 5
    lr_phase3: float = 1e-5

    # Global runtime settings
    random_seed: int = 42
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    mixed_precision: bool = True
    early_stopping_patience: int = 5
    
    # Class weights balancing options: 'auto' (computes inverse class counts), 'none', or a tuple [w0, w1]
    class_weights_strategy: str = "auto"
    
    # Hardware resources
    device: str = "cuda"  # Will fallback to cpu dynamically in script if CUDA is unavailable
    dataloader_num_workers: int = 2
    pin_memory: bool = True

    def validate(self) -> None:
        """Validates configuration values for range correctness.

        Raises:
            ValueError: If parameters are invalid.
        """
        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be a positive integer. Found: {self.batch_size}")
            
        if self.epochs_phase1 < 0 or self.epochs_phase2 < 0 or self.epochs_phase3 < 0:
            raise ValueError("Epoch counts cannot be negative.")
            
        if self.lr_phase1 <= 0 or self.lr_phase2 <= 0 or self.lr_phase3 <= 0:
            raise ValueError("Learning rates must be positive floats.")

        if self.gradient_accumulation_steps <= 0:
            raise ValueError("Gradient accumulation steps must be a positive integer.")

        if self.early_stopping_patience <= 0:
            raise ValueError("Early stopping patience must be a positive integer.")

        if self.checkpoint.monitor_mode not in {"min", "max"}:
            raise ValueError(f"Monitor mode must be 'min' or 'max'. Found: {self.checkpoint.monitor_mode}")

        if self.scheduler.scheduler_type not in {"cosine", "linear", "step", "none", None}:
            raise ValueError(f"Unknown scheduler type: {self.scheduler.scheduler_type}")

    def create_directories(self) -> None:
        """Creates output directories for checkpoints, logs, and TensorBoard logs."""
        self.checkpoint.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard.log_dir.mkdir(parents=True, exist_ok=True)
        self.logging.log_file.parent.mkdir(parents=True, exist_ok=True)


def setup_logger(name: str, log_file: Union[str, Path], level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a file and console logger for the training pipeline.

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
