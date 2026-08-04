"""
config.py

This module contains configuration dataclasses and utilities for the Authentix video
preprocessing pipeline. It defines all required project paths, image size constraints,
frame extraction settings, random seed, dataset split details, logging configurations,
validation rules, and folder structure initializations.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Tuple, Set, Optional

@dataclass(frozen=True)
class PathConfig:
    """
    Stores paths for the project workspace, datasets, logging, and model checkpoints.
    Paths are absolute and resolved relative to this config file's directory.
    """
    project_root: Path
    video_root: Path
    dataset_dir: Path
    raw_dir: Path
    processed_dir: Path
    train_dir: Path
    val_dir: Path
    test_dir: Path
    logs_dir: Path
    checkpoints_dir: Path
    exports_dir: Path


@dataclass(frozen=True)
class PipelineConfig:
    """
    Configuration parameters for frame extraction, face detection, cropping, and scaling.
    """
    image_size: Tuple[int, int] = (224, 224)
    fps: float = 2.0
    supported_extensions: Set[str] = field(
        default_factory=lambda: {".mp4", ".avi", ".mov", ".mkv"}
    )
    min_face_size: int = 50          # Minimum width or height in pixels to consider a face valid
    face_padding: float = 0.15        # Percentage of padding added around the face bounding box
    mp_detection_confidence: float = 0.5 # Minimum confidence for MediaPipe Face Detection
    max_videos: Optional[int] = None   # Limit the number of videos to process for test/dry-runs
    class_filter: Optional[str] = None # Filter to process only a specific class (e.g. 'real' or 'fake')


@dataclass(frozen=True)
class TrainingConfig:
    """
    Configuration parameters for model training, optimization, and learning rate schedulers.
    """
    batch_size: int = 32
    epochs_phase1: int = 5              # Classifier fine-tuning (frozen backbone)
    epochs_phase2: int = 10             # Fine-tuning of last MobileNet blocks
    epochs_phase3: int = 5              # Fine-tuning of entire network (optional)
    lr_phase1: float = 1e-3             # Learning rate for classifier training
    lr_phase2: float = 1e-4             # Learning rate for block fine-tuning
    lr_phase3: float = 1e-5             # Learning rate for full fine-tuning
    weight_decay: float = 1e-4
    dropout: float = 0.2
    num_workers: int = 4
    use_mixed_precision: bool = True
    auto_class_weights: bool = True
    lr_scheduler_type: str = "cosine"   # "cosine", "plateau", or "none"
    early_stopping_patience: int = 5


@dataclass(frozen=True)
class SplitConfig:
    """
    Ratios used for splitting the processed dataset into train, validation, and test sets.
    """
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15


@dataclass(frozen=True)
class AppConfig:
    """
    Root configuration object containing paths, pipeline params, split ratios, training parameters, and global random seed.
    """
    paths: PathConfig
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    random_seed: int = 42


def get_default_config() -> AppConfig:
    """
    Initializes and returns the default AppConfig with resolved absolute paths.
    
    Returns:
        AppConfig: The default application configuration.
    """
    # config.py lies in project_root/ai/video/preprocessing/config.py
    # parents[3] points to the project_root directory (Authentix)
    project_root = Path(__file__).resolve().parents[3]
    video_root = project_root / "ai" / "video"
    dataset_dir = video_root / "dataset"
    
    path_config = PathConfig(
        project_root=project_root,
        video_root=video_root,
        dataset_dir=dataset_dir,
        raw_dir=dataset_dir / "raw",
        processed_dir=dataset_dir / "processed",
        train_dir=dataset_dir / "train",
        val_dir=dataset_dir / "validation",
        test_dir=dataset_dir / "test",
        logs_dir=video_root / "logs",
        checkpoints_dir=video_root / "checkpoints",
        exports_dir=video_root / "exports",
    )
    
    return AppConfig(paths=path_config)


def validate_config(config: AppConfig) -> None:
    """
    Validates the configuration parameters.
    
    Args:
        config (AppConfig): The application configuration to validate.
        
    Raises:
        ValueError: If any validation rule is violated.
    """
    # Validate splits sum to 1.0 (with floating point tolerance)
    split_sum = config.split.train + config.split.val + config.split.test
    if abs(split_sum - 1.0) > 1e-5:
        raise ValueError(
            f"Dataset splits (train: {config.split.train}, val: {config.split.val}, "
            f"test: {config.split.test}) must sum to 1.0, but sum is {split_sum:.5f}"
        )
        
    # Validate FPS is positive
    if config.pipeline.fps <= 0:
        raise ValueError(f"FPS must be positive. Provided FPS: {config.pipeline.fps}")
        
    # Validate image size dimensions
    h, w = config.pipeline.image_size
    if h <= 0 or w <= 0:
        raise ValueError(f"Image dimensions must be positive. Provided size: ({h}, {w})")
        
    # Validate face padding is non-negative
    if config.pipeline.face_padding < 0:
        raise ValueError(f"Face padding cannot be negative. Provided padding: {config.pipeline.face_padding}")

    # Validate MediaPipe detection confidence
    conf = config.pipeline.mp_detection_confidence
    if not (0.0 <= conf <= 1.0):
        raise ValueError(f"MediaPipe detection confidence must be between 0.0 and 1.0. Provided: {conf}")

    # Validate TrainingConfig parameters
    if config.training.batch_size <= 0:
        raise ValueError(f"Batch size must be positive. Provided: {config.training.batch_size}")
    if config.training.epochs_phase1 < 0 or config.training.epochs_phase2 < 0 or config.training.epochs_phase3 < 0:
        raise ValueError("Epoch counts cannot be negative.")
    if config.training.lr_phase1 <= 0 or config.training.lr_phase2 <= 0 or config.training.lr_phase3 <= 0:
        raise ValueError("Learning rates must be positive.")
    if config.training.lr_scheduler_type not in {"cosine", "plateau", "none"}:
        raise ValueError(f"Invalid learning rate scheduler type: {config.training.lr_scheduler_type}")


def setup_directories(config: AppConfig) -> None:
    """
    Creates all necessary project directories defined in the path configuration.
    
    Args:
        config (AppConfig): The application configuration.
    """
    paths_to_create = [
        config.paths.dataset_dir,
        config.paths.raw_dir,
        config.paths.processed_dir,
        config.paths.train_dir,
        config.paths.val_dir,
        config.paths.test_dir,
        config.paths.logs_dir,
        config.paths.checkpoints_dir,
        config.paths.exports_dir,
    ]
    
    for path in paths_to_create:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            # We fail gracefully but print since logger may not be set up yet
            print(f"CRITICAL: Failed to create directory at {path}. Error: {e}")
            raise e


def setup_logging(config: AppConfig) -> None:
    """
    Configures python's logging system to write to both stdout and a log file in the logs directory.
    
    Args:
        config (AppConfig): The application configuration.
    """
    # Ensure the logs directory is created first
    config.paths.logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = config.paths.logs_dir / "video_preprocessing.log"
    
    # Configure formatter
    log_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove existing handlers to prevent duplicates if function is called multiple times
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    
    # Console stream handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    
    logging.info(f"Logging initialized. Log file path: {log_file}")


if __name__ == "__main__":
    # Self-test block to verify configuration setup and validation
    print("Testing config loading and initialization...")
    try:
        app_config = get_default_config()
        print(f"Loaded config. Project Root: {app_config.paths.project_root}")
        
        # Test Validation
        validate_config(app_config)
        print("Validation successful!")
        
        # Test Directory Creation
        setup_directories(app_config)
        print("Required directories created successfully.")
        
        # Test Logging Setup
        setup_logging(app_config)
        logging.info("Configuration module verification finished successfully.")
    except Exception as e:
        print(f"Verification failed: {e}")
        raise e
