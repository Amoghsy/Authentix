"""
config.py

This module contains configuration dataclasses and initialization/validation utilities 
for the Authentix Lip Sync preprocessing and model integration pipeline. It defines 
paths, pretrained model download configurations, logging specifications, video loading 
parameters, audio feature extraction standards, mouth cropping settings, and a global 
random seed.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import sys
from typing import Set, Optional, Tuple
import torch


@dataclass(frozen=True)
class PathConfig:
    """
    Stores paths for the project workspace, datasets, logging, and model checkpoints.
    All paths are absolute and resolved relative to this configuration file.
    """
    project_root: Path
    lip_sync_root: Path
    dataset_dir: Path
    checkpoints_dir: Path
    logs_dir: Path
    models_dir: Path


@dataclass(frozen=True)
class ModelConfig:
    """
    Stores Hugging Face download paths and execution device configurations for 
    pretrained auxiliary weights (SFD face detector and SyncNet evaluation model).
    """
    sfd_weights_name: str = "sfd_face.pth"
    sfd_weights_url: str = "https://huggingface.co/lithiumice/syncnet/resolve/main/sfd_face.pth"
    syncnet_weights_name: str = "syncnet_v2.model"
    syncnet_weights_url: str = "https://huggingface.co/lithiumice/syncnet/resolve/main/syncnet_v2.model"
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )


@dataclass(frozen=True)
class LoggingConfig:
    """
    Configuration parameters for Python's logging module.
    """
    log_file_name: str = "lip_sync_preprocessing.log"
    log_level: int = logging.INFO
    log_format: str = "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class VideoConfig:
    """
    Validation standards and frame extraction configurations for incoming video streams.
    """
    target_fps: float = 25.0  # SyncNet expects 25 FPS video sequences
    supported_extensions: Set[str] = field(
        default_factory=lambda: {".mp4", ".avi", ".mov", ".mkv"}
    )
    min_duration: float = 0.2  # 5 frames at 25 fps minimum required duration
    max_duration: Optional[float] = None
    min_resolution: Tuple[int, int] = (128, 128)  # Minimum (Width, Height)


@dataclass(frozen=True)
class AudioConfig:
    """
    Feature extraction parameters for the audio stream to be input to SyncNet.
    """
    sample_rate: int = 16000  # SyncNet expects 16 kHz audio
    channels: int = 1         # Mono
    normalize: bool = True     # Peak normalization
    num_mfcc: int = 13        # Number of Mel-Frequency Cepstral Coefficients
    mfcc_fps: float = 100.0   # 100 audio feature frames per second (matching 20 frames per 5 video frames)


@dataclass(frozen=True)
class LipConfig:
    """
    Specifications for extracting and scaling the mouth visual region of interest (ROI).
    """
    crop_size: int = 111       # Visual CNN input width/height is 111x111 pixels
    padding: float = 0.15      # Percentage padding relative to mouth bounding box
    sequence_length: int = 5   # Number of consecutive frames per evaluation window
    grayscale: bool = True     # SyncNet processes mouth crops in grayscale


@dataclass(frozen=True)
class ProjectConfig:
    """
    Root configuration containing all sub-configurations and the random seed.
    """
    paths: PathConfig
    models: ModelConfig = field(default_factory=ModelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    lip: LipConfig = field(default_factory=LipConfig)
    random_seed: int = 42


def get_default_config() -> ProjectConfig:
    """
    Initializes and returns the ProjectConfig with resolved absolute paths.
    config.py lies in project_root/ai/lip_sync/preprocessing/config.py.
    """
    # parents[3] points to the project_root directory (Authentix)
    project_root = Path(__file__).resolve().parents[3]
    lip_sync_root = project_root / "ai" / "lip_sync"
    
    path_config = PathConfig(
        project_root=project_root,
        lip_sync_root=lip_sync_root,
        dataset_dir=lip_sync_root / "dataset",
        checkpoints_dir=lip_sync_root / "checkpoints",
        logs_dir=lip_sync_root / "logs",
        models_dir=lip_sync_root / "models",
    )
    
    return ProjectConfig(paths=path_config)


def validate_config(config: ProjectConfig) -> None:
    """
    Validates configuration values to enforce logical consistency and compatibility with SyncNet.
    
    Args:
        config (ProjectConfig): The project configuration to validate.
        
    Raises:
        ValueError: If configuration validation constraints are breached.
    """
    # Video validations
    if config.video.target_fps <= 0:
        raise ValueError(f"Target FPS must be positive. Provided: {config.video.target_fps}")
    if config.video.min_duration <= 0:
        raise ValueError(f"Minimum duration must be positive. Provided: {config.video.min_duration}")
    if len(config.video.supported_extensions) == 0:
        raise ValueError("Video supported extensions set cannot be empty.")
        
    # Audio validations
    if config.audio.sample_rate <= 0:
        raise ValueError(f"Audio sample rate must be positive. Provided: {config.audio.sample_rate}")
    if config.audio.channels <= 0:
        raise ValueError(f"Audio channels must be positive. Provided: {config.audio.channels}")
    if config.audio.num_mfcc <= 0:
        raise ValueError(f"MFCC count must be positive. Provided: {config.audio.num_mfcc}")
    if config.audio.mfcc_fps <= 0:
        raise ValueError(f"MFCC FPS rate must be positive. Provided: {config.audio.mfcc_fps}")
        
    # Lip ROI validations
    if config.lip.crop_size <= 0:
        raise ValueError(f"Crop size must be positive. Provided: {config.lip.crop_size}")
    if config.lip.padding < 0:
        raise ValueError(f"ROI padding cannot be negative. Provided: {config.lip.padding}")
    if config.lip.sequence_length <= 0:
        raise ValueError(f"Sequence length must be positive. Provided: {config.lip.sequence_length}")


def setup_directories(config: ProjectConfig) -> None:
    """
    Creates the project directories for dataset, checkpoints, logs, and models.
    
    Args:
        config (ProjectConfig): The configuration containing path variables.
    """
    paths_to_create = [
        config.paths.dataset_dir,
        config.paths.checkpoints_dir,
        config.paths.logs_dir,
        config.paths.models_dir,
    ]
    
    for path in paths_to_create:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            # Print to stdout as logger might not be initialized
            print(f"CRITICAL: Failed to create directory at {path}. Error: {e}", file=sys.stderr)
            raise e


def setup_logging(config: ProjectConfig) -> None:
    """
    Configures python's logging module to route events to console and file.
    
    Args:
        config (ProjectConfig): The project configuration.
    """
    # Ensure logs folder exists
    config.paths.logs_dir.mkdir(parents=True, exist_ok=True)
    log_filepath = config.paths.logs_dir / config.logging.log_file_name
    
    log_formatter = logging.Formatter(
        config.logging.log_format,
        datefmt=config.logging.date_format
    )
    
    root_logger = logging.getLogger()
    root_logger.setLevel(config.logging.log_level)
    
    # Remove existing handlers to avoid duplicates
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    # File Handler
    file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(config.logging.log_level)
    root_logger.addHandler(file_handler)
    
    # Console Stream Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(config.logging.log_level)
    root_logger.addHandler(console_handler)
    
    logging.info(f"Initialized Logging. Logs written to {log_filepath}")


if __name__ == "__main__":
    print("Executing self-test for config.py...")
    try:
        cfg = get_default_config()
        print(f"Project root resolved to: {cfg.paths.project_root}")
        print(f"Lip sync root resolved to: {cfg.paths.lip_sync_root}")
        print(f"Execution device: {cfg.models.device}")
        
        validate_config(cfg)
        print("Validation checks: PASSED")
        
        setup_directories(cfg)
        print("Directory setup: PASSED")
        
        setup_logging(cfg)
        logging.info("Logging configuration self-test: PASSED")
        print("All self-tests completed successfully.")
    except Exception as e:
        print(f"Self-test failed: {e}", file=sys.stderr)
        sys.exit(1)
