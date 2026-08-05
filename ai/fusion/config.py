"""
config.py

This module contains configuration dataclasses and utilities for the Authentix Fusion Engine.
It defines configurable score weights (Video, Audio, Lip Sync), classification thresholds,
risk tier boundaries, logging settings, report file naming, and directory setups.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import sys
from typing import Optional


@dataclass(frozen=True)
class ScoreWeights:
    """
    Stores weights used for score aggregation. 
    Weights represent contribution percentages of each model to the final fusion score.
    Must sum to 1.0.
    """
    video: float = 0.50
    audio: float = 0.30
    lip_sync: float = 0.20


@dataclass(frozen=True)
class ThresholdConfig:
    """
    Contains classification decisions thresholds.
    Scores above this boundary are classified as Deepfakes.
    """
    decision_threshold: float = 0.50


@dataclass(frozen=True)
class RiskLevelConfig:
    """
    Defines classification scores ranges mapping to specific risk tiers:
    Low, Medium, High, and Critical.
    """
    low: float = 0.30       # Less than 0.30 = Low
    medium: float = 0.60    # Less than 0.60 = Medium
    high: float = 0.85      # Less than 0.85 = High, >= 0.85 = Critical


@dataclass(frozen=True)
class LoggingConfig:
    """
    Configuration parameters for Python's logging module inside Fusion.
    """
    log_file_name: str = "fusion_engine.log"
    log_level: int = logging.INFO
    log_format: str = "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class ReportConfig:
    """
    Paths and names for generated reports (JSON, Markdown, HTML).
    """
    output_dir: Path
    json_report_name: str = "fusion_result.json"
    md_report_name: str = "fusion_report.md"
    html_report_name: str = "fusion_report.html"


@dataclass(frozen=True)
class FusionConfig:
    """
    Root configuration containing all sub-configurations.
    """
    project_root: Path
    fusion_root: Path
    weights: ScoreWeights = field(default_factory=ScoreWeights)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    risk_levels: RiskLevelConfig = field(default_factory=RiskLevelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    reports: Optional[ReportConfig] = None


def get_default_config() -> FusionConfig:
    """
    Resolves directories and instantiates a default FusionConfig object.
    config.py resides under project_root/ai/fusion/config.py.
    """
    # parents[2] points to the project_root directory (Authentix)
    project_root = Path(__file__).resolve().parents[2]
    fusion_root = project_root / "ai" / "fusion"
    
    report_config = ReportConfig(
        output_dir=fusion_root / "reports"
    )
    
    return FusionConfig(
        project_root=project_root,
        fusion_root=fusion_root,
        reports=report_config
    )


def validate_config(config: FusionConfig) -> None:
    """
    Enforces rules on configuration parameters.
    
    Raises:
        ValueError: If weights do not sum to 1.0 or thresholds are out of range.
    """
    # 1. Weights Validation
    w_sum = config.weights.video + config.weights.audio + config.weights.lip_sync
    if abs(w_sum - 1.0) > 1e-5:
        raise ValueError(
            f"Fusion weights must sum to 1.0. Currently sums to: {w_sum:.4f} "
            f"(Video={config.weights.video}, Audio={config.weights.audio}, LipSync={config.weights.lip_sync})"
        )
    if config.weights.video < 0 or config.weights.audio < 0 or config.weights.lip_sync < 0:
        raise ValueError("Weights cannot be negative.")
        
    # 2. Decision Threshold Validation
    if not (0.0 <= config.thresholds.decision_threshold <= 1.0):
        raise ValueError(f"Decision threshold must be in range [0.0, 1.0]. Provided: {config.thresholds.decision_threshold}")
        
    # 3. Risk Thresholds Validation
    rl = config.risk_levels
    if not (0.0 <= rl.low <= rl.medium <= rl.high <= 1.0):
        raise ValueError(
            f"Risk thresholds must be in ascending order and in range [0.0, 1.0]. "
            f"Provided: Low={rl.low}, Medium={rl.medium}, High={rl.high}"
        )


def setup_directories(config: FusionConfig) -> None:
    """
    Initializes required directories.
    """
    paths_to_create = [
        config.fusion_root,
    ]
    if config.reports is not None:
        paths_to_create.append(config.reports.output_dir)
        
    for path in paths_to_create:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"CRITICAL: Failed to create directory at {path}. Error: {e}", file=sys.stderr)
            raise e


def setup_logging(config: FusionConfig) -> None:
    """
    Configures Python's root logging structure to print to stdout and log files.
    """
    # Logs directory under project_root/ai/fusion/logs/
    logs_dir = config.fusion_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_filepath = logs_dir / config.logging.log_file_name
    
    log_formatter = logging.Formatter(
        config.logging.log_format,
        datefmt=config.logging.date_format
    )
    
    root_logger = logging.getLogger()
    root_logger.setLevel(config.logging.log_level)
    
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
    
    logging.info(f"Initialized Fusion Engine Logging. Output file: {log_filepath}")


if __name__ == "__main__":
    print("Executing self-test for fusion/config.py...")
    try:
        cfg = get_default_config()
        print(f"Resolved Project Root: {cfg.project_root}")
        print(f"Resolved Fusion Root: {cfg.fusion_root}")
        
        validate_config(cfg)
        print("Configuration validation: PASSED")
        
        setup_directories(cfg)
        print("Required directories creation: PASSED")
        
        setup_logging(cfg)
        logging.info("Fusion Logging configuration: PASSED")
        print("All self-tests completed successfully.")
    except Exception as e:
        print(f"Self-test failed with error: {e}", file=sys.stderr)
        sys.exit(1)
