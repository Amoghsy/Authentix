"""Callback management module for model checkpoints, logging, and early stopping.

This module implements EarlyStopping, CheckpointSaver (handling best/last model, optimizer,
scheduler, and JSON state file save/restore), and TensorBoardLogger to track metrics and loss
curves during training.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter

from ai.audio.training.config import TrainingConfig


class EarlyStopping:
    """Monitors a metric and signals training termination if improvement stalls."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "min"):
        """Initializes the EarlyStopping monitor.

        Args:
            patience: Number of epochs to wait for improvement before stopping.
            min_delta: Minimum change in monitored metric to qualify as an improvement.
            mode: Metric direction ('min' for loss, 'max' for accuracy/f1).
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode.lower()
        
        self.logger = logging.getLogger("audio_training.callbacks.early_stopping")
        
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

        if self.mode not in {"min", "max"}:
            raise ValueError(f"Mode must be 'min' or 'max'. Found: {self.mode}")

    def __call__(self, val_metric: float) -> bool:
        """Evaluates metric score and updates stopping state.

        Args:
            val_metric: Current epoch metric score.

        Returns:
            bool: True if early stopping should trigger, False otherwise.
        """
        if self.best_score is None:
            self.best_score = val_metric
            self.logger.info(f"Initialized EarlyStopping with baseline score: {val_metric:.6f}")
            return False

        if self.mode == "min":
            improved = val_metric < (self.best_score - self.min_delta)
        else:
            improved = val_metric > (self.best_score + self.min_delta)

        if improved:
            self.logger.info(
                f"Metric improved from {self.best_score:.6f} to {val_metric:.6f}. Resetting patience counter."
            )
            self.best_score = val_metric
            self.counter = 0
        else:
            self.counter += 1
            self.logger.info(
                f"Metric did not improve. Patience: {self.counter}/{self.patience}. Best score: {self.best_score:.6f}"
            )
            if self.counter >= self.patience:
                self.logger.warning("Early stopping triggered. Training loop terminating.")
                self.early_stop = True

        return self.early_stop


class CheckpointSaver:
    """Handles serialization and state restoration for models, optimizers, and schedulers."""

    def __init__(self, config: TrainingConfig):
        """Initializes the CheckpointSaver.

        Args:
            config: TrainingConfig instance.
        """
        self.config = config
        self.logger = logging.getLogger("audio_training.callbacks.checkpointer")
        self.checkpoint_dir = self.config.checkpoint.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_model_path = self.checkpoint_dir / "best_model.pth"
        self.last_model_path = self.checkpoint_dir / "last_model.pth"
        self.optimizer_path = self.checkpoint_dir / "optimizer.pt"
        self.scheduler_path = self.checkpoint_dir / "scheduler.pt"
        self.state_path = self.checkpoint_dir / "training_state.json"

    def save(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        phase: int,
        monitor_val: float,
        is_best: bool,
        early_stopping_counter: int
    ) -> None:
        """Serializes current training elements to disk.

        Args:
            epoch: Current training epoch index.
            model: PyTorch model module.
            optimizer: Optimizer instance.
            scheduler: Learning rate scheduler instance.
            phase: Current staged training phase (1, 2, or 3).
            monitor_val: Value of monitored metric at this epoch.
            is_best: True if this model achieves the best monitored metric score.
            early_stopping_counter: Current early stopping patience counter value.
        """
        try:
            # Unwrap wrapper module if present
            raw_model = model.model if hasattr(model, "model") else model

            # 1. Save last model weights state
            torch.save(raw_model.state_dict(), self.last_model_path)
            self.logger.debug(f"Saved last model weights state to {self.last_model_path}")

            # 2. Save best model weights if flagged
            if is_best:
                torch.save(raw_model.state_dict(), self.best_model_path)
                self.logger.info(f"New best model weights saved to {self.best_model_path} (Metric: {monitor_val:.6f})")

            # 3. Save optimizer and scheduler states
            torch.save(optimizer.state_dict(), self.optimizer_path)
            if scheduler is not None:
                torch.save(scheduler.state_dict(), self.scheduler_path)

            # 4. Save metadata state JSON
            state = {
                "epoch": epoch,
                "phase": phase,
                "monitor_value": monitor_val,
                "early_stopping_counter": early_stopping_counter,
                "timestamp": str(Path(self.best_model_path).stat().st_mtime if self.best_model_path.exists() else 0)
            }
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=4)

            self.logger.info(f"Saved complete checkpoint state to disk for epoch {epoch}.")

        except Exception as e:
            self.logger.error(f"Failed to save training checkpoints: {e}")

    def load_resume_state(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler]
    ) -> Optional[Dict[str, Any]]:
        """Loads and restores training states if a last checkpoint is present on disk.

        Args:
            model: PyTorch model to restore.
            optimizer: Optimizer to restore.
            scheduler: Scheduler to restore.

        Returns:
            Optional[Dict]: Training state metadata dict if loaded successfully, None otherwise.
        """
        if not self.last_model_path.exists() or not self.state_path.exists():
            self.logger.info("No existing checkpoint found. Starting training from scratch.")
            return None

        try:
            # 1. Load metadata state
            with open(self.state_path, "r", encoding="utf-8") as f:
                state = json.load(f)

            # Unwrap wrapper module if present
            raw_model = model.model if hasattr(model, "model") else model

            # 2. Load model weights
            raw_model.load_state_dict(torch.load(self.last_model_path, map_location="cpu", weights_only=True))
            self.logger.info(f"Restored last model weights from {self.last_model_path}")

            # 3. Load optimizer and scheduler
            if self.optimizer_path.exists():
                optimizer.load_state_dict(torch.load(self.optimizer_path, map_location="cpu", weights_only=True))
                self.logger.info(f"Restored optimizer state from {self.optimizer_path}")

            if scheduler is not None and self.scheduler_path.exists():
                scheduler.load_state_dict(torch.load(self.scheduler_path, map_location="cpu", weights_only=True))
                self.logger.info(f"Restored scheduler state from {self.scheduler_path}")

            self.logger.info(
                f"Resumed pipeline state: Epoch: {state['epoch']}, Phase: {state['phase']}, "
                f"Best Metric Score: {state['monitor_value']:.6f}"
            )
            return state

        except Exception as e:
            self.logger.error(f"Failed to load resume state: {e}. Aborting resume.")
            return None


class TensorBoardLogger:
    """Logs scalars and metrics to TensorBoard runs logs."""

    def __init__(self, config: TrainingConfig):
        """Initializes the TensorBoardLogger.

        Args:
            config: TrainingConfig instance.
        """
        self.config = config
        self.writer = SummaryWriter(log_dir=str(self.config.tensorboard.log_dir))
        self.logger = logging.getLogger("audio_training.callbacks.tensorboard")
        self.logger.info(f"Initialized TensorBoard SummaryWriter logging to: {self.config.tensorboard.log_dir}")

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Logs a single scalar metric value.

        Args:
            tag: Metric identifier name.
            value: Metric scalar value.
            step: Global batch or epoch step count.
        """
        self.writer.add_scalar(tag, value, step)

    def log_metrics_dict(self, metrics: Dict[str, Any], step: int, prefix: str = "") -> None:
        """Logs a dictionary of metrics.

        Args:
            metrics: Dict containing metric values.
            step: Global step count.
            prefix: String prefix to group scalars (e.g. 'train/' or 'val/').
        """
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(f"{prefix}{k}", v, step)

    def close(self) -> None:
        """Closes the TensorBoard writer."""
        self.writer.close()
