"""Validator engine module for model evaluation.

This module coordinates PyTorch evaluation runs over validation dataloaders under disabled
gradient tracking. It calculates classification metrics and logs evaluation summaries to TensorBoard
and training log outputs.
"""

import logging
import time
from typing import Any, Dict, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ai.audio.training.callbacks import TensorBoardLogger
from ai.audio.training.config import TrainingConfig
from ai.audio.training.metrics import TrainingMetricsTracker


class AudioValidator:
    """Orchestrates model validation validation epochs, metrics collection, and evaluation logging."""

    def __init__(
        self,
        config: TrainingConfig,
        model: nn.Module,
        criterion: nn.Module,
        val_loader: DataLoader,
        tb_logger: TensorBoardLogger,
        device: torch.device
    ):
        """Initializes the AudioValidator.

        Args:
            config: TrainingConfig instance.
            model: Audio classifier model (wrapped or raw nn.Module).
            criterion: Loss module.
            val_loader: DataLoader for the validation partition.
            tb_logger: TensorBoardLogger callback instance.
            device: Target torch device (cuda or cpu).
        """
        self.config = config
        self.model = model
        self.criterion = criterion
        self.val_loader = val_loader
        self.tb_logger = tb_logger
        self.device = device

        self.logger = logging.getLogger("audio_training.validator")
        self.metrics_tracker = TrainingMetricsTracker()

    def validate(self, epoch: int, phase: int) -> Dict[str, Any]:
        """Runs the validation evaluation pass on the dataset.

        Args:
            epoch: Current training epoch index (1-indexed).
            phase: Current staged training phase (1, 2, or 3).

        Returns:
            Dict[str, Any]: Evaluated epoch metrics including accuracy, f1, and validation loss.
        """
        self.model.eval()
        self.metrics_tracker.reset()
        
        running_loss = 0.0
        val_start_time = time.time()

        # Disable gradients during validation loop
        with torch.no_grad():
            progress_bar = tqdm(
                self.val_loader,
                total=len(self.val_loader),
                desc=f"Validation {epoch}",
                leave=False
            )
            
            for batch in progress_bar:
                # Move inputs to target device
                input_values = batch["input_values"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)
                
                attention_mask = None
                if "attention_mask" in batch:
                    attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)

                # Forward pass inside autocast (Mixed Precision compatibility)
                with torch.amp.autocast(device_type=self.device.type, enabled=self.config.mixed_precision):
                    outputs = self.model(
                        input_values=input_values,
                        attention_mask=attention_mask
                    )
                    logits = outputs["logits"]
                    loss = self.criterion(logits, labels)

                running_loss += loss.item()
                self.metrics_tracker.update(logits, labels)

        # Calculate epoch evaluation stats
        val_duration = time.time() - val_start_time
        avg_loss = running_loss / len(self.val_loader)
        
        # Calculate evaluation metrics
        metrics = self.metrics_tracker.compute()
        
        # Add loss and duration to metrics dictionary
        metrics["loss"] = avg_loss
        metrics["duration"] = val_duration

        # Log formatted summary to files
        summary_report = self.metrics_tracker.generate_summary_report()
        self.logger.info(
            f"\nEpoch {epoch} Validation Complete | Phase: {phase} | Loss: {avg_loss:.4f} | "
            f"Accuracy: {metrics['accuracy']:.4f} | F1-Score: {metrics['f1']:.4f}\n"
            f"{summary_report}"
        )
        print(f"\nValidation {epoch} Finished | Loss: {avg_loss:.4f} | Accuracy: {metrics['accuracy']:.4f}")

        # Log metrics to TensorBoard
        self.tb_logger.log_scalar("val/epoch_loss", avg_loss, epoch)
        self.tb_logger.log_metrics_dict(metrics, epoch, prefix="val/")

        return metrics
