"""Trainer engine orchestration module for model optimization.

This module coordinates PyTorch training loops, mixed-precision (FP16) training passes,
gradient accumulation, gradient norm clipping, and performance logging (speed, loss,
learning rate, and GPU memory metrics).
"""

import logging
import time
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from ai.audio.training.callbacks import CheckpointSaver, TensorBoardLogger
from ai.audio.training.config import TrainingConfig
from ai.audio.training.metrics import TrainingMetricsTracker


class AudioTrainer:
    """Orchestrates the model training epochs, optimization steps, and metrics logging."""

    def __init__(
        self,
        config: TrainingConfig,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        criterion: nn.Module,
        train_loader: DataLoader,
        tb_logger: TensorBoardLogger,
        device: torch.device
    ):
        """Initializes the AudioTrainer.

        Args:
            config: TrainingConfig instance.
            model: Audio classifier model (wrapped or raw nn.Module).
            optimizer: Optimizer instance.
            scheduler: Learning rate scheduler instance.
            criterion: Loss module.
            train_loader: DataLoader for the training partition.
            tb_logger: TensorBoardLogger callback instance.
            device: Target torch device (cuda or cpu).
        """
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.train_loader = train_loader
        self.tb_logger = tb_logger
        self.device = device
        
        self.logger = logging.getLogger("audio_training.trainer")

        # Initialize GradScaler for mixed precision (FP16)
        self.scaler = torch.amp.GradScaler(
            device=self.device.type,
            enabled=self.config.mixed_precision
        )

        self.metrics_tracker = TrainingMetricsTracker()

    def train_epoch(self, epoch: int, phase: int) -> Dict[str, float]:
        """Runs a single epoch of training.

        Args:
            epoch: Current epoch index (1-indexed).
            phase: Current staged training phase (1, 2, or 3).

        Returns:
            Dict[str, float]: Dictionary of average training epoch stats.
        """
        self.model.train()
        self.metrics_tracker.reset()
        self.optimizer.zero_grad()

        running_loss = 0.0
        samples_processed = 0
        epoch_start_time = time.time()
        
        # Use tqdm for progress tracking in console
        progress_bar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {epoch} [Phase {phase}]",
            leave=False
        )

        for step, batch in progress_bar:
            batch_start_time = time.time()

            # Move inputs to target device
            input_values = batch["input_values"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            
            attention_mask = None
            if "attention_mask" in batch:
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)

            # 1. Forward Pass with Autocast (Mixed Precision)
            with torch.amp.autocast(device_type=self.device.type, enabled=self.config.mixed_precision):
                outputs = self.model(
                    input_values=input_values,
                    attention_mask=attention_mask
                )
                logits = outputs["logits"]
                loss = self.criterion(logits, labels)
                
                # Normalize loss to match gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps

            # 2. Backward Pass (with Scaled Gradient)
            self.scaler.scale(loss).backward()

            # Update running loss
            running_loss += loss.item() * self.config.gradient_accumulation_steps
            samples_processed += len(labels)

            # 3. Optimizer Step (handling gradient accumulation)
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Unscale gradients for clipping
                self.scaler.unscale_(self.optimizer)

                # Gradient Clipping
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )

                # Optimizer step and Scaler update
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            else:
                grad_norm = 0.0

            # Accumulate running metrics
            self.metrics_tracker.update(logits, labels)

            # Calculate step timing and speed
            step_duration = time.time() - batch_start_time
            throughput = len(labels) / step_duration if step_duration > 0 else 0.0

            # Log step metrics to TensorBoard at configured intervals
            global_step = (epoch - 1) * len(self.train_loader) + step
            if global_step % self.config.tensorboard.log_step_interval == 0:
                self.tb_logger.log_scalar("train/step_loss", loss.item() * self.config.gradient_accumulation_steps, global_step)
                self.tb_logger.log_scalar("train/step_grad_norm", float(grad_norm), global_step)
                self.tb_logger.log_scalar("train/step_throughput", throughput, global_step)

            # Update progress bar description
            progress_bar.set_postfix({
                "loss": f"{loss.item() * self.config.gradient_accumulation_steps:.4f}",
                "norm": f"{grad_norm:.3f}"
            })

        # Calculate epoch-level stats
        epoch_duration = time.time() - epoch_start_time
        avg_loss = running_loss / len(self.train_loader)
        
        # Compute metrics
        epoch_metrics = self.metrics_tracker.compute()
        accuracy = epoch_metrics.get("accuracy", 0.0)

        # Get current learning rate from optimizer
        current_lr = self.optimizer.param_groups[0]["lr"]

        # Track GPU memory utilization
        gpu_mem_mb = 0.0
        if self.device.type == "cuda":
            gpu_mem_mb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
            # Reset peak counter for next epoch monitoring
            torch.cuda.reset_peak_memory_stats(self.device)

        epoch_stats = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "duration": epoch_duration,
            "lr": current_lr,
            "gpu_memory_mb": gpu_mem_mb
        }

        # Log epoch summary details
        self.logger.info(
            f"Epoch {epoch} Finished | Phase: {phase} | "
            f"Train Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f} | "
            f"LR: {current_lr:.2e} | Duration: {epoch_duration:.2f}s | "
            f"Peak GPU Mem: {gpu_mem_mb:.1f} MB"
        )

        # Log epoch metrics to TensorBoard
        self.tb_logger.log_scalar("train/epoch_loss", avg_loss, epoch)
        self.tb_logger.log_scalar("train/epoch_accuracy", accuracy, epoch)
        self.tb_logger.log_scalar("train/epoch_lr", current_lr, epoch)
        self.tb_logger.log_scalar("train/epoch_gpu_memory_mb", gpu_mem_mb, epoch)

        return epoch_stats
