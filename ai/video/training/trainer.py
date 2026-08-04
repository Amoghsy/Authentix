"""
trainer.py

This module contains the Trainer class which encapsulates the PyTorch training epoch loop.
It handles forward and backward passes, gradient scaling/mixed precision (via torch.cuda.amp),
gradient clipping, metric aggregation (accuracy, F1, loss, etc.), learning rate scheduling,
resumption from checkpoint files, and performance monitoring (images/sec, GPU memory).
"""

import logging
from pathlib import Path
import sys
import time
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Insert project root to sys.path to allow executing the script directly
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.video.preprocessing.config import AppConfig, get_default_config, setup_logging
from ai.video.training.metrics import MetricTracker


class Trainer:
    """
    Manages the training loop and state updates for the Deepfake model.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        scheduler: Any,
        device: torch.device,
        config: AppConfig
    ) -> None:
        """
        Args:
            model (nn.Module): PyTorch model to train.
            optimizer (optim.Optimizer): Model parameters optimizer.
            criterion (nn.Module): Loss function.
            scheduler (Any): Learning rate scheduler.
            device (torch.device): Execution device (CPU or CUDA).
            config (AppConfig): Root application configuration.
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.config = config
        
        # Configure Mixed Precision (Automatic Mixed Precision - AMP)
        # Only enable AMP on CUDA devices
        self.use_amp = (self.device.type == "cuda" and config.training.use_mixed_precision)
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        if self.use_amp:
            logging.info("Mixed precision training (AMP) enabled on CUDA device.")
        else:
            logging.info(f"AMP disabled. Training on {self.device.type.upper()}.")

        self.metric_tracker = MetricTracker()

    def train_epoch(self, dataloader: torch.utils.data.DataLoader, epoch: int) -> Dict[str, Any]:
        """
        Runs one full epoch of training.
        
        Args:
            dataloader (DataLoader): PyTorch DataLoader containing training samples.
            epoch (int): Current epoch number.
            
        Returns:
            Dict[str, Any]: Metrics dictionary for the current epoch.
        """
        self.model.train()
        self.metric_tracker.reset()
        
        running_loss = 0.0
        total_samples = 0
        start_time = time.time()
        
        # Using tqdm progress bar inside logging context
        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch {epoch} [Train]",
            leave=False
        )
        
        for batch_idx, (inputs, targets, _) in pbar:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            batch_size = inputs.size(0)
            
            # Reset gradients
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass under autocast context
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                autocast_ctx = torch.amp.autocast("cuda", enabled=self.use_amp)
            else:
                autocast_ctx = torch.cuda.amp.autocast(enabled=self.use_amp)
                
            with autocast_ctx:
                logits = self.model(inputs)
                loss = self.criterion(logits, targets)
                
            # Backward pass under scaler context
            self.scaler.scale(loss).backward()
            
            # Gradient clipping (unscales gradients first to prevent clipping scaled grads)
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update metrics
            running_loss += loss.item() * batch_size
            total_samples += batch_size
            self.metric_tracker.update(logits, targets)
            
            # Update progress bar description with loss
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        epoch_time = time.time() - start_time
        avg_loss = running_loss / total_samples
        throughput = total_samples / epoch_time
        
        # Gather metrics
        epoch_metrics = self.metric_tracker.compute()
        epoch_metrics["loss"] = avg_loss
        epoch_metrics["duration_seconds"] = epoch_time
        epoch_metrics["throughput_images_per_sec"] = throughput
        
        # Track GPU memory utilization if training on CUDA
        if self.device.type == "cuda":
            # PyTorch returns bytes; convert to megabytes
            gpu_mem = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            epoch_metrics["gpu_memory_mb"] = gpu_mem
            logging.info(
                f"Epoch {epoch} [Train] - Loss: {avg_loss:.4f} | Acc: {epoch_metrics['accuracy']:.4f} | "
                f"Speed: {throughput:.1f} img/s | GPU Mem: {gpu_mem:.1f} MB | Time: {epoch_time:.2f}s"
            )
        else:
            epoch_metrics["gpu_memory_mb"] = 0.0
            logging.info(
                f"Epoch {epoch} [Train] - Loss: {avg_loss:.4f} | Acc: {epoch_metrics['accuracy']:.4f} | "
                f"Speed: {throughput:.1f} img/s | Time: {epoch_time:.2f}s"
            )
            
        return epoch_metrics

    def resume(self, checkpoint_path: Path) -> int:
        """
        Resumes model parameters and optimizer states from a checkpoint file.
        
        Args:
            checkpoint_path (Path): Path to last_model.pth checkpoint.
            
        Returns:
            int: The next epoch to start training from.
        """
        logging.info(f"Attempting to resume training state from checkpoint: {checkpoint_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file does not exist at {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load weights
        model_to_load = self.model.module if hasattr(self.model, "module") else self.model
        model_to_load.load_state_dict(checkpoint["model_state_dict"])
        logging.info("Model state dict successfully loaded.")
        
        # Load optimizer
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logging.info("Optimizer state dict successfully loaded.")
        
        # Load scheduler if present
        if self.scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            logging.info("Scheduler state dict successfully loaded.")
            
        resume_epoch = checkpoint["epoch"] + 1
        logging.info(f"Resuming pipeline execution successfully. Next training epoch: {resume_epoch}")
        
        return resume_epoch


if __name__ == "__main__":
    # Self-test block to verify trainer construction
    app_config = get_default_config()
    setup_logging(app_config)
    logging.info("Starting dry-run verification of trainer.py")
    
    try:
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Create a dummy model and data
        dummy_model = nn.Sequential(
            nn.Linear(10, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        ).to(device)
        
        # 2. Setup optimizer, loss, and scheduler
        opt = optim.AdamW(dummy_model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        
        # 3. Create dummy loader
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, x, y):
                self.x = x
                self.y = y
            def __len__(self):
                return len(self.x)
            def __getitem__(self, idx):
                return self.x[idx], self.y[idx], "dummy_path.jpg"
                
        x = torch.randn(10, 10)
        y = torch.randint(0, 2, (10,))
        ds = DummyDataset(x, y)
        loader = DataLoader(ds, batch_size=2)
        
        # 4. Instantiate trainer
        trainer = Trainer(
            model=dummy_model,
            optimizer=opt,
            criterion=loss_fn,
            scheduler=sched,
            device=device,
            config=app_config
        )
        
        # 5. Train for 1 dummy epoch
        epoch_stats = trainer.train_epoch(loader, epoch=1)
        logging.info(f"Self-test epoch loss: {epoch_stats['loss']:.4f}")
        logging.info(f"Self-test throughput: {epoch_stats['throughput_images_per_sec']:.1f} images/sec")
        
        logging.info("Trainer module verified successfully.")
    except Exception as e:
        logging.error(f"Trainer verification failed: {e}")
        raise e
