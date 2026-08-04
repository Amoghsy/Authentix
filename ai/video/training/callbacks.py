"""
callbacks.py

This module contains training utility callback classes:
1. EarlyStopping: Stops training early if validation metric doesn't improve.
2. CheckpointSaver: Saves best and last model weights, optimizer, scheduler, and json states.
3. TensorBoardLogger: Manages SummaryWriter logs for train/val metrics and learning rates.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
import sys
from typing import Dict, Any, Optional

import torch
from torch.utils.tensorboard import SummaryWriter

# Insert project root to sys.path to allow executing the script directly
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.video.preprocessing.config import AppConfig, get_default_config, setup_logging


class EarlyStopping:
    """
    Monitors a metric (e.g., validation loss) and signals to stop training
    early if it fails to improve after a set number of epochs.
    """
    def __init__(self, patience: int = 5, mode: str = "min", min_delta: float = 1e-4) -> None:
        """
        Args:
            patience (int): Number of epochs to wait for improvement.
            mode (str): 'min' (e.g., loss) or 'max' (e.g., accuracy, f1).
            min_delta (float): Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.mode = mode.lower()
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

        if self.mode not in {"min", "max"}:
            raise ValueError("EarlyStopping mode must be 'min' or 'max'")

    def step(self, score: float) -> bool:
        """
        Updates tracking statistics and checks if early stopping is triggered.
        
        Args:
            score (float): The monitored metric score for the current epoch.
            
        Returns:
            bool: True if early stopping should trigger, False otherwise.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            logging.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.praise_limit():
                self.early_stop = True
                
        return self.early_stop

    def praise_limit(self) -> int:
        return self.patience


class CheckpointSaver:
    """
    Manages saving and loading model checkpoints and training state metadata.
    """
    def __init__(self, checkpoints_dir: Path, mode: str = "min") -> None:
        """
        Args:
            checkpoints_dir (Path): Absolute folder path to store checkpoint files.
            mode (str): Optimization mode ('min' for loss, 'max' for accuracy/f1).
        """
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode.lower()
        self.best_score = float("inf") if self.mode == "min" else float("-inf")
        
        if self.mode not in {"min", "max"}:
            raise ValueError("CheckpointSaver mode must be 'min' or 'max'")

    def should_save_best(self, score: float) -> bool:
        """
        Evaluates if the current score is the best observed so far.
        """
        if self.mode == "min":
            return score < self.best_score
        else:
            return score > self.best_score

    def save(
        self, 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        scheduler: Any, 
        epoch: int, 
        metrics: Dict[str, Any], 
        score: float
    ) -> None:
        """
        Saves checkpoints to disk. Saves best_model.pth if score improved,
        and always saves last_model.pth alongside training states.
        """
        is_best = self.should_save_best(score)
        if is_best:
            self.best_score = score
            
        # Unwrap model if wrapped in DataParallel
        model_to_save = model.module if hasattr(model, "module") else model
        
        # Prepare state dicts
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "best_score": self.best_score,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save last checkpoint (resumption target)
        last_pth = self.checkpoints_dir / "last_model.pth"
        torch.save(checkpoint, last_pth)
        
        # Save metadata info as JSON
        state_meta = {
            "last_epoch": epoch,
            "best_score": self.best_score,
            "last_metrics": {k: v for k, v in metrics.items() if k != "classification_report"},
            "last_saved_at": datetime.now().isoformat()
        }
        
        meta_json_file = self.checkpoints_dir / "training_state.json"
        with open(meta_json_file, "w", encoding="utf-8") as f:
            json.dump(state_meta, f, indent=4)
            
        # Save standalone optimizer and scheduler states as requested
        torch.save(optimizer.state_dict(), self.checkpoints_dir / "optimizer.pt")
        if scheduler is not None:
            torch.save(scheduler.state_dict(), self.checkpoints_dir / "scheduler.pt")
            
        if is_best:
            best_pth = self.checkpoints_dir / "best_model.pth"
            torch.save(checkpoint, best_pth)
            logging.info(f"🏆 New best model checkpoint saved to {best_pth} (score: {score:.5f}).")
        else:
            logging.info(f"Checkpoint saved to {last_pth} for epoch {epoch}.")


class TensorBoardLogger:
    """
    Logs learning curves and scalars directly to TensorBoard SummaryWriter.
    """
    def __init__(self, log_dir: Path) -> None:
        """
        Args:
            log_dir (Path): Output directory for TensorBoard logs.
        """
        self.log_dir = Path(log_dir)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        logging.info(f"TensorBoard summary writer initialized at {self.log_dir}")

    def log_epoch(
        self, 
        epoch: int, 
        train_metrics: Dict[str, Any], 
        val_metrics: Dict[str, Any], 
        lr: float
    ) -> None:
        """
        Writes train/val metrics and learning rates at the end of an epoch.
        """
        # Log learning rate
        self.writer.add_scalar("Hyperparameters/Learning_Rate", lr, epoch)
        
        # Log losses
        if "loss" in train_metrics:
            self.writer.add_scalar("Loss/Train", train_metrics["loss"], epoch)
        if "loss" in val_metrics:
            self.writer.add_scalar("Loss/Validation", val_metrics["loss"], epoch)
            
        # Log metrics
        for metric_name in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]:
            if metric_name in train_metrics:
                self.writer.add_scalar(f"Metrics_Train/{metric_name.capitalize()}", train_metrics[metric_name], epoch)
            if metric_name in val_metrics:
                self.writer.add_scalar(f"Metrics_Val/{metric_name.capitalize()}", val_metrics[metric_name], epoch)

    def close(self) -> None:
        """
        Closes the TensorBoard writer.
        """
        self.writer.close()


if __name__ == "__main__":
    # Self-test block to verify callbacks execution
    app_config = get_default_config()
    setup_logging(app_config)
    logging.info("Starting dry-run verification of callbacks.py")
    
    try:
        # 1. Test EarlyStopping
        early_stopping = EarlyStopping(patience=3, mode="min")
        early_stopping.step(0.8) # epoch 1
        early_stopping.step(0.7) # epoch 2 (improved)
        early_stopping.step(0.72) # epoch 3 (worse)
        early_stopping.step(0.75) # epoch 4 (worse)
        stop = early_stopping.step(0.71) # epoch 5 (worse, patience exceeded)
        logging.info(f"EarlyStopping trigger check (expected True): {stop}")
        
        # 2. Test CheckpointSaver
        import torch.nn as nn
        import torch.optim as optim
        
        dummy_model = nn.Linear(10, 2)
        dummy_optimizer = optim.SGD(dummy_model.parameters(), lr=0.1)
        dummy_scheduler = optim.lr_scheduler.StepLR(dummy_optimizer, step_size=1)
        
        saver = CheckpointSaver(app_config.paths.checkpoints_dir, mode="min")
        saver.save(
            model=dummy_model,
            optimizer=dummy_optimizer,
            scheduler=dummy_scheduler,
            epoch=1,
            metrics={"loss": 0.5, "accuracy": 0.8},
            score=0.5
        )
        
        saver.save(
            model=dummy_model,
            optimizer=dummy_optimizer,
            scheduler=dummy_scheduler,
            epoch=2,
            metrics={"loss": 0.4, "accuracy": 0.85},
            score=0.4 # Improved (should save best)
        )
        
        # 3. Test TensorBoardLogger
        tb_logger = TensorBoardLogger(app_config.paths.checkpoints_dir / "runs" / "test_run")
        tb_logger.log_epoch(
            epoch=1,
            train_metrics={"loss": 0.5, "accuracy": 0.8},
            val_metrics={"loss": 0.6, "accuracy": 0.75},
            lr=0.1
        )
        tb_logger.close()
        logging.info("TensorBoard logging executed successfully.")
        
        logging.info("Callbacks module verified successfully.")
    except Exception as e:
        logging.error(f"Callbacks verification failed: {e}")
        raise e
