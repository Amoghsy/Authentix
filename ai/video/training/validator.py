"""
validator.py

This module contains the Validator class which handles model validation. It executes
in evaluation mode (with gradients disabled and mixed precision active), computes the loss,
aggregates validation metrics (accuracy, F1, AUC, etc.) via MetricTracker, and generates
confusion matrix plots using Matplotlib.
"""

import logging
from pathlib import Path
import sys
import time
from typing import Dict, Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Insert project root to sys.path to allow executing the script directly
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.video.preprocessing.config import AppConfig, get_default_config, setup_logging
from ai.video.training.metrics import MetricTracker


class Validator:
    """
    Manages evaluation/validation loops and reports metrics and performance indicators.
    """
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        device: torch.device,
        config: AppConfig
    ) -> None:
        """
        Args:
            model (nn.Module): Model to validate.
            criterion (nn.Module): Loss function.
            device (torch.device): CPU or CUDA device.
            config (AppConfig): Root application configuration.
        """
        self.model = model
        self.criterion = criterion
        self.device = device
        self.config = config
        
        self.use_amp = (self.device.type == "cuda" and config.training.use_mixed_precision)
        self.metric_tracker = MetricTracker()

    def validate(self, dataloader: torch.utils.data.DataLoader, epoch: Optional[int] = None) -> Dict[str, Any]:
        """
        Runs the evaluation loop over the validation or test set.
        
        Args:
            dataloader (DataLoader): PyTorch DataLoader containing validation/test samples.
            epoch (Optional[int]): Current training epoch number (for logging).
            
        Returns:
            Dict[str, Any]: Validation metrics.
        """
        self.model.eval()
        self.metric_tracker.reset()
        
        running_loss = 0.0
        total_samples = 0
        start_time = time.time()
        
        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc="[Validation]" if epoch is None else f"Epoch {epoch} [Val]",
            leave=False
        )
        
        # Disable gradient calculations for validation (efficiency + safety)
        with torch.no_grad():
            for batch_idx, (inputs, targets, _) in pbar:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                batch_size = inputs.size(0)
                
                # Forward pass under autocast context
                if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                    autocast_ctx = torch.amp.autocast("cuda", enabled=self.use_amp)
                else:
                    autocast_ctx = torch.cuda.amp.autocast(enabled=self.use_amp)
                    
                with autocast_ctx:
                    logits = self.model(inputs)
                    loss = self.criterion(logits, targets)
                    
                running_loss += loss.item() * batch_size
                total_samples += batch_size
                self.metric_tracker.update(logits, targets)
                
        epoch_time = time.time() - start_time
        avg_loss = running_loss / total_samples
        throughput = total_samples / epoch_time
        
        # Calculate metrics
        epoch_metrics = self.metric_tracker.compute()
        epoch_metrics["loss"] = avg_loss
        epoch_metrics["duration_seconds"] = epoch_time
        epoch_metrics["throughput_images_per_sec"] = throughput
        
        if epoch is not None:
            logging.info(
                f"Epoch {epoch} [Val]   - Loss: {avg_loss:.4f} | Acc: {epoch_metrics['accuracy']:.4f} | "
                f"F1: {epoch_metrics['f1']:.4f} | AUC: {epoch_metrics['roc_auc']:.4f} | Time: {epoch_time:.2f}s"
            )
        else:
            logging.info(
                f"Evaluation Completed - Loss: {avg_loss:.4f} | Acc: {epoch_metrics['accuracy']:.4f} | "
                f"F1: {epoch_metrics['f1']:.4f} | AUC: {epoch_metrics['roc_auc']:.4f}"
            )
            
        return epoch_metrics

    def save_confusion_matrix(self, cm: List[List[int]], output_path: Path, title: str = "Confusion Matrix") -> None:
        """
        Generates and saves a confusion matrix visualization using Matplotlib.
        
        Args:
            cm (List[List[int]]): 2D confusion matrix array.
            output_path (Path): File path to save the generated image.
            title (str): Graph title.
        """
        cm_array = np.array(cm)
        
        # Setup plot
        plt.figure(figsize=(6, 5))
        plt.imshow(cm_array, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(title, fontsize=14, pad=10)
        plt.colorbar()
        
        # Configure axis ticks
        classes = ["Real", "Fake"]
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=0)
        plt.yticks(tick_marks, classes)
        
        # Annotate text values inside cells
        thresh = cm_array.max() / 2.0
        for i in range(cm_array.shape[0]):
            for j in range(cm_array.shape[1]):
                plt.text(
                    j, i, format(cm_array[i, j], "d"),
                    horizontalalignment="center",
                    color="white" if cm_array[i, j] > thresh else "black",
                    fontsize=12
                )
                
        plt.ylabel("True Class", fontsize=12)
        plt.xlabel("Predicted Class", fontsize=12)
        plt.tight_layout()
        
        try:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logging.info(f"Saved confusion matrix plot to {output_path}")
        except Exception as e:
            logging.error(f"Failed to save confusion matrix plot: {e}")
        finally:
            plt.close()


if __name__ == "__main__":
    # Self-test block to verify validator execution
    app_config = get_default_config()
    setup_logging(app_config)
    logging.info("Starting dry-run verification of validator.py")
    
    try:
        import torch.nn as nn
        from torch.utils.data import DataLoader
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create dummy model and loaders
        dummy_model = nn.Sequential(
            nn.Linear(10, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        ).to(device)
        
        # Create dummy dataloader yielding 3 values
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
        loader = DataLoader(DummyDataset(x, y), batch_size=2)
        
        loss_fn = nn.CrossEntropyLoss()
        
        # Instantiate validator
        validator = Validator(
            model=dummy_model,
            criterion=loss_fn,
            device=device,
            config=app_config
        )
        
        # Validate
        val_stats = validator.validate(loader, epoch=1)
        logging.info(f"Self-test val accuracy: {val_stats['accuracy']:.4f}")
        
        # Plot confusion matrix
        cm_path = app_config.paths.checkpoints_dir / "test_confusion_matrix.png"
        validator.save_confusion_matrix(val_stats["confusion_matrix"], cm_path)
        
        # Clean up plotted image in dry run
        if cm_path.exists():
            cm_path.unlink()
            logging.info("Cleaned up test plot successfully.")
            
        logging.info("Validator module verified successfully.")
    except Exception as e:
        logging.error(f"Validator verification failed: {e}")
        raise e
