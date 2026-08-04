"""
metrics.py

This module defines the MetricTracker class, which collects predictions, targets, and
class probabilities batch-by-batch during training or validation, and computes comprehensive
classification metrics (Accuracy, Balanced Accuracy, F1, Precision, Recall, ROC AUC) and
confusion matrices at the end of each epoch.
"""

import logging
from pathlib import Path
import sys
from typing import Dict, Any, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import torch

# Insert project root to sys.path to allow executing the script directly
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.video.preprocessing.config import AppConfig, get_default_config, setup_logging


class MetricTracker:
    """
    Tracks and computes classification performance metrics incrementally over batches.
    """
    def __init__(self) -> None:
        """
        Initializes the metric lists.
        """
        self.reset()

    def reset(self) -> None:
        """
        Resets all accumulated predictions and targets.
        """
        self.all_preds: List[int] = []
        self.all_labels: List[int] = []
        self.all_probs: List[float] = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Updates the tracker with a batch of model logits and ground truth targets.
        
        Args:
            logits (torch.Tensor): Output raw logits from model forward pass, shape (batch_size, 2).
            targets (torch.Tensor): Ground truth labels, shape (batch_size,).
        """
        # Compute probabilities of the 'fake' class (index 1) via Softmax
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        labels = targets.detach().cpu().numpy()

        self.all_preds.extend(preds.tolist())
        self.all_labels.extend(labels.tolist())
        self.all_probs.extend(probs.tolist())

    def compute(self) -> Dict[str, Any]:
        """
        Computes all accumulated metrics.
        
        Returns:
            Dict[str, Any]: Metrics dictionary containing accuracy, precision, recall, f1, auc, etc.
        """
        y_true = np.array(self.all_labels)
        y_pred = np.array(self.all_preds)
        y_prob = np.array(self.all_probs)
        
        if len(y_true) == 0:
            return {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "roc_auc": 0.0,
                "confusion_matrix": np.zeros((2, 2), dtype=int).tolist(),
                "classification_report": ""
            }

        # Basic Accuracies
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        
        # Binary Classification Stats (focusing on class 1: 'fake' as the positive class)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, 
            y_pred, 
            average="binary", 
            zero_division=0
        )
        
        # ROC AUC (needs both classes present in y_true, otherwise returns 0.5)
        unique_classes = np.unique(y_true)
        if len(unique_classes) > 1:
            try:
                auc = roc_auc_score(y_true, y_prob)
            except Exception as e:
                logging.warning(f"Failed to compute ROC AUC: {e}")
                auc = 0.5
        else:
            auc = 0.5 # Default AUC if only one class is present in batch
            
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        
        # Textual Classification Report
        report = classification_report(
            y_true, 
            y_pred, 
            labels=[0, 1],
            target_names=["real", "fake"], 
            zero_division=0
        )
        
        return {
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(auc),
            "confusion_matrix": cm.tolist(),
            "classification_report": report
        }


if __name__ == "__main__":
    # Self-test block to verify metrics computation
    app_config = get_default_config()
    setup_logging(app_config)
    logging.info("Starting dry-run verification of metrics.py")
    
    try:
        tracker = MetricTracker()
        
        # Create mock batch results
        # batch 1
        logits_1 = torch.tensor([[2.0, -1.0], [-0.5, 1.5], [1.0, 1.2]]) # class predictions: 0, 1, 1
        targets_1 = torch.tensor([0, 1, 0])                             # ground truths: 0, 1, 0
        tracker.update(logits_1, targets_1)
        
        # batch 2
        logits_2 = torch.tensor([[-2.0, 3.0], [0.8, -0.4]])             # class predictions: 1, 0
        targets_2 = torch.tensor([1, 1])                                 # ground truths: 1, 1
        tracker.update(logits_2, targets_2)
        
        # Compute and output metrics
        stats = tracker.compute()
        
        logging.info(f"Accuracy: {stats['accuracy']:.4f}")
        logging.info(f"Balanced Accuracy: {stats['balanced_accuracy']:.4f}")
        logging.info(f"Precision: {stats['precision']:.4f}")
        logging.info(f"Recall: {stats['recall']:.4f}")
        logging.info(f"F1-Score: {stats['f1']:.4f}")
        logging.info(f"ROC AUC: {stats['roc_auc']:.4f}")
        logging.info(f"Confusion Matrix:\n{np.array(stats['confusion_matrix'])}")
        logging.info(f"Classification Report:\n{stats['classification_report']}")
        
        logging.info("Metrics module verified successfully.")
    except Exception as e:
        logging.error(f"Metrics verification failed: {e}")
        raise e
