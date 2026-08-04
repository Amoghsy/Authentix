"""Training metrics tracking and validation reporting module.

This module provides metric trackers to accumulate logits and labels during epoch steps,
and compute validation metrics: accuracy, balanced accuracy, precision, recall, F1,
and ROC AUC using scikit-learn.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


class TrainingMetricsTracker:
    """Accumulates prediction logits and targets, and computes classification metrics."""

    def __init__(self):
        """Initializes the TrainingMetricsTracker and resets state."""
        self.logger = logging.getLogger("audio_training.metrics")
        self.reset()

    def reset(self) -> None:
        """Clears all running validation targets and prediction scores."""
        self.all_logits: List[np.ndarray] = []
        self.all_labels: List[np.ndarray] = []

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """Accumulates prediction logits and true targets from a training/validation batch.

        Args:
            logits: Predicted class raw scores tensor [batch_size, num_classes].
            labels: Ground truth classification labels tensor [batch_size].
        """
        # Detach and move to CPU before converting to numpy
        logits_np = logits.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        self.all_logits.append(logits_np)
        self.all_labels.append(labels_np)

    def compute(self) -> Dict[str, Union[float, np.ndarray, str]]:
        """Calculates classification metrics over all accumulated steps.

        Returns:
            Dict[str, Union[float, np.ndarray, str]]: Dictionary of computed metrics.
        """
        if not self.all_labels:
            self.logger.warning("No samples accumulated. Returning empty metrics.")
            return {}

        # Concatenate arrays along batch dimension
        logits = np.concatenate(self.all_logits, axis=0)
        labels = np.concatenate(self.all_labels, axis=0)

        # Calculate prediction labels (argmax of logits)
        preds = np.argmax(logits, axis=1)

        # Calculate class probabilities (Softmax on logits)
        # Subtract max for numerical stability
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        # Binary probability of the positive class (class 1: 'fake')
        pos_probs = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]

        # Calculate metrics using sklearn
        acc = accuracy_score(labels, preds)
        bal_acc = balanced_accuracy_score(labels, preds)
        
        # Calculate binary class metrics (focusing on class 1: 'fake')
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )

        # Calculate ROC AUC safely (requires samples from both classes)
        roc_auc = 0.5
        unique_classes = np.unique(labels)
        if len(unique_classes) > 1:
            try:
                roc_auc = roc_auc_score(labels, pos_probs)
            except Exception as e:
                self.logger.warning(f"Failed to calculate ROC AUC: {e}")
        else:
            self.logger.debug("ROC AUC calculation skipped: only one unique class in labels.")

        conf_mat = confusion_matrix(labels, preds)
        class_rep = classification_report(
            labels, preds, target_names=["real", "fake"], zero_division=0
        )

        metrics = {
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "confusion_matrix": conf_mat,
            "classification_report": class_rep,
        }

        return metrics

    def generate_summary_report(self) -> str:
        """Generates a formatted text summary dashboard of calculated metrics.

        Returns:
            str: Human-readable metrics dashboard.
        """
        metrics = self.compute()
        if not metrics:
            return "No data recorded for summary."

        conf_mat = metrics["confusion_matrix"]
        
        report_str = (
            f"=== Classification Metrics Summary ===\n"
            f"Accuracy:          {metrics['accuracy']:.4f}\n"
            f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}\n"
            f"Precision:         {metrics['precision']:.4f}\n"
            f"Recall:            {metrics['recall']:.4f}\n"
            f"F1-Score:          {metrics['f1']:.4f}\n"
            f"ROC-AUC:           {metrics['roc_auc']:.4f}\n\n"
            f"Confusion Matrix:\n"
            f"  [TN: {conf_mat[0,0]:<5} FP: {conf_mat[0,1]:<5}]\n"
            f"  [FN: {conf_mat[1,0]:<5} TP: {conf_mat[1,1]:<5}]\n\n"
            f"Classification Report:\n"
            f"{metrics['classification_report']}\n"
            f"======================================"
        )
        return report_str
