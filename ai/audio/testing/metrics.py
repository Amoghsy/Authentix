"""Evaluation metrics calculation module for the Testing Pipeline.

This module computes performance classification metrics including Accuracy,
Balanced Accuracy, Precision, Recall, F1 Score, ROC AUC, False Positive Rate (FPR),
and False Negative Rate (FNR) using scikit-learn.
"""

import logging
from typing import Dict, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


class AudioTestMetrics:
    """Computes and formats classification metrics for audio deepfake validation."""

    def __init__(self):
        """Initializes the metrics helper and logger."""
        self.logger = logging.getLogger("audio_testing.metrics")

    def compute_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: np.ndarray
    ) -> Dict[str, Union[float, np.ndarray, str]]:
        """Calculates evaluation metrics over target classification lists.

        Args:
            y_true: Ground truth target labels (0 or 1).
            y_pred: Predicted class labels (0 or 1).
            y_probs: Predicted positive class probabilities.

        Returns:
            Dict: Comprehensive metrics storage map.
        """
        # 1. Base sklearn calculations
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )

        # 2. ROC AUC calculation
        roc_auc = 0.5
        unique_classes = np.unique(y_true)
        if len(unique_classes) > 1:
            try:
                roc_auc = roc_auc_score(y_true, y_probs)
            except Exception as e:
                self.logger.warning(f"Failed to calculate ROC AUC: {e}")

        # 3. Confusion Matrix and derived rates (FPR, FNR)
        conf_mat = confusion_matrix(y_true, y_pred)
        
        # Unpack confusion matrix safely (handling potential single class edge cases)
        if conf_mat.shape == (2, 2):
            tn, fp, fn, tp = conf_mat.ravel()
        else:
            # Fallback if only one class exists in true labels
            tn = fp = fn = tp = 0
            if len(unique_classes) == 1:
                single_cls = unique_classes[0]
                if single_cls == 0:  # Only 'real' samples
                    tn = conf_mat[0, 0]
                else:  # Only 'fake' samples
                    tp = conf_mat[0, 0]

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        # 4. Standard text classification report
        class_rep = classification_report(
            y_true, y_pred, target_names=["real", "fake"], zero_division=0
        )

        metrics = {
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "fpr": float(fpr),
            "fnr": float(fnr),
            "confusion_matrix": conf_mat,
            "classification_report": class_rep,
        }

        return metrics

    def generate_console_summary(self, metrics: Dict[str, Union[float, np.ndarray, str]]) -> str:
        """Generates a text-formatted evaluation report dashboard.

        Args:
            metrics: Calculated metrics dictionary.

        Returns:
            str: Human-readable metrics dashboard.
        """
        conf_mat = metrics["confusion_matrix"]
        
        # Format confusion matrix values
        if conf_mat.shape == (2, 2):
            tn_val, fp_val, fn_val, tp_val = conf_mat.ravel()
        else:
            tn_val = fp_val = fn_val = tp_val = 0

        report_str = (
            f"\n============================================\n"
            f"AUTHENTIX AUDIO EVALUATION SUMMARY REPORT\n"
            f"============================================\n"
            f"Accuracy:          {metrics['accuracy']:.4f}\n"
            f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}\n"
            f"Precision:         {metrics['precision']:.4f}\n"
            f"Recall:            {metrics['recall']:.4f}\n"
            f"F1-Score:          {metrics['f1']:.4f}\n"
            f"ROC-AUC:           {metrics['roc_auc']:.4f}\n"
            f"False Positive Rate (FPR): {metrics['fpr']:.4f}\n"
            f"False Negative Rate (FNR): {metrics['fnr']:.4f}\n\n"
            f"Confusion Matrix:\n"
            f"  [TN: {tn_val:<6} FP: {fp_val:<6}]\n"
            f"  [FN: {fn_val:<6} TP: {tp_val:<6}]\n\n"
            f"Classification Report:\n"
            f"{metrics['classification_report']}\n"
            f"============================================"
        )
        return report_str
