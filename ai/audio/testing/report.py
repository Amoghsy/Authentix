"""Report generation module for the Testing Pipeline.

This module exports model evaluation outputs to standard disk files:
- CSV Predictions table
- JSON Metrics summary
- Plain text classification report
- Markdown summary dashboard
- PNG plots (Confusion Matrix, ROC Curve, Precision-Recall Curve)
"""

import json
import logging
from pathlib import Path
from typing import Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from ai.audio.testing.config import TestingConfig


class AudioTestReportGenerator:
    """Generates visual plots and text reports for classification performance evaluation."""

    def __init__(self, config: TestingConfig):
        """Initializes the AudioTestReportGenerator.

        Args:
            config: TestingConfig instance.
        """
        self.config = config
        self.logger = logging.getLogger("audio_testing.report")
        self.output_dir = self.config.report.output_dir
        self.config.create_directories()

    def generate_all_reports(
        self,
        metrics: Dict[str, Union[float, np.ndarray, str]],
        predictions_df: pd.DataFrame,
        y_true: np.ndarray,
        y_probs: np.ndarray
    ) -> None:
        """Saves all evaluation reports and plots to the results output directory.

        Args:
            metrics: Dict of computed evaluation metrics.
            predictions_df: Detailed prediction rows.
            y_true: Ground truth target labels.
            y_probs: Predicted positive class probabilities.
        """
        self.logger.info(f"Generating evaluation reports in: {self.output_dir.resolve()}")

        # 1. Save CSV Predictions
        predictions_path = self.output_dir / self.config.report.predictions_csv
        predictions_df.to_csv(predictions_path, index=False)
        self.logger.info(f"Saved predictions CSV to {predictions_path.name}")

        # 2. Save Plain Text Classification Report
        report_path = self.output_dir / self.config.report.classification_report
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(str(metrics["classification_report"]))
        self.logger.info(f"Saved classification report text to {report_path.name}")

        # 3. Save JSON Metrics Summary
        json_path = self.output_dir / self.config.report.results_json
        serializable_metrics = self._make_serializable(metrics)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(serializable_metrics, f, indent=4)
        self.logger.info(f"Saved metrics JSON summary to {json_path.name}")

        # 4. Generate visual plots (CM, ROC, PR)
        self._plot_confusion_matrix(metrics["confusion_matrix"])
        self._plot_roc_curve(y_true, y_probs, metrics["roc_auc"])
        self._plot_precision_recall_curve(y_true, y_probs, metrics["precision"], metrics["recall"])

        # 5. Generate Markdown Summary Report
        self._generate_markdown_report(metrics)
        self.logger.info("All testing artifacts generated successfully.")

    def _make_serializable(self, metrics: Dict) -> Dict:
        """Prepares a dictionary containing numpy structures for JSON serialization."""
        serializable = {}
        for k, v in metrics.items():
            if isinstance(v, np.ndarray):
                serializable[k] = v.tolist()
            elif isinstance(v, (np.float32, np.float64)):
                serializable[k] = float(v)
            elif isinstance(v, (np.int32, np.int64)):
                serializable[k] = int(v)
            else:
                serializable[k] = v
        return serializable

    def _plot_confusion_matrix(self, conf_mat: np.ndarray) -> None:
        """Generates and saves the annotated Confusion Matrix plot."""
        plt.figure(figsize=(6, 5))
        
        # Unpack confusion matrix values
        if conf_mat.shape == (2, 2):
            tn, fp, fn, tp = conf_mat.ravel()
            # Standard labels mapping
            labels = [
                [f"True Real\n(TN: {tn})", f"False Fake\n(FP: {fp})"],
                [f"False Real\n(FN: {fn})", f"True Fake\n(TP: {tp})"]
            ]
        else:
            labels = [["", ""], ["", ""]]
            tn = fp = fn = tp = 0

        # Create plot
        plt.imshow(conf_mat, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix - Audio Evaluation")
        plt.colorbar()
        
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["real", "fake"])
        plt.yticks(tick_marks, ["real", "fake"])

        # Annotate labels
        thresh = conf_mat.max() / 2.0
        for i in range(conf_mat.shape[0]):
            for j in range(conf_mat.shape[1]):
                plt.text(
                    j, i, labels[i][j],
                    horizontalalignment="center",
                    color="white" if conf_mat[i, j] > thresh else "black"
                )

        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        
        cm_path = self.output_dir / self.config.report.confusion_matrix_img
        plt.savefig(cm_path, dpi=150)
        plt.close()
        self.logger.info(f"Saved confusion matrix plot to {cm_path.name}")

    def _plot_roc_curve(self, y_true: np.ndarray, y_probs: np.ndarray, auc_score: float) -> None:
        """Generates and saves the Receiver Operating Characteristic (ROC) curve plot."""
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc_score:.4f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.grid(True, linestyle=":", alpha=0.6)
        
        roc_path = self.output_dir / self.config.report.roc_curve_img
        plt.savefig(roc_path, dpi=150)
        plt.close()
        self.logger.info(f"Saved ROC curve plot to {roc_path.name}")

    def _plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        precision: float,
        recall: float
    ) -> None:
        """Generates and saves the Precision-Recall (PR) curve plot."""
        p_curve, r_curve, _ = precision_recall_curve(y_true, y_probs)
        
        plt.figure(figsize=(6, 5))
        plt.plot(r_curve, p_curve, color="blue", lw=2, label="Precision-Recall Curve")
        plt.scatter([recall], [precision], color="red", zorder=5, label=f"F1 Point (P={precision:.2f}, R={recall:.2f})")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall (PR) Curve")
        plt.legend(loc="lower left")
        plt.grid(True, linestyle=":", alpha=0.6)
        
        pr_path = self.output_dir / self.config.report.precision_recall_curve_img
        plt.savefig(pr_path, dpi=150)
        plt.close()
        self.logger.info(f"Saved PR curve plot to {pr_path.name}")

    def _generate_markdown_report(self, metrics: Dict) -> None:
        """Generates a comprehensive Markdown summary report including embedded plots."""
        conf_mat = metrics["confusion_matrix"]
        if conf_mat.shape == (2, 2):
            tn, fp, fn, tp = conf_mat.ravel()
        else:
            tn = fp = fn = tp = 0

        md_content = f"""# Authentix Audio AI Evaluation Report

This report summarizes the performance evaluation of the Audio Deepfake Classification model on the clean testing split.

## Performance Dashboard

- **Accuracy**: {metrics['accuracy']:.4f}
- **Balanced Accuracy**: {metrics['balanced_accuracy']:.4f}
- **Precision**: {metrics['precision']:.4f}
- **Recall (Sensitivity)**: {metrics['recall']:.4f}
- **F1-Score**: {metrics['f1']:.4f}
- **ROC AUC (Area Under Curve)**: {metrics['roc_auc']:.4f}
- **False Positive Rate (FPR)**: {metrics['fpr']:.4f}
- **False Negative Rate (FNR)**: {metrics['fnr']:.4f}

## Confusion Matrix Table

| | Predicted Real | Predicted Fake |
| :--- | :---: | :---: |
| **Actual Real** | **TN: {tn}** | FP: {fp} |
| **Actual Fake** | FN: {fn} | **TP: {tp}** |

---

## Visual Diagnostic Plots

### Confusion Matrix Plot
![Confusion Matrix]({self.config.report.confusion_matrix_img})

### ROC Curve Plot
![ROC Curve]({self.config.report.roc_curve_img})

### Precision-Recall Curve Plot
![Precision Recall Curve]({self.config.report.precision_recall_curve_img})

---
*Report generated automatically by the Authentix Testing Pipeline.*
"""
        md_path = self.output_dir / self.config.report.evaluation_report_md
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        self.logger.info(f"Saved Markdown evaluation report to {md_path.name}")
