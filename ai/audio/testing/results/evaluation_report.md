# Authentix Audio AI Evaluation Report

This report summarizes the performance evaluation of the Audio Deepfake Classification model on the clean testing split.

## Performance Dashboard

- **Accuracy**: 1.0000
- **Balanced Accuracy**: 1.0000
- **Precision**: 1.0000
- **Recall (Sensitivity)**: 1.0000
- **F1-Score**: 1.0000
- **ROC AUC (Area Under Curve)**: 1.0000
- **False Positive Rate (FPR)**: 0.0000
- **False Negative Rate (FNR)**: 0.0000

## Confusion Matrix Table

| | Predicted Real | Predicted Fake |
| :--- | :---: | :---: |
| **Actual Real** | **TN: 88** | FP: 0 |
| **Actual Fake** | FN: 0 | **TP: 101** |

---

## Visual Diagnostic Plots

### Confusion Matrix Plot
![Confusion Matrix](confusion_matrix.png)

### ROC Curve Plot
![ROC Curve](roc_curve.png)

### Precision-Recall Curve Plot
![Precision Recall Curve](precision_recall_curve.png)

---
*Report generated automatically by the Authentix Testing Pipeline.*
