"""
test.py

This module orchestrates testing and evaluation of the trained deepfake detection model
on the test set split. It loads the best saved weights, runs the validation loop,
computes metrics, and generates visual ROC curves, Precision-Recall curves, and confusion matrices.
"""

import argparse
import logging
from pathlib import Path
import sys
import time
from typing import Dict, Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Insert project root to sys.path to allow executing the script directly
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.video.preprocessing.config import AppConfig, get_default_config, setup_logging
from ai.video.training.dataset import DeepfakeDataset
from ai.video.training.augmentations import get_val_transforms
from ai.video.training.model import DeepfakeModel
from ai.video.training.validator import Validator


def plot_test_curves(y_true: np.ndarray, y_prob: np.ndarray, output_dir: Path) -> None:
    """
    Plots Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves.
    
    Args:
        y_true (np.ndarray): Binary ground truth labels.
        y_prob (np.ndarray): Probability estimates of the positive class.
        output_dir (Path): Save folder directory.
    """
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC)", fontsize=14, pad=10)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    roc_path = output_dir / "test_roc_curve.png"
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    # 2. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="blue", lw=2, label=f"PR curve (AUC = {pr_auc:.4f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall (PR) Curve", fontsize=14, pad=10)
    plt.legend(loc="lower left")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    pr_path = output_dir / "test_pr_curve.png"
    plt.savefig(pr_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logging.info(f"Saved evaluation curves to {output_dir}")


def evaluate_test_set(config: AppConfig, checkpoint_path: Optional[Path] = None, is_dry_run: bool = False) -> Dict[str, Any]:
    """
    Loads model checkpoints, executes testing, prints scores, and plots curves.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Selected device: {device.type.upper()}")
    
    # 1. Datasets & Dataloader
    val_transform = get_val_transforms(config.pipeline.image_size)
    test_dataset = DeepfakeDataset(config, split="test", transform=val_transform)
    
    if is_dry_run:
        logging.info("Dry-run mode active. Slicing test dataset size.")
        test_dataset.samples = test_dataset.samples[:4]
        
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False
    )
    
    # 2. Model & Weights loading
    model = DeepfakeModel(dropout=config.training.dropout, pretrained=False).to(device)
    
    if checkpoint_path is None:
        checkpoint_path = config.paths.checkpoints_dir / "best_model.pth"
        
    if not is_dry_run:
        logging.info(f"Loading best weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        logging.info("Dry-run: Skipping checkpoint loading, using random model weights.")
        
    criterion = nn.CrossEntropyLoss()
    validator = Validator(model, criterion, device, config)
    
    # 3. Validation Run
    test_metrics = validator.validate(test_loader, epoch=None)
    
    # 4. Generate curves and confusion matrix outputs
    output_dir = config.paths.checkpoints_dir
    
    # Format labels/probs as np arrays for plotting
    y_true = np.array(validator.metric_tracker.all_labels)
    y_prob = np.array(validator.metric_tracker.all_probs)
    
    if len(y_true) > 0:
        plot_test_curves(y_true, y_prob, output_dir)
        validator.save_confusion_matrix(
            test_metrics["confusion_matrix"], 
            output_dir / "test_confusion_matrix.png",
            title="Test Confusion Matrix"
        )
        
    logging.info("\n" + "="*50 + "\nTEST EVALUATION REPORT\n" + "="*50)
    logging.info(f"Loss:               {test_metrics['loss']:.4f}")
    logging.info(f"Accuracy:           {test_metrics['accuracy']:.4f}")
    logging.info(f"Balanced Accuracy:  {test_metrics['balanced_accuracy']:.4f}")
    logging.info(f"Precision:          {test_metrics['precision']:.4f}")
    logging.info(f"Recall:             {test_metrics['recall']:.4f}")
    logging.info(f"F1-Score:           {test_metrics['f1']:.4f}")
    logging.info(f"ROC AUC Score:      {test_metrics['roc_auc']:.4f}")
    logging.info(f"\nClassification Report:\n{test_metrics['classification_report']}")
    logging.info("="*50)
    
    return test_metrics


def main() -> None:
    """
    Main testing entrypoint.
    """
    default_config = get_default_config()
    
    parser = argparse.ArgumentParser(
        description="Authentix - MobileNetV3 Test Set Evaluation Pipeline"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        help="Custom path to model weights checkpoint. Default: checkpoints/best_model.pth"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help=f"Loader batch size override. Default: {default_config.training.batch_size}"
    )
    
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a quick end-to-end self-test/dry-run using limited mock data."
    )
    
    args = parser.parse_args()
    
    if args.self_test:
        logging.info("Starting dry-run verification of test.py")
        from dataclasses import replace
        test_training = replace(default_config.training, batch_size=2, num_workers=0)
        test_config = replace(default_config, training=test_training)
        try:
            evaluate_test_set(test_config, is_dry_run=True)
            # Clean up files created during dry-run validation
            output_dir = test_config.paths.checkpoints_dir
            for file_name in ["test_roc_curve.png", "test_pr_curve.png", "test_confusion_matrix.png"]:
                p = output_dir / file_name
                if p.exists():
                    p.unlink()
                    logging.info(f"Cleaned up mock plot: {file_name}")
            logging.info("test.py module verified successfully.")
            sys.exit(0)
        except Exception as e:
            logging.error(f"test.py verification failed: {e}")
            sys.exit(1)
            
    # Resolve overrides
    ckpt_path = Path(args.checkpoint_path) if args.checkpoint_path else None
    batch_size = args.batch_size if args.batch_size is not None else default_config.training.batch_size
    
    from dataclasses import replace
    new_training = replace(default_config.training, batch_size=batch_size, num_workers=0)
    app_config = replace(default_config, training=new_training)
    
    setup_logging(app_config)
    evaluate_test_set(app_config, ckpt_path, is_dry_run=False)


if __name__ == "__main__":
    main()
