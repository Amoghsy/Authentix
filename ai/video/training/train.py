"""
train.py

This module contains the primary training script for Authentix video deepfake detection.
It parses command-line overrides, initializes training and validation dataloaders,
computes inverse-frequency class weights dynamically, manages the progressive transfer
learning schedule (Phase 1: freeze backbone, Phase 2: unfreeze last layers, Phase 3: full fine-tuning),
monitors metrics/early stopping, logs training progress, and generates learning curves.
"""

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import time
from typing import Dict, Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Insert project root to sys.path to allow executing the script directly
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.video.preprocessing.config import AppConfig, get_default_config, setup_logging, setup_directories
from ai.video.training.dataset import DeepfakeDataset
from ai.video.training.augmentations import get_train_transforms, get_val_transforms
from ai.video.training.model import DeepfakeModel
from ai.video.training.callbacks import EarlyStopping, CheckpointSaver, TensorBoardLogger
from ai.video.training.trainer import Trainer
from ai.video.training.validator import Validator


def compute_class_weights(dataset: DeepfakeDataset) -> torch.Tensor:
    """
    Computes class weights based on inverse frequency from the dataset samples.
    
    Args:
        dataset (DeepfakeDataset): The dataset to compute weights for.
        
    Returns:
        torch.Tensor: Tensor containing weights for [real, fake].
    """
    labels = [sample[1] for sample in dataset.samples]
    count_real = labels.count(0)
    count_fake = labels.count(1)
    total = len(labels)
    
    if count_real == 0 or count_fake == 0:
        logging.warning("One of the classes has 0 samples. Defaulting class weights to 1.0.")
        return torch.tensor([1.0, 1.0], dtype=torch.float)
        
    # Standard inverse frequency formula: total / (num_classes * class_count)
    weight_real = total / (2.0 * count_real)
    weight_fake = total / (2.0 * count_fake)
    
    weights = torch.tensor([weight_real, weight_fake], dtype=torch.float)
    logging.info(
        f"Imbalance calculation - Real: {count_real} ({count_real/total*100:.1f}%), "
        f"Fake: {count_fake} ({count_fake/total*100:.1f}%). "
        f"Computed class weights: Real={weight_real:.4f}, Fake={weight_fake:.4f}"
    )
    return weights


def plot_curves(history: List[Dict[str, Any]], output_dir: Path) -> None:
    """
    Generates and saves training and validation loss/accuracy curves.
    
    Args:
        history (List[Dict[str, Any]]): List of epoch statistics.
        output_dir (Path): Save folder.
    """
    epochs = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]
    train_accs = [h["train_acc"] for h in history]
    val_accs = [h["val_acc"] for h in history]
    
    # 1. Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label="Train Loss", color="blue", marker="o")
    plt.plot(epochs, val_losses, label="Val Loss", color="orange", marker="x")
    plt.title("Training and Validation Loss", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    loss_path = output_dir / "loss_curve.png"
    plt.savefig(loss_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    # 2. Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accs, label="Train Accuracy", color="blue", marker="o")
    plt.plot(epochs, val_accs, label="Val Accuracy", color="orange", marker="x")
    plt.title("Training and Validation Accuracy", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    acc_path = output_dir / "accuracy_curve.png"
    plt.savefig(acc_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logging.info(f"Loss/Accuracy curves saved to {output_dir}")


def save_validation_plots(y_true: np.ndarray, y_prob: np.ndarray, cm: List[List[int]], output_dir: Path, epoch: int) -> None:
    """
    Generates and saves validation curves and confusion matrix for intermediate epochs.
    """
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Confusion Matrix
    cm_array = np.array(cm)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm_array, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Epoch {epoch} - Val Confusion Matrix", fontsize=12, pad=10)
    plt.colorbar()
    classes = ["Real", "Fake"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    thresh = cm_array.max() / 2.0
    for i in range(cm_array.shape[0]):
        for j in range(cm_array.shape[1]):
            plt.text(
                j, i, format(cm_array[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm_array[i, j] > thresh else "black"
            )
            
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(output_dir / "val_confusion_matrix.png", dpi=150)
    plt.close()
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Epoch {epoch} - Val ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / "val_roc_curve.png", dpi=150)
    plt.close()
    
    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="green", lw=2, label=f"PR Curve (AP = {ap:.4f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Epoch {epoch} - Val Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / "val_pr_curve.png", dpi=150)
    plt.close()


def run_training(config: AppConfig, resume: bool = False, is_dry_run: bool = False) -> None:
    """
    Coordinates progressive model training across Phase 1, Phase 2, and Phase 3.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Selected device: {device.type.upper()}")
    
    # Ensure folders are ready
    setup_directories(config)
    
    # 1. Load Datasets & Apply Augmentations
    logging.info("Initializing datasets and data loaders...")
    train_transform = get_train_transforms(config.pipeline.image_size)
    val_transform = get_val_transforms(config.pipeline.image_size)
    
    train_dataset = DeepfakeDataset(config, split="train", transform=train_transform)
    val_dataset = DeepfakeDataset(config, split="validation", transform=val_transform)
    
    if is_dry_run:
        logging.info("Dry-run mode active. Slicing datasets to speed up verification.")
        train_dataset.samples = train_dataset.samples[:4]
        val_dataset.samples = val_dataset.samples[:2]
    
    use_persistent = config.training.num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=use_persistent,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=use_persistent,
        drop_last=False
    )
    
    # 2. Compute Loss and Class Weights
    label_smoothing = getattr(config.training, "label_smoothing", 0.0)
    if config.training.auto_class_weights:
        class_weights = compute_class_weights(train_dataset).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
    # 3. Instantiate model
    model = DeepfakeModel(dropout=config.training.dropout, pretrained=True).to(device)
    
    # Setup Callbacks
    monitor_metric = getattr(config.training, "early_stopping_metric", "balanced_accuracy").lower()
    monitor_mode = "min" if any(m in monitor_metric for m in ["loss", "error"]) else "max"
    
    saver = CheckpointSaver(config.paths.checkpoints_dir, mode=monitor_mode)
    early_stopping = EarlyStopping(patience=config.training.early_stopping_patience, mode=monitor_mode)
    tb_logger = TensorBoardLogger(config.paths.checkpoints_dir / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    history: List[Dict[str, Any]] = []
    start_epoch = 1
    
    # 4. Handle Resumption if flag is set
    history_file = config.paths.checkpoints_dir / "training_history.json"
    last_ckpt = config.paths.checkpoints_dir / "last_model.pth"
    
    if resume and last_ckpt.exists():
        try:
            # We construct dummy states to load via resume()
            dummy_opt = optim.AdamW(model.parameters(), lr=config.training.lr_phase1)
            dummy_sched = optim.lr_scheduler.CosineAnnealingLR(dummy_opt, T_max=1)
            
            # Temporary trainer to load checkpoint
            tmp_trainer = Trainer(model, dummy_opt, criterion, dummy_sched, device, config)
            start_epoch = tmp_trainer.resume(last_ckpt)
            
            # Load training history json
            if history_file.exists():
                with open(history_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
            # Synchronize best score inside saver
            if history:
                val_key = monitor_metric.replace("val_", "")
                if f"val_{val_key}" in history[0]:
                    history_key = f"val_{val_key}"
                elif val_key in history[0]:
                    history_key = val_key
                else:
                    history_key = "val_loss"
                    
                scores = [h[history_key] for h in history]
                if monitor_mode == "min":
                    best_idx = np.argmin(scores)
                    saver.best_score = min(scores)
                else:
                    best_idx = np.argmax(scores)
                    saver.best_score = max(scores)
                    
                early_stopping.best_score = saver.best_score
                best_epoch = best_idx + 1
                early_stopping.counter = len(history) - best_epoch
        except Exception as e:
            logging.error(f"Failed to resume checkpoint: {e}. Starting training from scratch.")
            start_epoch = 1
            
    # Calculate Phase epoch bounds
    p1_end = config.training.epochs_phase1
    p2_end = p1_end + config.training.epochs_phase2
    p3_end = p2_end + config.training.epochs_phase3
    total_epochs = p3_end
    
    logging.info(
        f"Training Plan - Phase 1 (Classifier): {p1_end} epochs | "
        f"Phase 2 (Last Blocks): {config.training.epochs_phase2} epochs | "
        f"Phase 3 (Full Network): {config.training.epochs_phase3} epochs"
    )

    # 5. Training Loop
    active_phase = 0
    opt = None
    sched = None
    trainer = None
    validator = Validator(model, criterion, device, config)

    for epoch in range(start_epoch, total_epochs + 1):
        # Determine active Phase
        if epoch <= p1_end:
            phase = 1
            lr = config.training.lr_phase1
        elif epoch <= p2_end:
            phase = 2
            lr = config.training.lr_phase2
        else:
            phase = 3
            lr = config.training.lr_phase3
            
        # If phase changed (or optimizer not initialized), setup model parameters, optimizer, and scheduler
        if phase != active_phase:
            active_phase = phase
            logging.info(f"⚡ Entering Training Phase {active_phase} (Epoch {epoch}) | Target LR: {lr}")
            
            if active_phase == 1:
                model.freeze_backbone()
            elif active_phase == 2:
                model.unfreeze_last_blocks(num_layers=3)
            elif active_phase == 3:
                model.unfreeze_all()
                
            # Log parameter counts
            tot_p, train_p = model.count_parameters()
            logging.info(f"Phase {active_phase} Parameter Summary - Total: {tot_p:,}, Trainable: {train_p:,}")
            
            # Recreate optimizer and scheduler to bind to the newly unfrozen parameters
            opt = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr,
                weight_decay=config.training.weight_decay
            )
            
            # Recreate scheduler based on configuration
            if config.training.lr_scheduler_type == "cosine":
                # Max steps for the CosineAnnealing is the remainder of the phase epochs
                if active_phase == 1:
                    t_max = p1_end - epoch + 1
                elif active_phase == 2:
                    t_max = p2_end - epoch + 1
                else:
                    t_max = p3_end - epoch + 1
                sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, t_max))
            elif config.training.lr_scheduler_type == "plateau":
                sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
            else:
                sched = None
                
            trainer = Trainer(model, opt, criterion, sched, device, config)
            
        # Assert trainer is initialized
        assert trainer is not None, "Trainer initialization failed."
        
        # 6. Execute Train & Val step
        train_stats = trainer.train_epoch(train_loader, epoch)
        val_stats = validator.validate(val_loader, epoch)
        
        # Generate epoch-level validation plots
        plots_dir = config.paths.checkpoints_dir / "plots"
        y_true = np.array(validator.metric_tracker.all_labels)
        y_prob = np.array(validator.metric_tracker.all_probs)
        save_validation_plots(y_true, y_prob, val_stats["confusion_matrix"], plots_dir, epoch)
        
        # 7. Update scheduler
        curr_lr = opt.param_groups[0]["lr"]
        if sched is not None:
            if isinstance(sched, optim.lr_scheduler.ReduceLROnPlateau):
                sched.step(val_stats["loss"])
            else:
                sched.step()
                
        # 8. Log TensorBoard scalars
        tb_logger.log_epoch(epoch, train_stats, val_stats, curr_lr)
        
        # Record stats in history
        history_record = {
            "epoch": epoch,
            "phase": active_phase,
            "lr": curr_lr,
            "train_loss": train_stats["loss"],
            "train_acc": train_stats["accuracy"],
            "val_loss": val_stats["loss"],
            "val_acc": val_stats["accuracy"],
            "val_f1": val_stats["f1"],
            "val_auc": val_stats["roc_auc"],
            "duration_seconds": train_stats["duration_seconds"]
        }
        history.append(history_record)
        
        # Save training history JSON file
        try:
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=4)
        except Exception as e:
            logging.error(f"Failed to write history file: {e}")
            
        # 9. Save Checkpoint
        val_key = monitor_metric.replace("val_", "")
        score = val_stats.get(val_key, val_stats.get("loss"))
        
        saver.save(
            model=model,
            optimizer=opt,
            scheduler=sched,
            epoch=epoch,
            metrics=val_stats,
            score=score
        )
        
        # 10. Check EarlyStopping
        stop_training = early_stopping.step(score)
        if stop_training:
            logging.warning(f"Early stopping triggered at epoch {epoch}. Terminating training loop.")
            break
            
    # Close TensorBoard logging session
    tb_logger.close()
    
    # 11. Plot final training curves
    if history:
        plot_curves(history, config.paths.checkpoints_dir)
        
    # Restore best model weights upon completion of training run
    best_pth = config.paths.checkpoints_dir / "best_model.pth"
    if best_pth.exists():
        logging.info(f"Restoring best model weights from {best_pth}")
        checkpoint = torch.load(best_pth, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        
    logging.info("Training pipeline completed successfully.")


def main() -> None:
    """
    Main entrypoint parsing CLI overrides and launching train routine.
    """
    default_config = get_default_config()
    
    parser = argparse.ArgumentParser(
        description="Authentix - MobileNetV3 Training pipeline"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help=f"Batch size for dataloaders. Default: {default_config.training.batch_size}"
    )
    parser.add_argument(
        "--epochs-phase1",
        type=int,
        help=f"Classifier training epochs. Default: {default_config.training.epochs_phase1}"
    )
    parser.add_argument(
        "--epochs-phase2",
        type=int,
        help=f"Fine-tuning phase 2 epochs. Default: {default_config.training.epochs_phase2}"
    )
    parser.add_argument(
        "--epochs-phase3",
        type=int,
        help=f"Optional phase 3 epochs. Default: {default_config.training.epochs_phase3}"
    )
    parser.add_argument(
        "--lr-phase1",
        type=float,
        help=f"Learning rate for Phase 1. Default: {default_config.training.lr_phase1}"
    )
    parser.add_argument(
        "--lr-phase2",
        type=float,
        help=f"Learning rate for Phase 2. Default: {default_config.training.lr_phase2}"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Attempt to resume training state from checkpoints/last_model.pth"
    )
    
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a quick end-to-end self-test/dry-run using limited mock data."
    )
    
    args = parser.parse_args()
    
    if args.self_test:
        logging.info("Starting dry-run verification of train.py")
        from dataclasses import replace
        test_training = replace(
            default_config.training,
            batch_size=2,
            epochs_phase1=1,
            epochs_phase2=1,
            epochs_phase3=0,
            num_workers=0 # CPU threading safety in dry run
        )
        test_config = replace(default_config, training=test_training)
        try:
            run_training(test_config, resume=False, is_dry_run=True)
            logging.info("train.py module verified successfully.")
            sys.exit(0)
        except Exception as e:
            logging.error(f"train.py verification failed: {e}")
            sys.exit(1)
            
    # Reassemble configurations
    batch_size = args.batch_size if args.batch_size is not None else default_config.training.batch_size
    p1 = args.epochs_phase1 if args.epochs_phase1 is not None else default_config.training.epochs_phase1
    p2 = args.epochs_phase2 if args.epochs_phase2 is not None else default_config.training.epochs_phase2
    p3 = args.epochs_phase3 if args.epochs_phase3 is not None else default_config.training.epochs_phase3
    lr1 = args.lr_phase1 if args.lr_phase1 is not None else default_config.training.lr_phase1
    lr2 = args.lr_phase2 if args.lr_phase2 is not None else default_config.training.lr_phase2
    
    from dataclasses import replace
    new_training_config = replace(
        default_config.training,
        batch_size=batch_size,
        epochs_phase1=p1,
        epochs_phase2=p2,
        epochs_phase3=p3,
        lr_phase1=lr1,
        lr_phase2=lr2
    )
    
    app_config = replace(default_config, training=new_training_config)
    
    setup_logging(app_config)
    run_training(app_config, resume=args.resume)


if __name__ == "__main__":
    main()
