"""Main entry point to execute the staged Audio AI Training Pipeline.

This script parses arguments, initializes configurations, registers seeds, loads datasets and loaders,
instantiates the classifier and optimizer wrappers, and supervises staged transfer learning
transitions (Phase 1 -> Phase 2 -> Phase 3) with automatic checkpointing and resume support.
"""

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor

# Add project root to sys.path to enable absolute imports when run directly
PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ai.audio.training.callbacks import CheckpointSaver, EarlyStopping, TensorBoardLogger
from ai.audio.training.config import TrainingConfig, setup_logger
from ai.audio.training.dataset import AudioDataset, collate_audio_batches
from ai.audio.training.model import AudioClassifier
from ai.audio.training.trainer import AudioTrainer
from ai.audio.training.validator import AudioValidator


def set_seed(seed: int) -> None:
    """Sets random seeds globally across random, numpy, and PyTorch for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_pipeline(resume_training: bool) -> None:
    """Orchestrates staged transfer learning execution for audio deepfake classification.

    Args:
        resume_training: If True, attempts to restore states and resume training from disk.
    """
    config = TrainingConfig()
    config.validate()
    config.create_directories()

    # 1. Initialize Loggers
    main_logger = setup_logger("audio_training", config.logging.log_file, config.logging.log_level)
    main_logger.info("=== Starting Authentix Audio Training Pipeline ===")

    # Register seeds
    set_seed(config.random_seed)
    main_logger.info(f"Registered global random seed: {config.random_seed}")

    # Determine hardware device
    device_name = "cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu"
    device = torch.device(device_name)
    main_logger.info(f"Target execution hardware resolved: {device_name.upper()}")

    # 2. Initialize HuggingFace Processor (Feature Extractor)
    main_logger.info(f"Loading HuggingFace Feature Extractor: {config.model.backbone_name}")
    feature_extractor = AutoFeatureExtractor.from_pretrained(config.model.backbone_name)

    # 3. Initialize Datasets & Loaders
    main_logger.info("Initializing datasets partitions...")
    train_dataset = AudioDataset(config, split="train", processor=feature_extractor)
    val_dataset = AudioDataset(config, split="validation", processor=feature_extractor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_audio_batches,
        num_workers=config.dataloader_num_workers,
        pin_memory=config.pin_memory,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_audio_batches,
        num_workers=config.dataloader_num_workers,
        pin_memory=config.pin_memory
    )

    # 4. Load Audio Classifier model
    model = AudioClassifier(config)
    model.to(device)

    # Calculate and assign class weights to CrossEntropyLoss
    class_weights = None
    if config.class_weights_strategy == "auto":
        class_weights = train_dataset.calculate_class_weights().to(device)
        main_logger.info(f"Calculated training class balance weights: {class_weights}")
    
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=config.optimizer.label_smoothing
    )

    # 5. Initialize Callback Handlers
    checkpointer = CheckpointSaver(config)
    tb_logger = TensorBoardLogger(config)

    # Staged optimization training parameters setup
    start_phase = 1
    start_epoch = 1
    best_val_score = float("inf") if config.checkpoint.monitor_mode == "min" else float("-inf")
    early_stopping_counter = 0

    # 6. Resume training check
    # We must instantiate temporary optimizer and scheduler to load state safely if resuming
    temp_optimizer = AdamW(model.parameters(), lr=config.lr_phase1)
    temp_scheduler = CosineAnnealingLR(temp_optimizer, T_max=config.epochs_phase1)

    resume_metadata = None
    if resume_training:
        resume_metadata = checkpointer.load_resume_state(model, temp_optimizer, temp_scheduler)
        
    if resume_metadata is not None:
        start_epoch = resume_metadata["epoch"] + 1  # Start from next epoch
        start_phase = resume_metadata["phase"]
        best_val_score = resume_metadata["monitor_value"]
        early_stopping_counter = resume_metadata["early_stopping_counter"]
        main_logger.info(
            f"Resuming training loop from Phase {start_phase}, Epoch {start_epoch}. "
            f"Restoring best score register: {best_val_score:.6f}"
        )

    # Define training phases schedule
    phases_setup = {
        1: {
            "name": "Classifier Head Training (Backbone Frozen)",
            "epochs": config.epochs_phase1,
            "lr": config.lr_phase1,
            "freeze_fn": model.freeze_backbone
        },
        2: {
            "name": "Fine-Tuning Last Encoder Layers",
            "epochs": config.epochs_phase2,
            "lr": config.lr_phase2,
            "freeze_fn": model.unfreeze_last_layers
        },
        3: {
            "name": "Fine-Tuning Full Model",
            "epochs": config.epochs_phase3,
            "lr": config.lr_phase3,
            "freeze_fn": model.unfreeze_all
        }
    }

    # 7. Execute Staged Training Phases
    for phase_idx in sorted(phases_setup.keys()):
        if phase_idx < start_phase:
            main_logger.info(f"Skipping Phase {phase_idx} (completed prior to resume).")
            continue

        phase_info = phases_setup[phase_idx]
        epochs_in_phase = phase_info["epochs"]
        phase_lr = phase_info["lr"]

        if epochs_in_phase == 0:
            main_logger.info(f"Phase {phase_idx} configured with 0 epochs. Skipping.")
            continue

        main_logger.info(f"\n>>> Starting Phase {phase_idx}: {phase_info['name']} <<<")
        
        # Apply layer freeze/unfreeze rules
        phase_info["freeze_fn"]()
        model.print_model_summary()

        # Re-initialize optimizer and scheduler for the specific phase parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(
            trainable_params,
            lr=phase_lr,
            weight_decay=config.optimizer.weight_decay,
            betas=config.optimizer.adam_betas,
            eps=config.optimizer.adam_eps
        )
        
        scheduler = None
        if config.scheduler.scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=epochs_in_phase,
                eta_min=config.scheduler.min_lr
            )

        # Restore optimizer/scheduler state if we are in the middle of this phase during resume
        if resume_metadata is not None and phase_idx == start_phase:
            try:
                optimizer.load_state_dict(temp_optimizer.state_dict())
                if scheduler is not None:
                    scheduler.load_state_dict(temp_scheduler.state_dict())
                main_logger.info("Restored optimizer and scheduler state for the resumed phase.")
            except Exception as e:
                main_logger.warning(f"Could not restore optimizer state details: {e}. Using new schedules.")
            # Reset resume metadata after applying once
            resume_metadata = None

        # Initialize EarlyStopping callback
        early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            mode=config.checkpoint.monitor_mode
        )
        early_stopping.best_score = best_val_score
        early_stopping.counter = early_stopping_counter
        # Reset counters for subsequent clean phases
        early_stopping_counter = 0

        # Initialize Trainer and Validator for the phase
        trainer = AudioTrainer(
            config=config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            train_loader=train_loader,
            tb_logger=tb_logger,
            device=device
        )
        validator = AudioValidator(
            config=config,
            model=model,
            criterion=criterion,
            val_loader=val_loader,
            tb_logger=tb_logger,
            device=device
        )

        # Determine target epochs range
        phase_start_epoch = start_epoch if phase_idx == start_phase else 1
        # Reset start_epoch for subsequent phases
        start_epoch = 1

        # Phase Loop
        for epoch in range(phase_start_epoch, epochs_in_phase + 1):
            main_logger.info(f"\n--- Phase {phase_idx} | Epoch {epoch}/{epochs_in_phase} ---")
            
            # 1. Training Epoch Pass
            train_stats = trainer.train_epoch(epoch, phase=phase_idx)

            # 2. Validation Pass
            val_metrics = validator.validate(epoch, phase=phase_idx)
            val_loss = val_metrics["loss"]
            val_score = val_metrics[config.checkpoint.monitor_metric.replace("val_", "")]

            # 3. Scheduler Step
            if scheduler is not None:
                scheduler.step()

            # 4. Checkpoint saving logic
            is_best = False
            if config.checkpoint.monitor_mode == "min":
                if val_score < best_val_score:
                    best_val_score = val_score
                    is_best = True
            else:
                if val_score > best_val_score:
                    best_val_score = val_score
                    is_best = True

            # Save checkpoint state dicts
            checkpointer.save(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                phase=phase_idx,
                monitor_val=val_score,
                is_best=is_best,
                early_stopping_counter=early_stopping.counter
            )

            # 5. Early Stopping check
            if early_stopping(val_score):
                main_logger.warning(f"Early stopping limit hit. Ending Phase {phase_idx} prematurely.")
                break

        # Load best model weights from the completed phase before starting next fine-tuning phase
        if checkpointer.best_model_path.exists():
            main_logger.info(f"Loading best weights configuration from Phase {phase_idx} for future fine-tuning.")
            raw_model = model.model if hasattr(model, "model") else model
            raw_model.load_state_dict(torch.load(checkpointer.best_model_path, map_location="cpu", weights_only=True))

    tb_logger.close()
    main_logger.info("=== Audio AI Training Pipeline Completed ===")


def main() -> None:
    """CLI Entrypoint for training execution."""
    parser = argparse.ArgumentParser(description="Train Authentix Audio Classification model.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from previous checkpointer state files on disk."
    )
    args = parser.parse_args()

    train_pipeline(resume_training=args.resume)


if __name__ == "__main__":
    main()
