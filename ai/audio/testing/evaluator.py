"""Model evaluation execution engine.

This module coordinates loading the trained model checkpoint, disabling gradients,
setting evaluation modes, running batched inference, collecting logits and predicted probabilities,
and invoking metrics computation.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoFeatureExtractor

from ai.audio.testing.config import TestingConfig, setup_logger
from ai.audio.testing.metrics import AudioTestMetrics
from ai.audio.training.model import AudioClassifier


class AudioEvaluator:
    """Orchestrates checkpoint loading and batch inference for model evaluation."""

    def __init__(self, config: TestingConfig):
        """Initializes the AudioEvaluator.

        Args:
            config: TestingConfig instance.
        """
        self.config = config
        self.logger = logging.getLogger("audio_testing.evaluator")

        # Set hardware device
        device_name = "cuda" if torch.cuda.is_available() and self.config.device.device == "cuda" else "cpu"
        self.device = torch.device(device_name)
        self.logger.info(f"Target execution hardware resolved for testing: {device_name.upper()}")

        # Load HuggingFace processor
        self.logger.info(f"Loading processor for: {self.config.backbone_name}")
        self.processor = AutoFeatureExtractor.from_pretrained(self.config.backbone_name)

        # Load model structure and checkpoint
        self.model = self._load_model_checkpoint()
        self.model.to(self.device)
        self.model.eval()

        self.metrics_calculator = AudioTestMetrics()

    def _load_model_checkpoint(self) -> nn.Module:
        """Instantiates AudioClassifier and loads the best_model.pth state dict.

        Returns:
            nn.Module: The configured PyTorch model wrapper.
        """
        self.logger.info("Initializing AudioClassifier wrapper...")
        from ai.audio.training.config import TrainingConfig, ModelConfig
        training_config = TrainingConfig(
            model=ModelConfig(
                backbone_name=self.config.backbone_name,
                num_labels=2
            )
        )
        classifier = AudioClassifier(training_config)

        checkpoint_path = self.config.checkpoint_path
        self.logger.info(f"Loading weights state dict from checkpoint: {checkpoint_path}")
        
        # Unwrap model module matching training save structure
        raw_model = classifier.model if hasattr(classifier, "model") else classifier
        
        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            raw_model.load_state_dict(state_dict)
            self.logger.info("Checkpoint weights loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint weights: {e}")
            raise RuntimeError(f"Failed to load weights state dict: {e}")

        return classifier

    def evaluate(self, test_loader: DataLoader) -> Tuple[Dict, pd.DataFrame, np.ndarray, np.ndarray]:
        """Runs the validation evaluation pass on the test dataset.

        Args:
            test_loader: DataLoader for the test split.

        Returns:
            Tuple containing:
                - metrics: Dictionary of computed classification metrics.
                - predictions_df: Pandas DataFrame containing detailed prediction rows.
                - all_logits: Array of raw model logits.
                - all_probs: Array of predicted Softmax probabilities.
        """
        self.logger.info(f"Starting test evaluation loop on {len(test_loader.dataset)} samples...")

        logits_list = []
        labels_list = []
        preds_list = []
        probs_list = []
        paths_list = []

        # Disable gradient tracking
        with torch.no_grad():
            progress_bar = tqdm(
                test_loader,
                total=len(test_loader),
                desc="Evaluating test split",
                leave=False
            )
            
            for batch in progress_bar:
                # Move inputs to target device
                input_values = batch["input_values"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)
                file_paths = batch["file_paths"]

                # Forward pass inside autocast (Mixed Precision compatibility)
                with torch.amp.autocast(device_type=self.device.type, enabled=self.config.device.mixed_precision):
                    outputs = self.model(
                        input_values=input_values,
                        attention_mask=attention_mask
                    )
                    logits = outputs["logits"]

                # Apply Softmax to obtain probabilities
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)

                # Move values back to CPU and convert to numpy
                logits_list.append(logits.detach().cpu().numpy())
                labels_list.append(labels.detach().cpu().numpy())
                preds_list.append(preds.detach().cpu().numpy())
                probs_list.append(probs.detach().cpu().numpy())
                paths_list.extend(file_paths)

        # Concatenate batch lists
        all_logits = np.concatenate(logits_list, axis=0)
        all_labels = np.concatenate(labels_list, axis=0)
        all_preds = np.concatenate(preds_list, axis=0)
        all_probs = np.concatenate(probs_list, axis=0)

        # Compute positive class probability (class 1: 'fake')
        # Handle shape safety
        if all_probs.shape[1] > 1:
            positive_probs = all_probs[:, 1]
        else:
            positive_probs = all_probs[:, 0]

        # Calculate final metrics
        self.logger.info("Inference complete. Calculating final metrics...")
        metrics = self.metrics_calculator.compute_all_metrics(
            y_true=all_labels,
            y_pred=all_preds,
            y_probs=positive_probs
        )

        # Construct predictions DataFrame
        predictions_df = pd.DataFrame({
            "file_path": paths_list,
            "true_label": all_labels,
            "predicted_label": all_preds,
            "probability_fake": positive_probs,
            "logit_real": all_logits[:, 0],
            "logit_fake": all_logits[:, 1] if all_logits.shape[1] > 1 else np.nan
        })

        self.logger.info("Testing metrics computation complete.")
        return metrics, predictions_df, all_logits, all_probs
