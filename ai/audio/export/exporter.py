"""Model exporter execution module.

This module loads the trained PyTorch classifier, instantiates dynamic audio inputs,
and exports the underlying neural network backbone to a dynamic-shape ONNX format.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor

from ai.audio.export.config import ExportConfig
from ai.audio.training.config import ModelConfig, TrainingConfig
from ai.audio.training.model import AudioClassifier


class AudioModelExporter:
    """Handles PyTorch checkpoint loading and standard ONNX model serialization."""

    def __init__(self, config: ExportConfig):
        """Initializes the AudioModelExporter.

        Args:
            config: ExportConfig instance.
        """
        self.config = config
        self.logger = logging.getLogger("model_export.exporter")
        self.config.create_directories()

    def load_model(self) -> nn.Module:
        """Loads and returns the trained classifier model in eval mode.

        Returns:
            nn.Module: Loaded PyTorch classifier model.
        """
        self.logger.info("Instantiating AudioClassifier model wrapper...")
        
        # Build compatible training config mapping
        training_config = TrainingConfig(
            model=ModelConfig(
                backbone_name=self.config.backbone_name,
                num_labels=2
            )
        )
        classifier = AudioClassifier(training_config)

        checkpoint_path = self.config.checkpoint_path
        self.logger.info(f"Loading checkpoint state dict: {checkpoint_path}")
        
        # Extract underlying model module mapping
        raw_model = classifier.model if hasattr(classifier, "model") else classifier
        
        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            raw_model.load_state_dict(state_dict)
            self.logger.info("PyTorch weights loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise RuntimeError(f"Weights loading failed: {e}")

        # Set evaluation mode
        classifier.eval()
        return classifier

    def export_to_onnx(self) -> Dict[str, Union[float, int, str]]:
        """Converts the PyTorch model to ONNX format and saves to disk.

        Returns:
            Dict: Export execution summary metadata.
        """
        start_time = time.time()
        
        # 1. Load model and set parameters
        classifier = self.load_model()
        
        # We export the underlying HuggingFace model directly to avoid dict output wrapping bugs
        model_to_export = classifier.model if hasattr(classifier, "model") else classifier
        
        # 2. Create dummy input representing a 2-second mono audio segment at 16kHz
        self.logger.info("Constructing dummy input waveform for ONNX trace...")
        sample_rate = 16000
        duration_seconds = 2.0
        seq_length = int(sample_rate * duration_seconds)  # 32000 samples
        
        dummy_input_values = torch.randn(1, seq_length, dtype=torch.float32)
        dummy_attention_mask = torch.ones(1, seq_length, dtype=torch.long)

        # 3. Export model
        onnx_path = self.config.onnx_model_path
        self.logger.info(f"Starting ONNX export trace (opset={self.config.onnx.opset_version})...")
        
        try:
            # We explicitly export with return_dict=False to output a clean logits tuple
            torch.onnx.export(
                model_to_export,
                args=(dummy_input_values, dummy_attention_mask),
                f=str(onnx_path),
                input_names=["input_values", "attention_mask"],
                output_names=["logits"],
                dynamic_axes=self.config.onnx.dynamic_axes,
                opset_version=self.config.onnx.opset_version,
                do_constant_folding=True,
                verbose=False
            )
            self.logger.info(f"ONNX model saved successfully to: {onnx_path.resolve()}")
        except Exception as e:
            self.logger.error(f"ONNX export serialization failed: {e}")
            raise RuntimeError(f"ONNX export execution failed: {e}")

        elapsed_time = time.time() - start_time
        file_size_mb = onnx_path.stat().st_size / (1024 * 1024)

        report = {
            "status": "SUCCESS",
            "opset_version": self.config.onnx.opset_version,
            "onnx_model_path": str(onnx_path.resolve()),
            "file_size_mb": round(file_size_mb, 2),
            "export_duration_seconds": round(elapsed_time, 4),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Save export report
        report_path = self.config.export_report_path
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
            
        self.logger.info(f"Saved export execution report to: {report_path.name}")
        return report
