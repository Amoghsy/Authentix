"""Numerical correctness verifier module for ONNX models.

This module loads the exported ONNX model and the original PyTorch weights,
runs execution sweeps on identical random dummy inputs with variable shapes,
calculates discrepancies, and enforces strict precision tolerances.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

from ai.audio.export.config import ExportConfig


class AudioONNXVerifier:
    """Validates dynamic axis shapes and evaluates numerical tolerance alignments."""

    def __init__(self, config: ExportConfig):
        """Initializes the AudioONNXVerifier.

        Args:
            config: ExportConfig instance.
        """
        self.config = config
        self.logger = logging.getLogger("model_export.verifier")

    def verify_onnx_model(self, pytorch_model: nn.Module) -> Dict[str, Union[float, bool, str]]:
        """Compares outputs from PyTorch and ONNX models on test sweeps.

        Args:
            pytorch_model: Loaded PyTorch classifier wrapper.

        Returns:
            Dict: Verification performance metrics dictionary.
        """
        onnx_model_path = self.config.onnx_model_path
        self.logger.info(f"Loading ONNX model for verification: {onnx_model_path.name}")
        
        if not onnx_model_path.exists():
            raise FileNotFoundError(f"ONNX model file missing for verification: {onnx_model_path}")

        # Initialize ONNX Runtime Inference Session (using CPU provider for safety and portability)
        try:
            session = ort.InferenceSession(str(onnx_model_path), providers=["CPUExecutionProvider"])
        except Exception as e:
            self.logger.error(f"Failed to load ONNX model session: {e}")
            raise RuntimeError(f"ONNX session initialization failed: {e}")

        # Put PyTorch model in evaluation mode
        pytorch_model.eval()
        raw_pytorch_model = pytorch_model.model if hasattr(pytorch_model, "model") else pytorch_model

        # Run verification checks across different dynamic batch size and sequence length combinations
        # Test Case 1: Batch size 1, Sequence length 32000 (2 seconds)
        # Test Case 2: Batch size 4, Sequence length 16000 (1 second)
        test_shapes = [
            (1, 32000),
            (4, 16000)
        ]

        verification_results = []
        overall_passed = True

        for idx, (batch_size, seq_len) in enumerate(test_shapes, 1):
            self.logger.info(f"Running shape test case {idx}: batch_size={batch_size}, sequence_length={seq_len}")
            
            # Generate random test inputs
            x_np = np.random.randn(batch_size, seq_len).astype(np.float32)
            mask_np = np.ones((batch_size, seq_len)).astype(np.int64)

            # 1. Run PyTorch Inference
            x_torch = torch.from_numpy(x_np)
            mask_torch = torch.from_numpy(mask_np)
            
            with torch.no_grad():
                py_outputs = raw_pytorch_model(x_torch, attention_mask=mask_torch)
                py_logits = py_outputs.logits.numpy()

            # 2. Run ONNX Runtime Inference
            onnx_inputs = {
                "input_values": x_np,
                "attention_mask": mask_np
            }
            onnx_outputs = session.run(["logits"], onnx_inputs)
            onnx_logits = onnx_outputs[0]

            # 3. Calculate metrics
            max_abs_error = np.max(np.abs(py_logits - onnx_logits))
            
            # Relative error (avoiding division by zero)
            denom = np.abs(py_logits) + 1e-8
            max_rel_error = np.max(np.abs(py_logits - onnx_logits) / denom)

            # Softmax differences
            py_probs = self._softmax(py_logits)
            onnx_probs = self._softmax(onnx_logits)
            max_prob_diff = np.max(np.abs(py_probs - onnx_probs))

            # Class predictions matching
            py_preds = np.argmax(py_logits, axis=-1)
            onnx_preds = np.argmax(onnx_logits, axis=-1)
            predictions_match = bool(np.array_equal(py_preds, onnx_preds))

            # Validate tolerance bounds
            case_passed = (
                max_abs_error <= self.config.abs_tolerance and
                max_rel_error <= self.config.rel_tolerance and
                predictions_match
            )
            
            if not case_passed:
                overall_passed = False
                self.logger.warning(
                    f"Shape test case {idx} FAILED precision thresholds. "
                    f"Max Abs Error: {max_abs_error:.6f} (tolerance={self.config.abs_tolerance:.6f}), "
                    f"Max Rel Error: {max_rel_error:.6f} (tolerance={self.config.rel_tolerance:.6f}), "
                    f"Predictions Match: {predictions_match}"
                )
            else:
                self.logger.info(
                    f"Shape test case {idx} PASSED. "
                    f"Max Abs Error: {max_abs_error:.6e}, "
                    f"Max Rel Error: {max_rel_error:.6e}"
                )

            verification_results.append({
                "test_case": idx,
                "batch_size": batch_size,
                "sequence_length": seq_len,
                "max_absolute_error": float(max_abs_error),
                "max_relative_error": float(max_rel_error),
                "max_probability_difference": float(max_prob_diff),
                "predictions_equal": predictions_match,
                "status": "PASSED" if case_passed else "FAILED"
            })

        # Compile final summary
        summary = {
            "status": "PASSED" if overall_passed else "FAILED",
            "absolute_tolerance": self.config.abs_tolerance,
            "relative_tolerance": self.config.rel_tolerance,
            "overall_equivalence_match": overall_passed,
            "test_cases": verification_results
        }

        # Save verification report
        report_path = self.config.verification_report_path
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)
            
        self.logger.info(f"Saved verification validation report to: {report_path.name}")

        if not overall_passed:
            raise ValueError("ONNX verification failed: Model outputs exceed numerical difference tolerance.")

        return summary

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Helper to calculate numerically stable Softmax in numpy."""
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)
