"""Model predictor execution module for standalone audio inference.

This module initializes the ONNX Runtime session, manages inference pipelines
for single audio files, calculates predicted probabilities via Softmax, and formats
outputs into the standard JSON API required by the Fusion AI Engine.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Union

import sys
import numpy as np
import onnxruntime as ort

# Add project root to sys.path to enable absolute imports when run directly
PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ai.audio.inference.config import InferenceConfig
from ai.audio.inference.preprocessor import AudioPreprocessor


class AudioPredictor:
    """Loads ONNX models and executes deepfake detection on single audio uploads."""

    def __init__(self, config: InferenceConfig):
        """Initializes the AudioPredictor.

        Args:
            config: InferenceConfig instance.
        """
        self.config = config
        self.logger = logging.getLogger("audio_inference.predictor")
        
        # Validate configuration and verify model existence
        self.config.validate()
        self.config.create_directories()

        self.preprocessor = AudioPreprocessor(self.config)
        self.session = self._initialize_onnx_session()

    def _initialize_onnx_session(self) -> ort.InferenceSession:
        """Configures execution providers and loads the ONNX runtime model.

        Returns:
            ort.InferenceSession: The loaded ONNX session.
        """
        model_path = self.config.model_path
        self.logger.info(f"Loading ONNX model for prediction: {model_path.resolve()}")
        
        # Select execution providers (fallback to CPU if CUDA unavailable or fails)
        providers = ["CPUExecutionProvider"]
        if self.config.device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
            self.logger.info("ONNX Runtime configured with CUDA execution support.")
        else:
            self.logger.info("ONNX Runtime configured with CPU execution support.")

        try:
            session = ort.InferenceSession(str(model_path), providers=providers)
            return session
        except Exception as e:
            self.logger.error(f"Failed to initialize ONNX Runtime session: {e}")
            raise RuntimeError(f"ONNX session initialization failed: {e}")

    def predict(self, file_path: Union[str, Path]) -> Dict[str, Union[str, float]]:
        """Runs end-to-end classification on a single audio recording.

        Args:
            file_path: Path of the audio file.

        Returns:
            Dict: Classification result matching Fusion AI Engine schema:
                {
                    "prediction": "Fake" | "Real",
                    "confidence": float,
                    "fake_probability": float,
                    "real_probability": float,
                    "latency_ms": float
                }
        """
        start_time = time.perf_counter()
        
        # 1. Run preprocessing
        try:
            input_values, attention_mask = self.preprocessor.preprocess(file_path)
        except Exception as e:
            self.logger.error(f"Preprocessing failed for {file_path}: {e}")
            raise

        # 2. Run ONNX Inference
        onnx_inputs = {
            "input_values": input_values,
            "attention_mask": attention_mask
        }
        
        try:
            onnx_outputs = self.session.run(["logits"], onnx_inputs)
            logits = onnx_outputs[0][0]  # Unpack batch dimension
        except Exception as e:
            self.logger.error(f"ONNX inference run failed: {e}")
            raise RuntimeError(f"Inference execution failed: {e}")

        # 3. Calculate Softmax probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        real_prob = float(probs[0])
        fake_prob = float(probs[1]) if len(probs) > 1 else 0.0

        # 4. Classify based on the threshold configuration
        if fake_prob >= self.config.confidence_threshold:
            prediction = "Fake"
            confidence = fake_prob
        else:
            prediction = "Real"
            confidence = real_prob

        # Measure latency in milliseconds
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        result = {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "fake_probability": round(fake_prob, 4),
            "real_probability": round(real_prob, 4),
            "latency_ms": round(latency_ms, 2)
        }

        self.logger.debug(f"Prediction for {Path(file_path).name}: {result}")
        return result


def main() -> None:
    """CLI Entrypoint for running single-file predictions."""
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="Run single-file audio deepfake prediction.")
    parser.add_argument("file_path", type=str, help="Path to the audio file.")
    args = parser.parse_args()

    config = InferenceConfig()
    predictor = AudioPredictor(config)
    
    try:
        result = predictor.predict(args.file_path)
        print(json.dumps(result, indent=4))
    except Exception as e:
        print(f"Prediction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
