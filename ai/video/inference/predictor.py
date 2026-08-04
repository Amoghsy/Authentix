"""Model predictor execution module for video deepfake detection.

This module initializes the ONNX Runtime session, manages inference pipelines
for single video files, aggregates frame-level predictions to produce video-level
probabilities, measures latency, and structures outputs matching the Fusion AI schema.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Union

import numpy as np
import onnxruntime as ort

import sys
# Configure path references to enable direct execution
PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ai.video.inference.config import InferenceConfig
from ai.video.inference.preprocessor import VideoPreprocessor


class VideoPredictor:
    """Loads ONNX models and executes deepfake detection on video files."""

    def __init__(self, config: InferenceConfig):
        """Initializes the VideoPredictor.

        Args:
            config: InferenceConfig instance.
        """
        self.config = config
        self.logger = logging.getLogger("video_inference.predictor")

        # Validate configuration and verify model existence
        self.config.validate()
        self.config.create_directories()

        self.preprocessor = VideoPreprocessor(self.config)
        self.session = self._initialize_onnx_session()

        # Input and output node names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _initialize_onnx_session(self) -> ort.InferenceSession:
        """Configures execution providers and loads the ONNX runtime model.

        Returns:
            ort.InferenceSession: Loaded ONNX session.
        """
        model_path = self.config.model_path
        self.logger.info(f"Loading Video ONNX model for prediction: {model_path.resolve()}")

        # Configure execution providers (fallback to CPU if CUDA is unavailable)
        providers = ["CPUExecutionProvider"]
        if self.config.onnx.device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
            self.logger.info("ONNX Runtime configured with CUDA execution support.")
        else:
            self.logger.info("ONNX Runtime configured with CPU execution support.")

        # Set session options
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = self.config.onnx.intra_op_num_threads
        session_options.inter_op_num_threads = self.config.onnx.inter_op_num_threads

        try:
            session = ort.InferenceSession(
                str(model_path),
                sess_options=session_options,
                providers=providers
            )
            return session
        except Exception as e:
            self.logger.error(f"Failed to initialize ONNX Runtime session: {e}")
            raise RuntimeError(f"ONNX session initialization failed: {e}")

    def predict(self, file_path: Union[str, Path]) -> Dict[str, Union[str, float, int]]:
        """Runs end-to-end classification on a video file.

        Args:
            file_path: Path of the video file.

        Returns:
            Dict: Classification result matching Fusion AI Engine schema:
                {
                    "prediction": "Fake" | "Real",
                    "confidence": float,
                    "fake_probability": float,
                    "real_probability": float,
                    "frames_processed": int,
                    "latency_ms": float
                }
        """
        start_time = time.perf_counter()

        # 1. Run preprocessing (extract batch of frames and timestamps)
        try:
            batch_tensor, frame_timestamps = self.preprocessor.preprocess(file_path)
        except Exception as e:
            self.logger.error(f"Preprocessing failed for {file_path}: {e}")
            raise

        frames_processed = batch_tensor.shape[0]

        # 2. Run ONNX Inference (all frames in a single batch call)
        onnx_inputs = {self.input_name: batch_tensor}
        try:
            onnx_outputs = self.session.run([self.output_name], onnx_inputs)
            logits = onnx_outputs[0]  # Shape: [num_frames, 2]
        except Exception as e:
            self.logger.error(f"ONNX inference run failed: {e}")
            raise RuntimeError(f"Inference execution failed: {e}")

        # 3. Calculate Softmax probabilities on frame logits
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        frame_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)  # Shape: [num_frames, 2]

        # 4. Aggregate frame probabilities (mean pooling)
        video_probs = np.mean(frame_probs, axis=0)  # Shape: [2,]

        real_prob = float(video_probs[0])
        fake_prob = float(video_probs[1]) if len(video_probs) > 1 else 0.0

        # 5. Classify based on confidence threshold
        if fake_prob >= self.config.prediction.confidence_threshold:
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
            "frames_processed": frames_processed,
            "latency_ms": round(latency_ms, 2)
        }

        # 6. Generate visualizations if enabled
        if self.config.visualization.enable_plots:
            try:
                from ai.video.inference.visualization import VideoInferenceVisualizer
                visualizer = VideoInferenceVisualizer(self.config)
                visualizer.generate_all_plots(
                    video_name=Path(file_path).name,
                    frame_timestamps=frame_timestamps,
                    frame_probs=frame_probs,
                    prediction=prediction,
                    overall_confidence=confidence
                )
            except Exception as e:
                self.logger.warning(f"Visualization generation failed for {Path(file_path).name}: {e}")

        self.logger.debug(f"Prediction for {Path(file_path).name}: {result}")
        return result


def main() -> None:
    """CLI Entrypoint for running single video predictions."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Run single-file video deepfake prediction.")
    parser.add_argument("file_path", type=str, help="Path to the video file.")
    args = parser.parse_args()

    config = InferenceConfig()
    predictor = VideoPredictor(config)

    try:
        result = predictor.predict(args.file_path)
        print(json.dumps(result, indent=4))
    except Exception as e:
        print(f"Prediction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
