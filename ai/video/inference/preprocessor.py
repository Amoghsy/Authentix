"""Video preprocessing module for production inference.

This module loads video files, extracts a uniform sample of frames, resizes,
normalizes, and converts them to standard NumPy batches ready for ONNX inference.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np

from ai.video.inference.config import InferenceConfig


class VideoPreprocessor:
    """Preprocesses raw video files into standardized batches of normalized frame arrays."""

    def __init__(self, config: InferenceConfig):
        """Initializes the VideoPreprocessor.

        Args:
            config: InferenceConfig instance.
        """
        self.config = config
        self.logger = logging.getLogger("video_inference.preprocessor")
        self.target_size = self.config.prediction.input_size

        # Standard ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, List[float]]:
        """Loads a video, extracts sampled frames, and normalizes them into a batch.

        Args:
            file_path: Path to the target video file.

        Returns:
            Tuple[np.ndarray, List[float]]: Containing:
                - batch_tensor: Normalized frames batch of shape [num_frames, 3, H, W].
                - frame_timestamps: List of timestamp positions for each sampled frame in seconds.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Video file not found: {file_path}")

        # Open video capture
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            raise IOError(f"Failed to open video file (codec or corrupt file): {file_path.name}")

        # Retrieve video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        self.logger.debug(f"Video loaded: {file_path.name} | Total Frames: {total_frames} | FPS: {fps}")

        if total_frames <= 0 or fps <= 0:
            cap.release()
            raise IOError(f"Invalid video structure or empty frame stream in: {file_path.name}")

        # Determine target frames to process
        max_frames = self.config.prediction.max_frames
        sampling_rate = self.config.prediction.frame_sampling_rate

        # Apply uniform sampling across the entire video duration
        # Determine the subset of indices to fetch
        usable_frame_indices = list(range(0, total_frames, sampling_rate))
        usable_count = len(usable_frame_indices)

        if usable_count <= max_frames:
            target_indices = usable_frame_indices
        else:
            # Linearly sample index positions uniformly
            indices_float = np.linspace(0, usable_count - 1, max_frames)
            target_indices = [usable_frame_indices[int(round(i))] for i in indices_float]

        frames = []
        timestamps = []

        # Read frames sequentially or seek where possible (sequential reading is safer across formats)
        current_idx = 0
        target_set = set(target_indices)
        max_target = max(target_set) if target_set else 0

        while cap.isOpened() and current_idx <= max_target:
            ret, frame = cap.read()
            if not ret:
                break

            if current_idx in target_set:
                # Get frame timestamp in seconds
                msec_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
                sec_timestamp = msec_timestamp / 1000.0 if msec_timestamp > 0 else (current_idx / fps)
                
                # Preprocess single frame
                try:
                    processed_frame = self._preprocess_frame(frame)
                    frames.append(processed_frame)
                    timestamps.append(sec_timestamp)
                except Exception as e:
                    self.logger.warning(f"Failed to preprocess frame {current_idx} in {file_path.name}: {e}")

            current_idx += 1

        cap.release()

        # Handle zero-frames edge cases
        if not frames:
            raise IOError(f"No valid frames could be decoded from video: {file_path.name}")

        # Stack into a batch array: Shape [num_frames, 3, Height, Width]
        batch_tensor = np.stack(frames, axis=0)
        self.logger.debug(f"Preprocessed batch generated. Shape: {batch_tensor.shape}")

        return batch_tensor, timestamps

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Applies resize, channel conversions, normalization, and transposes layout.

        Args:
            frame: OpenCV BGR frame array of shape [H, W, 3].

        Returns:
            np.ndarray: Preprocessed frame of shape [3, target_H, target_W].
        """
        # 1. Resize
        frame_resized = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)

        # 2. Convert channel layout from BGR (OpenCV) to RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # 3. Normalize: Scale to [0, 1] and apply ImageNet mean and std
        frame_norm = (frame_rgb.astype(np.float32) / 255.0 - self.mean) / self.std

        # 4. Transpose layout from HWC to CHW
        frame_transposed = np.transpose(frame_norm, (2, 0, 1))

        return frame_transposed
