"""Audio preprocessing module for production inference.

This module loads audio files (supporting WAV, MP3, FLAC, M4A, OGG), normalizes their
sampling rate to 16 kHz, downmixes to Mono, applies peak normalization, and extracts
numpy input features using the HuggingFace AutoFeatureExtractor for ONNX Runtime consumption.
"""

import logging
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import AutoFeatureExtractor

from ai.audio.inference.config import InferenceConfig


class AudioPreprocessor:
    """Preprocesses raw audio files into standard numpy arrays for ONNX Runtime inference."""

    def __init__(self, config: InferenceConfig):
        """Initializes the AudioPreprocessor.

        Args:
            config: InferenceConfig instance.
        """
        self.config = config
        self.logger = logging.getLogger("audio_inference.preprocessor")
        
        self.logger.info(f"Loading feature extractor processor: {self.config.backbone_name}")
        self.processor = AutoFeatureExtractor.from_pretrained(self.config.backbone_name)

    def preprocess(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
        """Loads, standardizes, and processes an audio file.

        Args:
            file_path: Path to the target audio file.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Containing:
                - input_values: Prepared audio feature array [1, sequence_length].
                - attention_mask: Mask array [1, sequence_length].
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        self.logger.debug(f"Preprocessing file: {file_path.name}")

        # 1. Load Audio Waveform
        try:
            try:
                waveform, sr = torchaudio.load(str(file_path))
            except Exception:
                # Fallback to soundfile loading
                data, sr = sf.read(str(file_path), dtype="float32")
                if len(data.shape) == 1:
                    data = data[np.newaxis, :]
                else:
                    data = data.T  # Shape: (channels, frames)
                waveform = torch.from_numpy(data)
        except Exception as e:
            self.logger.error(f"Failed to load audio file {file_path.name}: {e}")
            raise IOError(f"Could not load or decode audio file: {e}")

        # 2. Downmix to Mono if stereo/multi-channel
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 3. Resample to 16 kHz if necessary
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)

        # 4. Peak Normalization (scale to absolute peak of 0.9, matching training preprocessing)
        max_amplitude = torch.max(torch.abs(waveform)).item()
        if max_amplitude > 0:
            waveform = (waveform / max_amplitude) * 0.9

        # Squeeze to 1D array for the processor
        waveform_1d = waveform.squeeze().numpy()

        # 5. Extract features using processor
        processed = self.processor(
            waveform_1d,
            sampling_rate=16000,
            return_tensors="np"  # Return numpy arrays directly for ONNX Runtime
        )

        input_values = processed.input_values
        
        # If processor returns attention mask, use it, otherwise create standard ones mask
        if "attention_mask" in processed:
            attention_mask = processed.attention_mask
        else:
            attention_mask = np.ones_like(input_values, dtype=np.int64)

        return input_values, attention_mask
