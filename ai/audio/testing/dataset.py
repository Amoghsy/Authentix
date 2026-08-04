"""Dataset loading and feature extraction module for evaluation.

This module reads dataset manifests, loads test partition audio waveforms,
applies the feature extractor, and prepares items for batch collation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor

from ai.audio.testing.config import TestingConfig


class AudioTestDataset(Dataset):
    """PyTorch Dataset wrapper for evaluating audio classifier on the test split."""

    def __init__(self, config: TestingConfig, processor: AutoFeatureExtractor):
        """Initializes the AudioTestDataset.

        Args:
            config: TestingConfig instance.
            processor: HuggingFace AutoFeatureExtractor.
        """
        self.config = config
        self.processor = processor
        self.logger = logging.getLogger("audio_testing.dataset")

        # Load label mapping
        self.label_mapping = self._load_label_mapping()
        
        # Load and filter manifest
        self.manifest_data = self._load_and_filter_manifest()

    def _load_label_mapping(self) -> Dict[str, int]:
        """Loads label mapping from JSON file or falls back to default.

        Returns:
            Dict[str, int]: Label mapping dictionary.
        """
        if self.config.label_mapping_path.exists():
            try:
                with open(self.config.label_mapping_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load label mapping from file: {e}. Using defaults.")
        return {"real": 0, "fake": 1}

    def _load_and_filter_manifest(self) -> List[Dict]:
        """Reads dataset_manifest.csv and filters rows belonging ONLY to the test split.

        Returns:
            List[Dict]: List of metadata records for the test partition.
        """
        if not self.config.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found at: {self.config.manifest_path}")

        try:
            df = pd.read_csv(self.config.manifest_path)
        except Exception as e:
            raise IOError(f"Failed to read manifest file: {e}")

        # Check required columns
        required_cols = {"file_path", "label", "split"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Manifest is missing required columns: {missing_cols}")

        # Filter strictly by the test split
        filtered_df = df[df["split"].str.lower() == "test"]
        records = filtered_df.to_dict("records")
        
        self.logger.info(f"Loaded {len(records)} files for split 'test'")
        if len(records) == 0:
            self.logger.warning("The test split is empty! Please verify dataset_manifest.csv splits.")

        return records

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.manifest_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Loads and processes the test audio file at index.

        Args:
            idx: Index in manifest.

        Returns:
            Dict[str, Tensor]: Contains:
                - 'input_values': Tensor of processed audio features.
                - 'attention_mask': Attention mask tensor.
                - 'label': Class label index tensor.
        """
        record = self.manifest_data[idx]
        file_path = Path(record["file_path"])
        label_str = record["label"]
        
        label_idx = self.label_mapping[label_str.lower()]

        # Load Audio Waveform
        try:
            try:
                waveform, sr = torchaudio.load(str(file_path))
            except Exception:
                # Fallback to soundfile loading
                data, sr = sf.read(str(file_path), dtype="float32")
                if len(data.shape) == 1:
                    data = data[np.newaxis, :]
                else:
                    data = data.T
                waveform = torch.from_numpy(data)
                
            # Verify sample rate match
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                waveform = resampler(waveform)

            # Squeeze to 1D for processor
            waveform_1d = waveform.mean(dim=0).squeeze().numpy()

        except Exception as e:
            self.logger.warning(
                f"Failed to load file at index {idx} ({file_path}): {e}. "
                f"Attempting to load fallback first index item."
            )
            # Prevent evaluation crash by loading index 0 recursively if not index 0 itself
            if idx != 0:
                return self.__getitem__(0)
            else:
                raise RuntimeError(f"Failed to load first baseline audio sample: {e}")

        # Process waveform using processor
        processed = self.processor(
            waveform_1d,
            sampling_rate=16000,
            return_tensors="pt"
        )

        input_values = processed.input_values.squeeze(0)
        
        # If processor returns attention mask, use it, otherwise create standard ones mask
        if "attention_mask" in processed:
            attention_mask = processed.attention_mask.squeeze(0)
        else:
            attention_mask = torch.ones(input_values.shape[0], dtype=torch.long)

        item = {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "label": torch.tensor(label_idx, dtype=torch.long),
            "file_path": str(file_path.resolve())
        }

        return item


def collate_audio_test_batches(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collates variable-length lists of test items into a padded batch tensor.

    Pads 'input_values' and 'attention_mask' using dynamic padding to match the longest sequence.

    Args:
        batch: List of dataset item dictionaries.

    Returns:
        Dict[str, Tensor]: Batched tensors including padded 'input_values', 'attention_mask', 'labels', and 'file_paths'.
    """
    labels = torch.stack([x["label"] for x in batch])
    input_values_list = [x["input_values"] for x in batch]
    attention_mask_list = [x["attention_mask"] for x in batch]
    file_paths = [x["file_path"] for x in batch]

    # Pad input values: Shape [batch_size, max_seq_length]
    padded_inputs = torch.nn.utils.rnn.pad_sequence(
        input_values_list,
        batch_first=True,
        padding_value=0.0
    )

    # Pad attention masks: Shape [batch_size, max_seq_length]
    padded_masks = torch.nn.utils.rnn.pad_sequence(
        attention_mask_list,
        batch_first=True,
        padding_value=0  # Pad with 0 (ignored token)
    )

    collated = {
        "input_values": padded_inputs,
        "attention_mask": padded_masks,
        "labels": labels,
        "file_paths": file_paths
    }

    return collated
