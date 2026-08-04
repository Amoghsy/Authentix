"""Dataset loading and feature extraction module.

This module reads preprocessing manifests, loads audio waveforms using torchaudio/soundfile,
applies the HuggingFace AutoProcessor for tokenization/feature extraction, maps classifications,
and calculates class weights to handle data imbalance during training.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor

from ai.audio.training.config import TrainingConfig


class AudioDataset(Dataset):
    """PyTorch Dataset wrapper for Wav2Vec2/WavLM audio classifications."""

    def __init__(
        self,
        config: TrainingConfig,
        split: str,
        processor: Optional[AutoFeatureExtractor] = None
    ):
        """Initializes the AudioDataset.

        Args:
            config: TrainingConfig instance.
            split: Split name ('train', 'validation', 'test').
            processor: Pre-instantiated HuggingFace AutoFeatureExtractor. If None, loaded from backbone name.
        """
        self.config = config
        self.split = split.lower()
        
        # Configure logging
        self.logger = logging.getLogger("audio_training.dataset")

        # Load Processor (Feature Extractor)
        if processor is not None:
            self.processor = processor
        else:
            self.logger.info(f"Loading feature extractor for model: {self.config.model.backbone_name}")
            self.processor = AutoFeatureExtractor.from_pretrained(self.config.model.backbone_name)

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
        """Reads dataset_manifest.csv and filters rows belonging to this split.

        Returns:
            List[Dict]: List of metadata records for the partition.
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

        # Filter by split
        filtered_df = df[df["split"].str.lower() == self.split]
        records = filtered_df.to_dict("records")
        
        self.logger.info(f"Loaded {len(records)} files for split '{self.split}'")
        return records

    def get_class_counts(self) -> Dict[int, int]:
        """Computes the distribution of classes in this dataset split.

        Returns:
            Dict[int, int]: Mapping from label index to count.
        """
        counts = {val: 0 for val in self.label_mapping.values()}
        for record in self.manifest_data:
            label_str = record["label"]
            label_idx = self.label_mapping.get(label_str.lower())
            if label_idx is not None:
                counts[label_idx] += 1
        return counts

    def calculate_class_weights(self) -> torch.Tensor:
        """Calculates inverse frequency class weights to balance cross entropy loss.

        Formula: weight = total_samples / (num_classes * class_samples)

        Returns:
            torch.Tensor: Class weights tensor.
        """
        counts = self.get_class_counts()
        total_samples = len(self.manifest_data)
        num_classes = len(self.label_mapping)
        
        weights = []
        for class_idx in sorted(counts.keys()):
            class_count = counts[class_idx]
            if class_count > 0:
                weight = total_samples / (num_classes * class_count)
            else:
                weight = 1.0
            weights.append(weight)
            
        return torch.tensor(weights, dtype=torch.float32)

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.manifest_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Loads and processes the audio file at index.

        Args:
            idx: Index in manifest.

        Returns:
            Dict[str, Tensor]: Contains:
                - 'input_values': Tensor of processed audio features.
                - 'attention_mask': Optional attention mask tensor.
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
                # Force resample if it slips past preprocessing checks
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                waveform = resampler(waveform)

            # Squeeze to 1D for processor
            waveform_1d = waveform.mean(dim=0).squeeze().numpy()

        except Exception as e:
            self.logger.warning(
                f"Failed to load file at index {idx} ({file_path}): {e}. "
                f"Attempting to load fallback first index item."
            )
            # Prevent training crash by loading index 0 recursively if not index 0 itself
            if idx != 0:
                return self.__getitem__(0)
            else:
                raise RuntimeError(f"Failed to load first baseline audio sample: {e}")

        # Process waveform using HuggingFace processor
        processed = self.processor(
            waveform_1d,
            sampling_rate=16000,
            return_tensors="pt"
        )

        item = {
            "input_values": processed.input_values.squeeze(0),
            "label": torch.tensor(label_idx, dtype=torch.long)
        }

        # Some processors include attention masks
        if "attention_mask" in processed:
            item["attention_mask"] = processed.attention_mask.squeeze(0)

        return item


def collate_audio_batches(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collates variable-length lists of items into a padded batch tensor.

    Pads 'input_values' and 'attention_mask' using dynamic padding to match the longest sequence.

    Args:
        batch: List of dataset item dictionaries.

    Returns:
        Dict[str, Tensor]: Batched tensors including padded 'input_values', 'attention_mask', and 'labels'.
    """
    labels = torch.stack([x["label"] for x in batch])
    input_values_list = [x["input_values"] for x in batch]
    
    # Pad input values: Shape [batch_size, max_seq_length]
    padded_inputs = torch.nn.utils.rnn.pad_sequence(
        input_values_list,
        batch_first=True,
        padding_value=0.0
    )

    collated = {
        "input_values": padded_inputs,
        "labels": labels
    }

    # Pad attention masks if they are present in items
    if "attention_mask" in batch[0]:
        attention_mask_list = [x["attention_mask"] for x in batch]
        padded_masks = torch.nn.utils.rnn.pad_sequence(
            attention_mask_list,
            batch_first=True,
            padding_value=0  # Pad with 0 (ignored token)
        )
        collated["attention_mask"] = padded_masks

    return collated
