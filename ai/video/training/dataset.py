"""
dataset.py

This module contains the custom PyTorch Dataset implementation for loading preprocessed
deepfake detection face crops. It reads the dataset_manifest.csv to extract image paths,
labels, and splits, loads the images in RGB format using Pillow, and supports image
transformations (with robust fallback for Albumentations/Torchvision).
"""

import csv
import logging
from pathlib import Path
import sys
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

# Insert project root to sys.path to allow executing the script directly
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.video.preprocessing.config import AppConfig, get_default_config, setup_logging


class DeepfakeDataset(Dataset):
    """
    Custom Dataset class for loading face crops for Authentix deepfake detection training.
    Supports manifest-based loading and fallback directory-based crawls.
    """
    def __init__(
        self, 
        config: AppConfig, 
        split: str, 
        transform: Optional[Any] = None
    ) -> None:
        """
        Initializes the dataset.
        
        Args:
            config (AppConfig): Root application configuration.
            split (str): One of 'train', 'validation', or 'test'.
            transform (Optional[Any]): Albumentations or Torchvision transform pipeline.
        """
        self.config = config
        self.split = split.lower()
        self.transform = transform
        
        if self.split not in {"train", "validation", "test"}:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'validation', or 'test'.")
            
        self.dataset_dir = config.paths.dataset_dir
        self.manifest_path = self.dataset_dir / "dataset_manifest.csv"
        
        self.samples: List[Tuple[Path, int]] = []
        self.label_mapping = {"real": 0, "fake": 1}
        
        # Load samples
        if self.manifest_path.exists():
            self._load_from_manifest()
        else:
            logging.warning(
                f"Manifest file not found at {self.manifest_path}. "
                f"Falling back to recursive directory crawl."
            )
            self._load_from_crawl()
            
        logging.info(
            f"Successfully loaded {len(self.samples)} samples for '{self.split}' split "
            f"({self.manifest_path.name if self.manifest_path.exists() else 'directory crawl'})."
        )

    def _load_from_manifest(self) -> None:
        """
        Loads dataset records from the manifest CSV file.
        """
        with open(self.manifest_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Filter by split
                if row["split"].strip().lower() != self.split:
                    continue
                    
                rel_path = row["file_path"].strip()
                abs_path = self.dataset_dir / rel_path
                
                # Check file existence to prevent RuntimeErrors during training
                if not abs_path.exists():
                    logging.warning(f"File listed in manifest does not exist: {abs_path}")
                    continue
                    
                label_str = row["label"].strip().lower()
                label_idx = self.label_mapping.get(label_str)
                if label_idx is None:
                    logging.error(f"Unknown label in manifest: '{label_str}' for file {abs_path}")
                    continue
                    
                self.samples.append((abs_path, label_idx))

    def _load_from_crawl(self) -> None:
        """
        Crawls the corresponding split subdirectory directly if the manifest is missing.
        """
        split_dir = self.dataset_dir / self.split
        if not split_dir.exists():
            logging.error(f"Split directory does not exist: {split_dir}")
            return
            
        # Discover class folders (real / fake)
        for class_name, label_idx in self.label_mapping.items():
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
                
            for path in class_dir.rglob("*.jpg"):
                self.samples.append((path, label_idx))

    def __len__(self) -> int:
        """
        Returns the number of samples in this split.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Loads and returns a sample.
        
        Args:
            idx (int): Sample index.
            
        Returns:
            Tuple[torch.Tensor, int, str]: (image_tensor, label, absolute_image_path)
        """
        img_path, label = self.samples[idx]
        
        # Load image via PIL
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logging.error(f"Failed to load image at {img_path}: {e}")
            # Return a dummy tensor if file load crashes mid-training
            dummy_tensor = torch.zeros((3, self.config.pipeline.image_size[0], self.config.pipeline.image_size[1]))
            return dummy_tensor, label, str(img_path.resolve())
            
        # Apply transformations (Pillow -> numpy array if using Albumentations)
        if self.transform is not None:
            try:
                # Check if it is an Albumentations transform or torchvision transform
                # Albumentations expects image as numpy array, returning a dict
                if hasattr(self.transform, "processors") or callable(getattr(self.transform, "index_of", None)):
                    image_np = np.array(image)
                    augmented = self.transform(image=image_np)
                    transformed_image = augmented["image"]
                else:
                    # Torchvision transform
                    transformed_image = self.transform(image)
            except Exception as e:
                logging.error(f"Transform failed on image {img_path}: {e}")
                transformed_image = image
        else:
            transformed_image = image
            
        # Convert output to PyTorch tensor if it is not already a tensor
        if isinstance(transformed_image, torch.Tensor):
            image_tensor = transformed_image
        else:
            # Convert NumPy array or PIL Image to normalized float tensor
            image_tensor = TF.to_tensor(transformed_image)
            
        return image_tensor, label, str(img_path.resolve())


if __name__ == "__main__":
    # Self-test code
    app_config = get_default_config()
    setup_logging(app_config)
    logging.info("Starting dry-run verification of dataset.py")
    
    try:
        # Load train dataset
        train_ds = DeepfakeDataset(app_config, split="train")
        if len(train_ds) > 0:
            img_t, lbl, path = train_ds[0]
            logging.info(f"Verified sample 0! Shape: {img_t.shape}, Label: {lbl}, Path: {path}")
        else:
            logging.warning("No samples found. Please run the preprocessing pipeline first.")
    except Exception as e:
        logging.error(f"Dataset verification failed: {e}")
        raise e
