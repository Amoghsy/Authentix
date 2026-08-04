"""
augmentations.py

This module defines image augmentation pipelines using Albumentations for both training
and validation. The training augmentations include horizontal flips, color jitter, JPEG
compression simulation, and light blurs to build model robustness, while validation/testing
is kept clean to preserve actual artifacts.
"""

import logging
from pathlib import Path
import sys
from typing import Tuple, Any

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

# Insert project root to sys.path to allow executing the script directly
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.video.preprocessing.config import AppConfig, get_default_config, setup_logging


def get_train_transforms(image_size: Tuple[int, int]) -> A.Compose:
    """
    Constructs the Albumentations transformation pipeline for training.
    
    Args:
        image_size (Tuple[int, int]): Target image dimension (height, width).
        
    Returns:
        A.Compose: The training augmentation pipeline.
    """
    height, width = image_size
    return A.Compose([
        # Resize to target dimension
        A.Resize(height=height, width=width),
        
        # Horizontal flips (preserves face topology but adds variability)
        A.HorizontalFlip(p=0.5),
        
        # Slight brightness and contrast adjustments
        A.RandomBrightnessContrast(
            brightness_limit=0.1, 
            contrast_limit=0.1, 
            p=0.5
        ),
        
        # Simulates JPEG compression artifacts (common in shared/manipulated internet videos)
        A.ImageCompression(
            quality_range=(60, 100), 
            p=0.3
        ),
        
        # Light Gaussian Blur (simulates lower resolution or camera out-of-focus)
        A.GaussianBlur(
            blur_limit=(3, 5), 
            p=0.3
        ),
        
        # Light sensor noise simulation
        A.GaussNoise(
            std_range=(0.01, 0.03), 
            p=0.3
        ),
        
        # Normalize to standard ImageNet mean and standard deviation
        A.Normalize(
            mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0
        ),
        
        # Convert to PyTorch Tensor format
        ToTensorV2()
    ])


def get_val_transforms(image_size: Tuple[int, int]) -> A.Compose:
    """
    Constructs the Albumentations transformation pipeline for validation and testing.
    Uses strictly resize and normalization.
    
    Args:
        image_size (Tuple[int, int]): Target image dimension (height, width).
        
    Returns:
        A.Compose: The validation pipeline.
    """
    height, width = image_size
    return A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize(
            mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])


if __name__ == "__main__":
    # Self-test block to verify augmentations compile and run
    app_config = get_default_config()
    setup_logging(app_config)
    logging.info("Starting dry-run verification of augmentations.py")
    
    try:
        # Construct pipeline
        train_pipe = get_train_transforms(app_config.pipeline.image_size)
        val_pipe = get_val_transforms(app_config.pipeline.image_size)
        logging.info("Augmentation pipelines compiled successfully.")
        
        # Create a dummy image to test pipelines
        dummy_img = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
        
        # Run through train pipeline
        train_out = train_pipe(image=dummy_img)
        train_tensor = train_out["image"]
        logging.info(f"Train pipeline output tensor shape: {train_tensor.shape}")
        
        # Run through val pipeline
        val_out = val_pipe(image=dummy_img)
        val_tensor = val_out["image"]
        logging.info(f"Val pipeline output tensor shape: {val_tensor.shape}")
        
        logging.info("Augmentations module verified successfully.")
    except Exception as e:
        logging.error(f"Augmentations verification failed: {e}")
        raise e
