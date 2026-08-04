"""
model.py

This module contains the DeepfakeModel class which wraps PyTorch's pre-trained
MobileNetV3-Small architecture. It replaces the default classifier head with a custom
fully connected binary head, and provides methods to freeze/unfreeze model blocks
for progressive fine-tuning (Phase 1, Phase 2, and Phase 3 transfer learning).
"""

import logging
from pathlib import Path
import sys
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V3_Small_Weights

# Insert project root to sys.path to allow executing the script directly
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.video.preprocessing.config import AppConfig, get_default_config, setup_logging


class DeepfakeModel(nn.Module):
    """
    MobileNetV3-Small based binary classifier for deepfake detection.
    Supports transfer learning strategies with backbone freezing/unfreezing.
    """
    def __init__(self, dropout: float = 0.2, pretrained: bool = True) -> None:
        """
        Initializes the model.
        
        Args:
            dropout (float): Dropout probability for the classifier head.
            pretrained (bool): Whether to load pre-trained ImageNet weights.
        """
        super().__init__()
        
        # Load backbone
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v3_small(weights=weights)
        
        # Original backbone features end with a 576-channel output.
        # Replacing the default classifier head.
        # MobileNetV3-Small has classifier:
        # Sequential(Linear(576, 1024), Hardswish, Dropout, Linear(1024, 1000))
        in_features = self.backbone.classifier[0].in_features # 576
        
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.Hardswish(),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(1024, 2) # Binary classification (real, fake)
        )
        
        logging.info("Deepfake model successfully initialized with MobileNetV3-Small backbone.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs forward pass.
        
        Args:
            x (torch.Tensor): Input batch tensor of shape (batch_size, 3, 224, 224).
            
        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, 2).
        """
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        """
        Freezes the entire backbone (all features and average pooling layers),
        leaving only the classifier head trainable.
        """
        for param in self.backbone.features.parameters():
            param.requires_grad = False
            
        for param in self.backbone.avgpool.parameters():
            param.requires_grad = False
            
        # Ensure classifier is trainable
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True
            
        logging.info("Model backbone frozen. Only classifier head is trainable (Phase 1).")

    def unfreeze_last_blocks(self, num_layers: int = 3) -> None:
        """
        Freezes the backbone, then unfreezes the last N layers of the features backbone
        and average pool to allow block-level fine-tuning.
        
        Args:
            num_layers (int): Number of final Conv/InvertedResidual blocks to unfreeze.
        """
        # First freeze everything
        self.freeze_backbone()
        
        # Unfreeze final pooling
        for param in self.backbone.avgpool.parameters():
            param.requires_grad = True
            
        # Unfreeze final N layers of the features sequential container
        features_len = len(self.backbone.features)
        start_idx = max(0, features_len - num_layers)
        
        for idx in range(start_idx, features_len):
            for param in self.backbone.features[idx].parameters():
                param.requires_grad = True
                
        logging.info(
            f"Unfroze last {num_layers} blocks of features and avgpool for fine-tuning (Phase 2)."
        )

    def unfreeze_all(self) -> None:
        """
        Unfreezes the entire model (backbone + classifier) for global fine-tuning.
        """
        for param in self.parameters():
            param.requires_grad = True
            
        logging.info("Unfroze all model parameters for full network fine-tuning (Phase 3).")

    def count_parameters(self) -> Tuple[int, int]:
        """
        Counts total and trainable parameters in the model.
        
        Returns:
            Tuple[int, int]: (total_params, trainable_params).
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


if __name__ == "__main__":
    # Self-test block to verify model architecture and freezing logic
    app_config = get_default_config()
    setup_logging(app_config)
    logging.info("Starting dry-run verification of model.py")
    
    try:
        # Initialize model
        model = DeepfakeModel(dropout=app_config.training.dropout, pretrained=True)
        
        # Test baseline parameter count
        total, trainable = model.count_parameters()
        logging.info(f"Initial - Total params: {total:,}, Trainable: {trainable:,}")
        
        # Test freeze backbone
        model.freeze_backbone()
        _, trainable_frozen = model.count_parameters()
        logging.info(f"Frozen Backbone - Trainable params: {trainable_frozen:,}")
        
        # Test unfreeze last 3 layers
        model.unfreeze_last_blocks(num_layers=3)
        _, trainable_last = model.count_parameters()
        logging.info(f"Fine-Tuning Last Blocks - Trainable params: {trainable_last:,}")
        
        # Test unfreeze all
        model.unfreeze_all()
        _, trainable_all = model.count_parameters()
        logging.info(f"Unfrozen All - Trainable params: {trainable_all:,}")
        
        # Test forward pass with dummy tensor
        dummy_input = torch.randn(1, 3, 224, 224)
        logits = model(dummy_input)
        logging.info(f"Forward pass completed. Input shape: {dummy_input.shape}, Output logits shape: {logits.shape}")
        
        logging.info("Model module verified successfully.")
    except Exception as e:
        logging.error(f"Model verification failed: {e}")
        raise e
