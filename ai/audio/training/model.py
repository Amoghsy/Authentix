"""Model architecture and parameter management wrapper.

This module loads the pretrained HuggingFace audio classification model backbone,
manages model parameter states, counts parameter distributions, and controls the unfreezing
schedules required for staged transfer learning phases.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForAudioClassification

from ai.audio.training.config import TrainingConfig


class AudioClassifier(nn.Module):
    """Wrapper module for Wav2Vec2/WavLM audio deepfake classification models.

    Manages parameter freezing/unfreezing configurations for staged training.
    """

    def __init__(self, config: TrainingConfig):
        """Initializes the AudioClassifier.

        Args:
            config: TrainingConfig instance.
        """
        super().__init__()
        self.config = config
        self.logger = logging.getLogger("audio_training.model")

        self.logger.info(f"Loading pretrained audio model from: {self.config.model.backbone_name}")
        self.model = AutoModelForAudioClassification.from_pretrained(
            self.config.model.backbone_name,
            num_labels=self.config.model.num_labels,
            ignore_mismatched_sizes=True  # Resets classification head if labels count changed
        )

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Performs a forward pass through the model.

        Args:
            input_values: Batched audio input features tensor [batch_size, sequence_length].
            attention_mask: Optional attention mask tensor.
            labels: Ground truth classification labels tensor [batch_size].

        Returns:
            Dict[str, Tensor]: Contains:
                - 'loss': Classification loss tensor (if labels provided).
                - 'logits': Predicted class scores tensor [batch_size, num_labels].
        """
        outputs = self.model(
            input_values=input_values,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        output_dict = {"logits": outputs.logits}
        if outputs.loss is not None:
            output_dict["loss"] = outputs.loss
            
        return output_dict

    def freeze_backbone(self) -> None:
        """Freezes the entire backbone, keeping only classification heads trainable.

        Typically used during Phase 1 training to prevent early gradient noise from
        destabilizing pretrained representations.
        """
        self.logger.info("Freezing model backbone. Only classification heads remain trainable.")
        
        # 1. Freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 2. Unfreeze classifier components
        unfrozen_count = 0
        for name, param in self.model.named_parameters():
            if "classifier" in name or "projector" in name:
                param.requires_grad = True
                unfrozen_count += 1
                
        self.logger.info(f"Phase 1 freeze complete. Unfroze {unfrozen_count} classifier parameters.")

    def unfreeze_last_layers(self, num_layers: Optional[int] = None) -> None:
        """Unfreezes the classifier and the last N transformer encoder blocks of the backbone.

        Args:
            num_layers: Number of encoder layers to unfreeze. Defaults to config value.
        """
        if num_layers is None:
            num_layers = self.config.model.unfreeze_layers_count

        self.logger.info(f"Unfreezing classification heads and the last {num_layers} encoder layers.")
        
        # 1. Start by freezing the entire model
        for param in self.model.parameters():
            param.requires_grad = False

        # 2. Identify the transformer encoder layers ModuleList
        encoder_layers = None
        # Standard Wav2Vec2/WavLM backbone mappings
        if hasattr(self.model, "wav2vec2") and hasattr(self.model.wav2vec2, "encoder"):
            encoder_layers = self.model.wav2vec2.encoder.layers
        elif hasattr(self.model, "wavlm") and hasattr(self.model.wavlm, "encoder"):
            encoder_layers = self.model.wavlm.encoder.layers
        elif hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layers"):
            encoder_layers = self.model.encoder.layers

        # 3. Unfreeze the last N layers
        if encoder_layers is not None and len(encoder_layers) > 0:
            total_layers = len(encoder_layers)
            unfreeze_start_idx = max(0, total_layers - num_layers)
            
            for idx in range(unfreeze_start_idx, total_layers):
                for param in encoder_layers[idx].parameters():
                    param.requires_grad = True
            self.logger.info(f"Unfroze encoder blocks index range: [{unfreeze_start_idx} to {total_layers - 1}].")
        else:
            self.logger.warning("Could not automatically locate transformer encoder layers. Fine-tuning full model fallback.")
            self.unfreeze_all()
            return

        # 4. Ensure classification projection heads are also unfrozen
        for name, param in self.model.named_parameters():
            if "classifier" in name or "projector" in name:
                param.requires_grad = True

    def unfreeze_all(self) -> None:
        """Unfreezes all parameters across the entire model backbone and head."""
        self.logger.info("Unfreezing all model parameters for full fine-tuning.")
        for param in self.model.parameters():
            param.requires_grad = True

    def count_parameters(self) -> Tuple[int, int, int]:
        """Calculates total, trainable, and frozen parameter counts.

        Returns:
            Tuple[int, int, int]: (total_params, trainable_params, frozen_params).
        """
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen = total - trainable
        return total, trainable, frozen

    def print_model_summary(self) -> None:
        """Logs a formatted summary of parameter statistics and structural blocks."""
        total, trainable, frozen = self.count_parameters()
        summary = (
            f"\n=== AudioClassifier Summary ===\n"
            f"Pretrained Backbone: {self.config.model.backbone_name}\n"
            f"Total Parameters:    {total:,}\n"
            f"Trainable Parameters:{trainable:,}\n"
            f"Frozen Parameters:   {frozen:,}\n"
            f"Requires Grad Ratio: {trainable / total * 100:.2f}%\n"
            f"================================="
        )
        self.logger.info(summary)
        print(summary)
