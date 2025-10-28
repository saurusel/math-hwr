"""
M2: Segmentation + MLP Classifier
Classical OCR approach with connected components segmentation and MLP classification.
"""
import torch
import torch.nn as nn
from typing import List


class SegmentationMLP(nn.Module):
    """
    Simple MLP (Multi-Layer Perceptron) for character classification.

    Architecture:
    - Input: flattened character image (32x32 = 1024 pixels)
    - Hidden layers: 1024 -> 512 -> 256 -> 128
    - Output: num_classes (character probabilities)

    Used after segmentation to classify each extracted character.
    """

    def __init__(self, num_classes: int, input_size: int = 1024, dropout: float = 0.3):
        """
        Args:
            num_classes: Number of character classes (from TOKEN_LIST)
            input_size: Flattened image size (default 32x32 = 1024)
            dropout: Dropout probability for regularization
        """
        super().__init__()

        self.input_size = input_size
        self.num_classes = num_classes

        # MLP layers
        self.network = nn.Sequential(
            # Layer 1: 1024 -> 512
            nn.Linear(input_size, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),

            # Layer 2: 512 -> 256
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),

            # Layer 3: 256 -> 128
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),

            # Output layer: 128 -> num_classes
            nn.Linear(128, num_classes)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 1, H, W) or (batch_size, H, W)
               or already flattened (batch_size, input_size)

        Returns:
            logits: Output tensor of shape (batch_size, num_classes)
        """
        # Flatten if needed
        if x.dim() == 4:  # (B, 1, H, W)
            x = x.view(x.size(0), -1)
        elif x.dim() == 3:  # (B, H, W)
            x = x.view(x.size(0), -1)

        # Pass through network
        logits = self.network(x)
        return logits


class SegmentationOCR(nn.Module):
    """
    Full M2 model: Segmentation + MLP Classification.

    This is a wrapper that handles the full pipeline:
    1. Segment image into characters
    2. Classify each character
    3. Combine into sequence

    Note: Segmentation happens in preprocessing, this model only does classification.
    """

    def __init__(self, num_classes: int, input_size: int = 1024, dropout: float = 0.3):
        super().__init__()
        self.classifier = SegmentationMLP(num_classes, input_size, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for batch of segmented characters.

        Args:
            x: Batch of character images (batch_size, 1, H, W) or (batch_size, H*W)

        Returns:
            logits: (batch_size, num_classes)
        """
        return self.classifier(x)

    def predict_sequence(self, segments: List[torch.Tensor]) -> List[int]:
        """
        Predict sequence of character IDs from list of segmented character tensors.

        Args:
            segments: List of character tensors, each of shape (1, H, W)

        Returns:
            List of predicted character IDs
        """
        if not segments:
            return []

        # Stack into batch
        batch = torch.stack([s.squeeze(0) if s.dim() == 3 else s for s in segments])

        # Classify
        with torch.no_grad():
            logits = self.forward(batch)  # (num_segments, num_classes)
            predictions = torch.argmax(logits, dim=1)  # (num_segments,)

        return predictions.cpu().tolist()
