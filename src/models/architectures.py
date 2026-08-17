"""Model definitions.

Two deep architectures are trained and compared:

* ``cnn``      -- a small CNN written from scratch (the required baseline).
* ``transfer`` -- MobileNetV3-Small with ImageNet weights and a retrained head.

A scikit-learn logistic regression on flattened pixels is trained alongside them
in `train.py` as the classical baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 2


class SimpleCNN(nn.Module):
    """Four conv blocks + global pooling. ~1.2M parameters, trains on CPU."""

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3) -> None:
        super().__init__()
        channels = [3, 32, 64, 128, 256]
        blocks: list[nn.Module] = []
        for in_ch, out_ch in zip(channels[:-1], channels[1:], strict=True):
            blocks += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


def build_transfer_model(num_classes: int = NUM_CLASSES, freeze_backbone: bool = True) -> nn.Module:
    """MobileNetV3-Small with a fresh classification head."""
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = models.mobilenet_v3_small(weights=weights)
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def build_model(name: str) -> nn.Module:
    if name == "cnn":
        return SimpleCNN()
    if name == "transfer":
        return build_transfer_model()
    raise ValueError(f"Unknown model '{name}' (expected 'cnn' or 'transfer')")


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
