"""Implement convolutional AlexNet geometry used by the original Caffe weights.

Caffe AlexNet has 96/256/384/384/256 channels, grouped convolutions, and local
response normalization. Torchvision AlexNet has different widths and no LRN,
so the two weight sets cannot be interchanged. This module loads tensor-only
conversions of the official Places205 and BVLC ImageNet convolutional layers.
The input adapter reverses the loader's RGB normalization, changes to BGR in
0..255 units, and subtracts the documented channel mean. This channel-mean
approximation does not reproduce historical spatial mean-image subtraction.
"""

from __future__ import annotations
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn
from type_definitions import Options


class CaffeAlexNetFeatures(nn.Sequential):
    """Run Caffe-compatible conv1..relu5 with an optional final pooling layer."""

    def __init__(self, options: Options, scene: bool) -> None:
        """Build grouped convolutions and load the scene or head convolutional tensors."""
        layers = OrderedDict(
            [
                ("conv1", nn.Conv2d(3, 96, 11, stride=4)),
                ("relu1", nn.ReLU()),
                ("pool1", nn.MaxPool2d(3, stride=2, ceil_mode=True)),
                ("norm1", nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=1)),
                ("conv2", nn.Conv2d(96, 256, 5, padding=2, groups=2)),
                ("relu2", nn.ReLU()),
                ("pool2", nn.MaxPool2d(3, stride=2, ceil_mode=True)),
                ("norm2", nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=1)),
                ("conv3", nn.Conv2d(256, 384, 3, padding=1)),
                ("relu3", nn.ReLU()),
                ("conv4", nn.Conv2d(384, 384, 3, padding=1, groups=2)),
                ("relu4", nn.ReLU()),
                ("conv5", nn.Conv2d(384, 256, 3, padding=1, groups=2)),
                ("relu5", nn.ReLU()),
            ]
        )
        if not scene:
            layers["pool5"] = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        super().__init__(layers)
        if not getattr(options, "random_init", False):
            path = Path(options.placesmodelpath if scene else options.imagenet_model)
            self.load_state_dict(
                torch.load(path, map_location="cpu", weights_only=True), strict=True
            )

    def forward(self, normalized_rgb: torch.Tensor) -> torch.Tensor:
        """Convert normalized RGB to mean-subtracted BGR before Caffe convolutional layers."""
        rgb = (
            normalized_rgb
            * normalized_rgb.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
            + normalized_rgb.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        )
        bgr = rgb[:, [2, 1, 0]] * 255
        bgr = bgr - bgr.new_tensor([104.00648, 116.66902, 122.67946])[None, :, None, None]
        return super().forward(bgr)
