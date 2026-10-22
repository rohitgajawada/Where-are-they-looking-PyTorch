"""Define the S3FD face-detection architecture and learned feature normalization.

L2Norm normalizes channel vectors and applies learned per-channel scale values.
s3fd builds a VGG-style convolutional backbone and six paired classification/
box-regression heads. forward returns twelve alternating feature maps: face/
background logits and anchor-relative box offsets at each detector scale. The
first scale uses max-out background classification.

The network expects BGR float inputs with the detector mean already subtracted.
Image preparation, probability thresholds, decoding, and NMS are handled by
face_detection.py. Existing learned-module names are preserved for S3FD state
dictionaries; this file does not itself load a detector checkpoint.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class L2Norm(nn.Module):
    """Normalize feature vectors across channels with learnable per-channel scales."""

    def __init__(self, n_channels: int, scale: float = 1.0) -> None:
        """Create one learned normalization scale per feature channel."""
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.scale = scale
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        nn.init.constant_(self.weight, self.scale)

    def forward(self, feature_maps: torch.Tensor) -> torch.Tensor:
        """Normalize across channels and rescale a B×C×H×W feature tensor without changing shape."""
        channel_norms = feature_maps.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        feature_maps = feature_maps / channel_norms * self.weight.view(1, -1, 1, 1)
        return feature_maps


class s3fd(nn.Module):
    """Predict multi-scale face scores and anchor-relative box offsets."""

    def __init__(self) -> None:
        """Construct the S3FD backbone and six classification/regression heads."""
        super(s3fd, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.fc6 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=3)
        self.fc7 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0)

        self.conv6_1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.conv6_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.conv7_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.conv7_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.conv3_3_norm = L2Norm(256, scale=10)
        self.conv4_3_norm = L2Norm(512, scale=8)
        self.conv5_3_norm = L2Norm(512, scale=5)

        self.conv3_3_norm_mbox_conf = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)
        self.conv3_3_norm_mbox_loc = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)
        self.conv4_3_norm_mbox_conf = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
        self.conv4_3_norm_mbox_loc = nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)
        self.conv5_3_norm_mbox_conf = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
        self.conv5_3_norm_mbox_loc = nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)

        self.fc7_mbox_conf = nn.Conv2d(1024, 2, kernel_size=3, stride=1, padding=1)
        self.fc7_mbox_loc = nn.Conv2d(1024, 4, kernel_size=3, stride=1, padding=1)
        self.conv6_2_mbox_conf = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
        self.conv6_2_mbox_loc = nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)
        self.conv7_2_mbox_conf = nn.Conv2d(256, 2, kernel_size=3, stride=1, padding=1)
        self.conv7_2_mbox_loc = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)

    def forward(self, detector_images: torch.Tensor) -> list[torch.Tensor]:
        """Return six alternating class-logit and box-offset feature maps.

        Args:
            detector_images: B×3×H×W float tensor, with BGR detector mean already subtracted.

        Returns:
            Twelve tensors: class logits (B,2,H_i,W_i) and offsets (B,4,H_i,W_i)
            for each of six scales. First-scale background channels are max-pooled.
        """
        feature_maps = F.relu(self.conv1_1(detector_images))
        feature_maps = F.relu(self.conv1_2(feature_maps))
        feature_maps = F.max_pool2d(feature_maps, 2, 2)

        feature_maps = F.relu(self.conv2_1(feature_maps))
        feature_maps = F.relu(self.conv2_2(feature_maps))
        feature_maps = F.max_pool2d(feature_maps, 2, 2)

        feature_maps = F.relu(self.conv3_1(feature_maps))
        feature_maps = F.relu(self.conv3_2(feature_maps))
        feature_maps = F.relu(self.conv3_3(feature_maps))
        stage3_features = feature_maps
        feature_maps = F.max_pool2d(feature_maps, 2, 2)

        feature_maps = F.relu(self.conv4_1(feature_maps))
        feature_maps = F.relu(self.conv4_2(feature_maps))
        feature_maps = F.relu(self.conv4_3(feature_maps))
        stage4_features = feature_maps
        feature_maps = F.max_pool2d(feature_maps, 2, 2)

        feature_maps = F.relu(self.conv5_1(feature_maps))
        feature_maps = F.relu(self.conv5_2(feature_maps))
        feature_maps = F.relu(self.conv5_3(feature_maps))
        stage5_features = feature_maps
        feature_maps = F.max_pool2d(feature_maps, 2, 2)

        feature_maps = F.relu(self.fc6(feature_maps))
        feature_maps = F.relu(self.fc7(feature_maps))
        stage_fc7_features = feature_maps
        feature_maps = F.relu(self.conv6_1(feature_maps))
        feature_maps = F.relu(self.conv6_2(feature_maps))
        stage6_features = feature_maps
        feature_maps = F.relu(self.conv7_1(feature_maps))
        feature_maps = F.relu(self.conv7_2(feature_maps))
        stage7_features = feature_maps

        stage3_features = self.conv3_3_norm(stage3_features)
        stage4_features = self.conv4_3_norm(stage4_features)
        stage5_features = self.conv5_3_norm(stage5_features)

        scale1_class_logits = self.conv3_3_norm_mbox_conf(stage3_features)
        scale1_box_offsets = self.conv3_3_norm_mbox_loc(stage3_features)
        scale2_class_logits = self.conv4_3_norm_mbox_conf(stage4_features)
        scale2_box_offsets = self.conv4_3_norm_mbox_loc(stage4_features)
        scale3_class_logits = self.conv5_3_norm_mbox_conf(stage5_features)
        scale3_box_offsets = self.conv5_3_norm_mbox_loc(stage5_features)
        scale4_class_logits = self.fc7_mbox_conf(stage_fc7_features)
        scale4_box_offsets = self.fc7_mbox_loc(stage_fc7_features)
        scale5_class_logits = self.conv6_2_mbox_conf(stage6_features)
        scale5_box_offsets = self.conv6_2_mbox_loc(stage6_features)
        scale6_class_logits = self.conv7_2_mbox_conf(stage7_features)
        scale6_box_offsets = self.conv7_2_mbox_loc(stage7_features)

        # max-out background label
        background_and_face_logits = torch.chunk(scale1_class_logits, 4, 1)
        background_max_logits = torch.max(
            torch.max(background_and_face_logits[0], background_and_face_logits[1]),
            background_and_face_logits[2],
        )
        scale1_class_logits = torch.cat(
            [background_max_logits, background_and_face_logits[3]], dim=1
        )

        return [
            scale1_class_logits,
            scale1_box_offsets,
            scale2_class_logits,
            scale2_box_offsets,
            scale3_class_logits,
            scale3_box_offsets,
            scale4_class_logits,
            scale4_box_offsets,
            scale5_class_logits,
            scale5_box_offsets,
            scale6_class_logits,
            scale6_box_offsets,
        ]
