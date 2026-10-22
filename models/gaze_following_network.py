"""Define the two-branch AlexNet gaze model with consistent shifted-grid decoding.

AlexSal estimates a 13x13 scene saliency map from Places-initialized convolutional
features. AlexGaze combines ImageNet-initialized head features with a quantized
eye location to produce a 13x13 gaze map. Net multiplies the two maps and applies
five independent 25-class heads. forward returns grid log-probabilities;
predict combines head probabilities into an interpolated 227x227 heatmap.

Registered learned-module names retain historical parameter prefixes. The
--architecture option selects the paper's FC widths and final sigmoid or the
historical 500-unit/post-sigmoid convolution layout. --backbone selects Caffe
or torchvision features. This does not assert complete 2015 numerical parity:
head-crop and preprocessing details remain documented approximations.
"""

from __future__ import annotations
from type_definitions import Options
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from device_and_initialization import load_places_features


class AlexSal(nn.Module):
    """Estimate scene saliency using a Places-initialized convolutional backbone."""

    def __init__(self, options: Options) -> None:
        """Initialize the Places backbone and 1×1 saliency projection."""
        super(AlexSal, self).__init__()

        self.features = load_places_features(options)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.conv6 = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, scene_images: torch.Tensor) -> torch.Tensor:
        """Map normalized B×3×227×227 scene images to B×13×13 saliency maps."""
        scene_features = self.relu(self.features(scene_images))
        saliency_maps = self.relu(self.conv6(scene_features))
        return saliency_maps.squeeze(1)


class AlexGaze(nn.Module):
    """Condition a gaze map on a head crop and the person's scene location."""

    def __init__(self, options: Options) -> None:
        """Initialize the head backbone, location-conditioned fully connected layers, and convolution."""
        super(AlexGaze, self).__init__()
        if getattr(options, "backbone", "torchvision") == "caffe":
            from models.caffe_alexnet_backbone import CaffeAlexNetFeatures

            self.features = CaffeAlexNetFeatures(options, scene=False)
        else:
            backbone = models.alexnet(weights=None)
            if not getattr(options, "random_init", False):
                checkpoint = torch.load(
                    options.imagenet_model, map_location="cpu", weights_only=True
                )
                backbone.load_state_dict(checkpoint, strict=True)
            self.features = backbone.features
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.fc1 = nn.Linear(
            9216, 500 if getattr(options, "architecture", "paper") == "legacy" else 100
        )
        self.fc2 = nn.Linear(self.fc1.out_features + 169, 400)
        self.fc3 = nn.Linear(400, 200)
        self.fc4 = nn.Linear(200, 169)

        self.finalconv = (
            nn.Conv2d(1, 1, kernel_size=3, padding=1)
            if getattr(options, "architecture", "paper") == "legacy"
            else nn.Identity()
        )

    def forward(
        self,
        head_images: torch.Tensor,
        eye_position_grids: torch.Tensor,
    ) -> torch.Tensor:
        """Map head images and eye-location grids to B×13×13 gaze maps.

        Args:
            head_images: Normalized head crops with shape B×3×227×227.
            eye_position_grids: One-hot scene locations with shape B×13×13.

        Returns:
            Gaze masks bounded by sigmoid in paper mode; legacy mode applies
            an additional convolution and can produce unbounded values.
        """
        head_features = self.relu(self.features(head_images))
        flattened_head_features = head_features.view(-1, 9216)
        head_embeddings = self.relu(self.fc1(flattened_head_features))

        flattened_eye_grids = eye_position_grids.view(-1, 169)
        scaled_eye_grids = flattened_eye_grids * 24

        gaze_features = torch.cat((head_embeddings, scaled_eye_grids), dim=1)
        gaze_features = self.relu(self.fc2(gaze_features))
        gaze_features = self.relu(self.fc3(gaze_features))
        gaze_masks = self.sigmoid(self.fc4(gaze_features))

        gaze_masks = gaze_masks.view(-1, 1, 13, 13)
        gaze_maps = self.finalconv(gaze_masks)
        return gaze_maps.squeeze(1)


class Net(nn.Module):
    """Combine scene saliency and head-conditioned gaze through five grid classifiers."""

    def __init__(self, options: Options) -> None:
        """Initialize scene/head branches and five independent 25-class grid heads."""
        super(Net, self).__init__()
        # Learned module names are retained for historical state_dict compatibility.
        self.salpath = AlexSal(options)
        self.gazepath = AlexGaze(options)
        self.options = options

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

        self.fc_0_0 = nn.Linear(169, 25)
        self.fc_0_m1 = nn.Linear(169, 25)
        self.fc_0_1 = nn.Linear(169, 25)
        self.fc_m1_0 = nn.Linear(169, 25)
        self.fc_1_0 = nn.Linear(169, 25)

    @property
    def saliency_branch(self) -> AlexSal:
        """Expose the scene branch without changing its saved salpath.* parameter keys."""
        return self.salpath

    @property
    def gaze_branch(self) -> AlexGaze:
        """Expose the head/location branch without changing saved gazepath.* keys."""
        return self.gazepath

    def forward(
        self,
        scene_images: torch.Tensor,
        head_images: torch.Tensor,
        eye_position_grids: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Predict five shifted-grid log-probability tensors, each with shape B×25.

        Args:
            scene_images: Normalized full scenes, B×3×227×227.
            head_images: Normalized head crops, B×3×227×227.
            eye_position_grids: One-hot eye-location grids, B×13×13.

        Returns:
            Grid outputs ordered center, positive x, negative x, negative y, positive y.
        """

        saliency_maps = self.saliency_branch(scene_images)
        gaze_maps = self.gaze_branch(head_images, eye_position_grids)
        fused_features = saliency_maps * gaze_maps
        fused_features = fused_features.view(-1, 169)

        center_grid_log_probabilities = self.log_softmax(self.fc_0_0(fused_features))
        positive_x_grid_log_probabilities = self.log_softmax(self.fc_1_0(fused_features))
        negative_x_grid_log_probabilities = self.log_softmax(self.fc_m1_0(fused_features))
        negative_y_grid_log_probabilities = self.log_softmax(self.fc_0_m1(fused_features))
        positive_y_grid_log_probabilities = self.log_softmax(self.fc_0_1(fused_features))

        return [
            center_grid_log_probabilities,
            positive_x_grid_log_probabilities,
            negative_x_grid_log_probabilities,
            negative_y_grid_log_probabilities,
            positive_y_grid_log_probabilities,
        ]

    def predict(
        self,
        scene_images: torch.Tensor,
        head_images: torch.Tensor,
        eye_position_grids: torch.Tensor,
    ) -> torch.Tensor:
        """Assemble historical shifted-grid predictions into B×227×227 heatmaps.

        Inputs have the same shapes and meaning as forward. The returned scores are
        interpolated heatmap values, not normalized image-wide probabilities. The shared
        grid definition keeps training labels and heatmap decoding consistent.
        """

        from shifted_grids import decode_grids

        return decode_grids(self(scene_images, head_images, eye_position_grids))
