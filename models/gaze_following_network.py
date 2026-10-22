"""Define the two-branch AlexNet gaze model and historical shifted-grid decoder.

AlexSal estimates a 13x13 scene saliency map from Places-initialized convolutional
features. AlexGaze combines ImageNet-initialized head features with a quantized
eye location to produce a 13x13 gaze map. Net multiplies the two maps and applies
five independent 25-class heads. forward returns grid log-probabilities;
predict combines head probabilities into an interpolated 227x227 heatmap.

Registered learned-module names are retained for existing state_dict checkpoint
keys, while readable properties expose the scene and head branches. This model
retains the original reproduction's layer sizes, extra gaze convolution, and
known decoder/target shift inconsistencies; renaming files does not fix those
scientific reproduction differences.
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
        self.features = nn.Sequential(
            *list(
                models.alexnet(
                    weights=(
                        None
                        if getattr(options, "random_init", False)
                        else models.AlexNet_Weights.IMAGENET1K_V1
                    )
                ).features.children()
            )
        )
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.fc1 = nn.Linear(9216, 500)
        self.fc2 = nn.Linear(669, 400)
        self.fc3 = nn.Linear(400, 200)
        self.fc4 = nn.Linear(200, 169)

        self.finalconv = nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

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
            Gaze maps after the historical post-sigmoid convolution. They are
            not bounded probabilities because that convolution follows sigmoid.
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
        interpolated heatmap values, not normalized image-wide probabilities. Known
        training/decoding shift inconsistencies are preserved pending separate fixes.
        """

        saliency_maps = self.saliency_branch(scene_images)
        gaze_maps = self.gaze_branch(head_images, eye_position_grids)
        fused_features = saliency_maps * gaze_maps
        fused_features = fused_features.view(-1, 169)

        heatmap_sums = fused_features.new_zeros((fused_features.size(0), 15, 15))
        heatmap_counts = fused_features.new_zeros((fused_features.size(0), 15, 15))

        center_grid_probabilities = self.softmax(self.fc_0_0(fused_features)).view(-1, 5, 5)
        positive_x_grid_probabilities = self.softmax(self.fc_1_0(fused_features)).view(-1, 5, 5)
        negative_x_grid_probabilities = self.softmax(self.fc_m1_0(fused_features)).view(-1, 5, 5)
        negative_y_grid_probabilities = self.softmax(self.fc_0_m1(fused_features)).view(-1, 5, 5)
        positive_y_grid_probabilities = self.softmax(self.fc_0_1(fused_features)).view(-1, 5, 5)

        # Preserve historical head ordering and offset semantics pending the grid fix.
        decoder_grid_probabilities = []
        decoder_grid_probabilities.extend(
            [
                negative_y_grid_probabilities,
                positive_y_grid_probabilities,
                negative_x_grid_probabilities,
                positive_x_grid_probabilities,
                center_grid_probabilities,
            ]
        )

        decoder_row_offsets = [0, 1, -1, 0, 0]
        decoder_column_offsets = [0, 0, 0, -1, 1]

        for grid_index in range(5):
            row_offset, column_offset = (
                decoder_row_offsets[grid_index],
                decoder_column_offsets[grid_index],
            )
            grid_probabilities = decoder_grid_probabilities[grid_index]
            for grid_row in range(5):
                for grid_column in range(5):

                    region_top = 3 * grid_row - row_offset
                    region_top = max(region_top, 0)
                    if grid_row == 0:
                        region_top = 0

                    region_left = 3 * grid_column - column_offset
                    region_left = max(region_left, 0)
                    if grid_column == 0:
                        region_left = 0

                    region_bottom = 3 * grid_row + 2 - row_offset
                    region_bottom = min(14, region_bottom)
                    if grid_row == 4:
                        region_bottom = 14

                    region_right = 3 * grid_column + 2 - column_offset
                    region_right = min(14, region_right)
                    if grid_column == 4:
                        region_right = 14

                    cell_probability = grid_probabilities[:, grid_row, grid_column].contiguous()
                    cell_probability = cell_probability.view(fused_features.size(0), 1, 1)

                    heatmap_sums[
                        :, region_top : region_bottom + 1, region_left : region_right + 1
                    ] = (
                        heatmap_sums[
                            :, region_top : region_bottom + 1, region_left : region_right + 1
                        ]
                        + cell_probability
                    )
                    heatmap_counts[
                        :, region_top : region_bottom + 1, region_left : region_right + 1
                    ] = (
                        heatmap_counts[
                            :, region_top : region_bottom + 1, region_left : region_right + 1
                        ]
                        + 1
                    )

        averaged_heatmaps = heatmap_sums.div(heatmap_counts)

        averaged_heatmaps = averaged_heatmaps.unsqueeze(1)

        averaged_heatmaps = F.interpolate(
            input=averaged_heatmaps, size=(227, 227), mode="bicubic", align_corners=False
        )

        averaged_heatmaps = averaged_heatmaps.squeeze(1)

        return averaged_heatmaps
