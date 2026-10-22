"""Render gaze predictions for annotated people in a GazeFollow test batch.

load_checkpoint restores a complete trained gaze state dictionary without
separately loading pretrained backbones. display_image reverses ImageNet
normalization for visualization. main loads the first test batch, predicts
heatmaps, and writes up to sixteen scene images with eye-to-prediction lines
plus separate heatmap images under --output-dir.

The input people come from the supplied annotations; no face detector is used
here. --resume and dataset files are required for actual inference. All command
execution is guarded by main, so importing the helpers does not load assets.
"""

from __future__ import annotations
from type_definitions import ImageArray, Options
from pathlib import Path

import matplotlib.pyplot as plt
import torch

import gazefollow_dataset
import command_line_options
from models.gaze_following_network import Net
from device_and_initialization import resolve_device, seed_everything


def load_checkpoint(options: Options) -> tuple[Net, torch.device]:
    """Load a complete trained gaze state dictionary in evaluation mode.

    Requires options.resume. Sets options.random_init to skip unnecessary pretrained
    backbone downloads before restoring the complete saved model.
    """
    if options.resume == "none":
        raise ValueError("Supply a trained gaze checkpoint with --resume")
    device = resolve_device(options.device)
    # A complete trained checkpoint supplies both backbones: no download required.
    options.random_init = True
    model = Net(options).to(device)
    checkpoint = torch.load(options.resume, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint))
    return model.eval(), device


def display_image(normalized_image: torch.Tensor) -> ImageArray:
    """Undo ImageNet normalization on a 3×H×W tensor and return a CPU H×W×3 RGB array."""
    channel_means = normalized_image.new_tensor([0.485, 0.456, 0.406])[:, None, None]
    channel_standard_deviations = normalized_image.new_tensor([0.229, 0.224, 0.225])[:, None, None]
    return (
        (normalized_image * channel_standard_deviations + channel_means)
        .clamp(0, 1)
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .numpy()
    )


def main() -> None:
    """Load one annotated test batch and save images with predicted gaze lines and heatmaps."""
    options = command_line_options.optionargparser().parse_args()
    seed_everything(options.seed)
    model, device = load_checkpoint(options)
    options.device = str(device)
    dataset_loaders = gazefollow_dataset.GazeFollow(options)
    scene_images, head_images, eye_position_grids, _, eye_positions, sample_indices, _, _ = next(
        iter(dataset_loaders.test_loader)
    )
    with torch.inference_mode():
        heatmaps = model.predict(
            scene_images.to(device), head_images.to(device), eye_position_grids.to(device)
        ).cpu()
    output_directory = Path(options.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    for sample_index in range(min(16, len(scene_images))):
        predicted_flat_index = heatmaps[sample_index].argmax().item()
        predicted_row, predicted_column = divmod(predicted_flat_index, heatmaps.shape[-1])
        figure, axes = plt.subplots()
        axes.imshow(display_image(scene_images[sample_index]))
        axes.plot(
            [eye_positions[sample_index, 0].item() * 227, predicted_column],
            [eye_positions[sample_index, 1].item() * 227, predicted_row],
        )
        figure.savefig(output_directory / f"{int(sample_indices[sample_index])}.jpg")
        plt.close(figure)
        plt.imsave(
            output_directory / f"{int(sample_indices[sample_index])}_heat.jpg",
            heatmaps[sample_index].numpy(),
        )


if __name__ == "__main__":
    main()
