"""Measure center and training-derived fixation-prior baselines on development data.

Both baselines use only training labels to define predictions and are evaluated
on the image-disjoint development partition. The prior is an unconditional
64x64 fixation histogram, unlike the paper's head-conditioned fixed-bias model.
The center heatmap is a fixed Gaussian. Distance, minimum distance, angular
error, and explicit 64x64 fixation AUC use the same evaluator as neural runs.
No RGB image loading or official-test evaluation is needed for these baselines.
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
import command_line_options
import gazefollow_dataset
import gaze_metrics


def main() -> None:
    """Build label-only baselines and evaluate their held-out development predictions."""
    options = command_line_options.optionargparser().parse_args()
    torch.set_num_threads(4)
    loaders = gazefollow_dataset.GazeFollow(options)
    dataset = loaders.train_dataset
    training_points = np.stack(
        [dataset.gaze_points[index][:2] for index in loaders.training_indices]
    )
    histogram, _, _ = np.histogram2d(
        training_points[:, 1], training_points[:, 0], bins=64, range=((0, 1), (0, 1))
    )
    prior = torch.tensor(gaussian_filter(histogram, 1), dtype=torch.float32)[None, None]
    prior = torch.nn.functional.interpolate(
        prior, size=(227, 227), mode="bilinear", align_corners=False
    )[0, 0]
    axis = torch.arange(227) / 227
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    center = torch.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / (0.1**2))
    annotations = torch.tensor(
        np.stack([dataset.gaze_points[index] for index in loaders.validation_indices]),
        dtype=torch.float32,
    )
    eyes = torch.tensor(
        np.stack([dataset.eye_positions[index] for index in loaders.validation_indices]),
        dtype=torch.float32,
    )
    results = {}
    for name, heatmap in [("center", center), ("unconditional_training_prior", prior)]:
        count = len(annotations)
        peaks = heatmap.argmax().expand(count)
        results[name] = {
            "examples": count,
            "mean_distance": gaze_metrics.euclid_dist(peaks, annotations, count),
            "minimum_distance": gaze_metrics.euclid_mindist(peaks, annotations, count),
            "angular_error_degrees": gaze_metrics.angular_error(peaks, annotations, eyes),
            "auc_64": gaze_metrics.fixation_auc(heatmap.expand(count, -1, -1), annotations),
        }
    Path("runs/development_baselines.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
