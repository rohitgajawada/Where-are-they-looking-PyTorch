"""Predict and visualize gaze for automatically detected faces in an image folder.

main restores the trained gaze network and an S3FD face detector, reads supported
RGB image files from --input-dir, and extracts face crops and approximate eye
locations from detected box centers. It normalizes the scene/head inputs,
predicts one gaze heatmap per detected face, and saves a scene visualization
with eye-to-prediction lines for every detected person.

This is a detector-driven demo rather than the annotated-head paper benchmark.
It requires separate trained gaze and S3FD state dictionaries and skips images
with no retained faces. It has no model-loading or inference side effects on
import; use --help to inspect the shared CLI options.
"""

from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

import command_line_options
from visualize_gaze_predictions import display_image, load_checkpoint
from device_and_initialization import seed_everything
from ssd_detector.face_detection import getFaces, load_detector


def main() -> None:
    """Load trained gaze/face models and render predictions for supported image files."""
    options = command_line_options.optionargparser().parse_args()
    seed_everything(options.seed)
    gaze_model, device = load_checkpoint(options)
    detector = load_detector(options.face_model, device)
    output_directory = Path(options.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    for image_path in sorted(Path(options.input_dir).iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        with Image.open(image_path) as source_image:
            rgb_image = source_image.convert("RGB")
            head_images, eye_position_grids, eye_positions = getFaces(
                np.asarray(rgb_image), detector=detector
            )
            scene_image = TF.to_tensor(TF.resize(rgb_image, [227, 227]))
        if not len(head_images):
            continue
        scene_image = TF.normalize(scene_image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        head_images = TF.normalize(head_images, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        with torch.inference_mode():
            heatmaps = gaze_model.predict(
                scene_image.unsqueeze(0).expand(len(head_images), -1, -1, -1).to(device),
                head_images.to(device),
                eye_position_grids.to(device),
            ).cpu()
        figure, axes = plt.subplots()
        axes.imshow(display_image(scene_image))
        for heatmap, (eye_x, eye_y) in zip(heatmaps, eye_positions):
            predicted_row, predicted_column = divmod(heatmap.argmax().item(), 227)
            axes.plot([eye_x * 227, predicted_column], [eye_y * 227, predicted_row])
        figure.savefig(output_directory / f"{image_path.stem}_gaze.jpg")
        plt.close(figure)


if __name__ == "__main__":
    main()
