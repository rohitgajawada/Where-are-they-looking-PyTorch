"""Transform scenes and gaze annotations without altering stored dataset arrays.

Flips mirror both images and all corresponding coordinates on the selected axis.
Scene crops resize to 240x240 before sampling a valid 227x227 window. All scene
annotations and the eye must remain inside the crop; if no such window exists,
the original sample is returned. Head-local coordinates stay unchanged when
only the scene is cropped. Each transform owns copies of its coordinate arrays.
Vertical flips remain available as an explicit experimental option.
"""

from __future__ import annotations
import random
import numpy as np
from skimage import transform
from type_definitions import TransformSample


def copy_sample(sample: TransformSample) -> TransformSample:
    """Copy mutable coordinates while sharing images that transforms only read."""
    return {
        "img": sample["img"],
        "bbox": sample["bbox"],
        "eyes": sample["eyes"].copy(),
        "eyes_bbox": sample["eyes_bbox"].copy(),
        "gaze": sample["gaze"].copy(),
    }


class RandomHorizontalFlip:
    """Mirror scene/head images and every normalized x coordinate together."""

    def __call__(self, sample: TransformSample) -> TransformSample:
        """Return a copied sample, horizontally flipped with probability one half."""
        sample = copy_sample(sample)
        if random.random() > 0.5:
            sample["img"] = sample["img"][:, ::-1]
            sample["bbox"] = sample["bbox"][:, ::-1]
            sample["eyes"][0] = 1 - sample["eyes"][0]
            sample["eyes_bbox"][0] = 1 - sample["eyes_bbox"][0]
            sample["gaze"][0::2] = 1 - sample["gaze"][0::2]
        return sample


class RandomVerticalFlip:
    """Mirror scene/head images and every normalized y coordinate together."""

    def __call__(self, sample: TransformSample) -> TransformSample:
        """Return a copied sample, vertically flipped with probability one half."""
        sample = copy_sample(sample)
        if random.random() > 0.5:
            sample["img"] = sample["img"][::-1]
            sample["bbox"] = sample["bbox"][::-1]
            sample["eyes"][1] = 1 - sample["eyes"][1]
            sample["eyes_bbox"][1] = 1 - sample["eyes_bbox"][1]
            sample["gaze"][1::2] = 1 - sample["gaze"][1::2]
        return sample


class RandomCrop:
    """Sample only scene crops that retain the eye and every gaze annotation."""

    def __init__(self) -> None:
        """Configure the resized scene dimensions and output crop dimensions."""
        self.scale_size: tuple[int, int] = (240, 240)
        self.crop_size: tuple[int, int] = (227, 227)

    def __call__(self, sample: TransformSample) -> TransformSample:
        """Transform normalized scene coordinates with resized dimensions and crop offsets.

        Coordinates use (position*resized_size-offset)/crop_size. Head-local
        coordinates do not change because the head image itself is not cropped.
        There is no coordinate clamping that could invent a new target location.
        """
        sample = copy_sample(sample)
        resized_height, resized_width = self.scale_size
        crop_height, crop_width = self.crop_size
        points = np.concatenate((sample["eyes"].reshape(-1, 2), sample["gaze"].reshape(-1, 2)))
        pixel_points = points * [resized_width, resized_height]
        smallest_offset = np.maximum(
            0, np.ceil(pixel_points.max(0) - [crop_width, crop_height])
        ).astype(int)
        largest_offset = np.minimum(
            [resized_width - crop_width, resized_height - crop_height],
            np.floor(pixel_points.min(0)),
        ).astype(int)
        if (smallest_offset > largest_offset).any():
            return sample
        crop_left = np.random.randint(smallest_offset[0], largest_offset[0] + 1)
        crop_top = np.random.randint(smallest_offset[1], largest_offset[1] + 1)
        scene = transform.resize(sample["img"], self.scale_size, anti_aliasing=True)
        sample["img"] = scene[
            crop_top : crop_top + crop_height, crop_left : crop_left + crop_width
        ]
        sample["eyes"] = (
            sample["eyes"] * [resized_width, resized_height] - [crop_left, crop_top]
        ) / [crop_width, crop_height]
        sample["gaze"] = (
            (
                sample["gaze"].reshape(-1, 2) * [resized_width, resized_height]
                - [crop_left, crop_top]
            )
            / [crop_width, crop_height]
        ).reshape(-1)
        return sample
