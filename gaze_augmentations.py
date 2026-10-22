"""Apply historical image flips and random scene crops to gaze-training samples.

Each transform accepts and returns a TransformSample dictionary containing the
scene image, head image, scene eye position, head-relative eye position, and gaze
coordinates. Horizontal/vertical flips mirror the corresponding image axes and
coordinates with probability one half. RandomCrop resizes the scene to 240x240
and chooses a 227x227 window; the separately supplied head image is not cropped.

This module documents existing behavior rather than fixing it: flip coordinates
can alias dataset storage, crop coordinate scaling incorrectly uses 227 instead
of 240, head-local eyes are adjusted without cropping the head, and excluded
points are clamped. These defects must be repaired before relying on augmented
training for a paper-faithful reproduction.
"""

from __future__ import annotations
from type_definitions import TransformSample
from skimage import transform
import numpy as np
import random


class RandomHorizontalFlip(object):
    """Mirror RGB images and normalized eye/gaze x coordinates together."""

    def __init__(self) -> None:
        """Create a horizontal flip with probability one half."""
        pass

    def __call__(self, sample: TransformSample) -> TransformSample:
        """Optionally mirror scene/head images and x coordinates.

        Coordinate arrays are currently modified in place and may alias stored
        dataset annotations. This known defect is documented, not repaired here.
        """

        scene_image = sample["img"]
        head_image = sample["bbox"]
        eye_position = sample["eyes"]
        head_eye_position = sample["eyes_bbox"]
        gaze_points = sample["gaze"]

        scene_height, scene_width = scene_image.shape[:2]
        head_height, head_width = head_image.shape[:2]

        if random.random() > 0.5:

            scene_image = scene_image[:, ::-1]
            head_image = head_image[:, ::-1]
            eye_position[0] = 1 - eye_position[0]
            head_eye_position[0] = 1 - head_eye_position[0]
            gaze_points[0] = 1 - gaze_points[0]

        sample = {
            "img": scene_image,
            "bbox": head_image,
            "eyes": eye_position,
            "eyes_bbox": head_eye_position,
            "gaze": gaze_points,
        }

        return sample


class RandomVerticalFlip(object):
    """Mirror RGB images and normalized eye/gaze y coordinates together."""

    def __init__(self) -> None:
        """Create a vertical flip with probability one half."""
        pass

    def __call__(self, sample: TransformSample) -> TransformSample:
        """Optionally mirror scene/head images and y coordinates.

        Coordinate arrays are currently modified in place and may alias stored
        dataset annotations. This known defect is documented, not repaired here.
        """

        scene_image = sample["img"]
        head_image = sample["bbox"]
        eye_position = sample["eyes"]
        head_eye_position = sample["eyes_bbox"]
        gaze_points = sample["gaze"]

        scene_height, scene_width = scene_image.shape[:2]
        head_height, head_width = head_image.shape[:2]

        if random.random() > 0.5:

            scene_image = scene_image[::-1, :]
            head_image = head_image[::-1, :]
            eye_position[1] = 1 - eye_position[1]
            head_eye_position[1] = 1 - head_eye_position[1]
            gaze_points[1] = 1 - gaze_points[1]

        sample = {
            "img": scene_image,
            "bbox": head_image,
            "eyes": eye_position,
            "eyes_bbox": head_eye_position,
            "gaze": gaze_points,
        }

        return sample


class RandomCrop(object):
    """Apply the original scene crop, retaining known coordinate defects for now."""

    def __init__(self) -> None:
        """Configure a 240×240 scene resize followed by a random 227×227 crop."""

        self.scale_size: tuple[int, int] = (240, 240)
        self.crop_size: tuple[int, int] = (227, 227)

    def __call__(self, sample: TransformSample) -> TransformSample:
        """Apply the historical scene crop and coordinate adjustment.

        Known defects remain: coordinate scaling uses 227 despite resizing to 240,
        head-local eyes are changed without cropping the head image, and excluded
        points are clamped to arbitrary coordinates. This is not corrected geometry.
        """

        scene_image = sample["img"]
        head_image = sample["bbox"]
        eye_position = sample["eyes"]
        head_eye_position = sample["eyes_bbox"]
        gaze_points = sample["gaze"]

        scene_height, scene_width = scene_image.shape[:2]

        resized_height, resized_width = self.scale_size
        crop_height, crop_width = self.crop_size

        scene_image = transform.resize(scene_image, (resized_height, resized_width))

        crop_top = np.random.randint(0, resized_height - crop_height)
        crop_left = np.random.randint(0, resized_width - crop_width)

        scene_image = scene_image[
            crop_top : crop_top + crop_height, crop_left : crop_left + crop_width
        ]

        eye_position = (eye_position * 227 - [crop_left, crop_top]) / (
            1.0 * 227
        )  ##assuming that w is equal to h
        head_eye_position = (head_eye_position * 227 - [crop_left, crop_top]) / (1.0 * 227)
        gaze_points = (gaze_points * 227 - [crop_left, crop_top]) / (1.0 * 227)

        if eye_position[0] < 0:
            eye_position[0] = 0.05
        if eye_position[1] < 0:
            eye_position[1] = 0.05
        if eye_position[0] > 1:
            eye_position[0] = 0.95
        if eye_position[1] > 1:
            eye_position[1] = 0.95

        if head_eye_position[0] < 0:
            head_eye_position[0] = 0.05
        if head_eye_position[1] < 0:
            head_eye_position[1] = 0.05
        if head_eye_position[0] > 1:
            head_eye_position[0] = 0.95
        if head_eye_position[1] > 1:
            head_eye_position[1] = 0.95

        if gaze_points[0] < 0:
            gaze_points[0] = 0.05
        if gaze_points[1] < 0:
            gaze_points[1] = 0.05
        if gaze_points[0] > 1:
            gaze_points[0] = 0.95
        if gaze_points[1] > 1:
            gaze_points[1] = 0.95

        sample = {
            "img": scene_image,
            "bbox": head_image,
            "eyes": eye_position,
            "eyes_bbox": head_eye_position,
            "gaze": gaze_points,
        }

        return sample
