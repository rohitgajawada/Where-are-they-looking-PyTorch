"""Load original-format GazeFollow annotations and construct scene/head model inputs.

GazeDataset reads the MATLAB cell arrays for one person-image example at a time,
loads the RGB scene, derives the historical eye-centered head crop from the
person box, and creates a 13x13 one-hot eye-position grid. It returns normalized
scene/head tensors, five 5x5 classification targets, sample metadata, and a
100-slot gaze-coordinate array padded with -1 for unused annotation pairs.
GazeFollow owns the train/test datasets and their seeded PyTorch DataLoaders.

Scene and gaze coordinates use normalized (x,y) positions; tensor images use
channel-first layout. Annotation storage is immutable. RGB conversion precedes
cropping, crops are padded consistently, and the grid specification is shared
with decoding. Validation defaults to an image-disjoint partition of training images.
With --validation-split test, all training examples are used and corrected
test2 supplies epoch validation, including its multiple fixation annotations.
"""

from __future__ import print_function, division
from __future__ import annotations
from typing import Literal
from numpy.typing import NDArray
from type_definitions import FloatArray, GazeSample, ImageArray, MatlabAnnotations, Options
from pathlib import Path
from device_and_initialization import seed_worker
import torch
from skimage import io, transform
import numpy as np
import hashlib
from PIL import Image
from torch.utils.data import Subset
from shifted_grids import encode_points
import math
from torch.utils.data import Dataset
from torchvision import transforms
import scipy.io as scipy_io
from gaze_augmentations import RandomHorizontalFlip, RandomVerticalFlip, RandomCrop


def getCropped(person_image: ImageArray, eye_position: FloatArray) -> ImageArray:
    """Extract a padded eye-centered 30% crop and resize it to 227×227.

    Args:
        person_image: H×W×3 RGB image, usually an already resized person bounding box.
        eye_position: Normalized (x,y) eye location relative to that image.

    Returns:
        The resized RGB crop with mean-colored padding outside the person image.
        Bounds use zero-based half-open indexing with no silent failure fallback.
    """

    crop_height = max(1, int(round(0.3 * person_image.shape[0])))
    crop_width = max(1, int(round(0.3 * person_image.shape[1])))
    center_x = int(np.floor(eye_position[0] * person_image.shape[1]))
    center_y = int(np.floor(eye_position[1] * person_image.shape[0]))
    left = center_x - crop_width // 2
    top = center_y - crop_height // 2
    right, bottom = left + crop_width, top + crop_height
    canvas = np.empty((crop_height, crop_width, 3), dtype=np.float32)
    canvas[:] = [0.485, 0.456, 0.406]
    source_left, source_top = max(0, left), max(0, top)
    source_right, source_bottom = min(person_image.shape[1], right), min(
        person_image.shape[0], bottom
    )
    if source_right > source_left and source_bottom > source_top:
        canvas[
            source_top - top : source_bottom - top, source_left - left : source_right - left
        ] = person_image[source_top:source_bottom, source_left:source_right]
    return transform.resize(canvas, (227, 227), anti_aliasing=True)


class GazeDataset(Dataset[GazeSample]):
    """Expose one person-image example per original GazeFollow annotation."""

    def __init__(
        self,
        annotations: MatlabAnnotations,
        split: Literal["train", "test"],
        dataset_directory: str,
        options: Options,
    ) -> None:
        """Read split-specific MATLAB cell arrays and resolve image paths under dataset_directory.

        MATLAB cell contents are copied into immutable arrays; the supplied mapping
        is never rewritten and repeated augmented reads cannot corrupt annotations.
        """

        self.split = split
        self.options = options

        self.image_paths = np.array(
            [
                str(Path(dataset_directory) / str(cell.item()))
                for cell in annotations[f"{split}_path"].reshape(-1)
            ],
            dtype=object,
        )
        self.person_boxes = [
            np.asarray(cell, dtype=np.float64).reshape(-1).copy()
            for cell in annotations[f"{split}_bbox"].reshape(-1)
        ]
        self.eye_positions = [
            np.asarray(cell, dtype=np.float64).reshape(-1).copy()
            for cell in annotations[f"{split}_eyes"].reshape(-1)
        ]
        self.gaze_points = [
            np.asarray(cell, dtype=np.float64).reshape(-1).copy()
            for cell in annotations[f"{split}_gaze"].reshape(-1)
        ]
        for values in (self.person_boxes, self.eye_positions, self.gaze_points):
            for value in values:
                value.setflags(write=False)

    def __len__(self) -> int:
        """Return the number of person-image annotation examples in the split."""
        return len(self.image_paths)

    def __getitem__(self, sample_index: int) -> GazeSample:
        """Build one scene/head pair, eye grid, five grid labels, and gaze annotations.

        Returns:
            Scene and head float tensors (3,227,227); eye grid (13,13); grid labels
            (1,5,25); scene eyes (2,); integer sample index; head-local eyes (2,);
            and flattened gaze coordinates (100,) padded with -1.

        Training augmentation is optional and never aliases stored annotations.
        Targets use the shared shifted-grid convention and bounded eye indexing.
        """
        image_path = self.image_paths[sample_index]
        with Image.open(image_path) as image:
            scene_image = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
        image_height, image_width = scene_image.shape[:2]
        person_box = self.person_boxes[sample_index].copy()
        left, top = np.floor(person_box[:2] * [image_width, image_height]).astype(int)
        right, bottom = np.ceil(
            (person_box[:2] + person_box[2:]) * [image_width, image_height]
        ).astype(int)
        left, top = max(0, left), max(0, top)
        right, bottom = min(image_width, right), min(image_height, bottom)
        if right <= left or bottom <= top:
            raise ValueError(f"Invalid person box for {image_path}: {person_box}")
        gaze_points = self.gaze_points[sample_index].copy()
        eye_position = self.eye_positions[sample_index].copy()
        if not np.isfinite(gaze_points).all() or not np.isfinite(eye_position).all():
            raise ValueError(f"Non-finite annotations for {image_path}")
        head_eye_position = (eye_position * [image_width, image_height] - [left, top]) / [
            right - left,
            bottom - top,
        ]
        head_image = getCropped(scene_image[top:bottom, left:right], head_eye_position)
        # getCropped centers the eye in its padded canvas.
        head_eye_position = np.array([0.5, 0.5], dtype=np.float64)
        scene_image = transform.resize(scene_image, (227, 227), anti_aliasing=True)

        eye_grid_resolution = 13
        gaze_grid_resolution = 5

        ######DO DATA AUG HERE###########

        if self.split == "train" and getattr(self.options, "withaug", True):

            augmentation_pipeline = transforms.Compose(
                [RandomHorizontalFlip()]
                + (
                    [RandomVerticalFlip()]
                    if getattr(self.options, "vertical_flip", False)
                    else []
                )
                + ([RandomCrop()] if getattr(self.options, "random_crop", True) else [])
            )
            augmented_sample = {
                "img": scene_image,
                "bbox": head_image,
                "eyes": eye_position,
                "eyes_bbox": head_eye_position,
                "gaze": gaze_points,
            }
            augmented_sample = augmentation_pipeline(augmented_sample)

            scene_image = augmented_sample["img"]
            head_image = augmented_sample["bbox"]
            eye_position = augmented_sample["eyes"]
            head_eye_position = augmented_sample["eyes_bbox"]
            gaze_points = augmented_sample["gaze"]

        eye_position_grid = torch.zeros((13, 13), dtype=torch.float32)
        eye_column, eye_row = np.floor(13 * eye_position).clip(0, 12).astype(int)
        eye_position_grid[eye_row, eye_column] = 1
        labels = encode_points(torch.as_tensor(gaze_points[:2]))
        shifted_grid_targets = torch.nn.functional.one_hot(labels, 25).float().view(1, 5, 25)
        padded_gaze_annotations = np.ones(100)
        padded_gaze_annotations *= -1
        if gaze_points.size > 100:
            raise ValueError("More than 50 gaze annotations exceed the sample format")
        padded_gaze_annotations[: gaze_points.shape[0]] = gaze_points

        scene_image = scene_image.transpose((2, 0, 1))
        scene_image = torch.from_numpy(scene_image.copy()).contiguous()

        head_image = head_image.transpose((2, 0, 1))
        head_image = torch.from_numpy(head_image.copy()).contiguous()

        normalize_rgb = transforms.Compose(
            [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

        scene_image = normalize_rgb(scene_image.float())
        head_image = normalize_rgb(head_image.float())

        gaze_sample = (
            scene_image.float(),
            head_image.float(),
            eye_position_grid.float(),
            shifted_grid_targets.float(),
            eye_position,
            sample_index,
            head_eye_position,
            padded_gaze_annotations,
        )

        return gaze_sample


class GazeFollow:
    """Own the training/test datasets and their PyTorch DataLoaders."""

    def __init__(self, options: Options) -> None:
        """Load train/test annotation files and create seeded, optionally pinned DataLoaders."""

        training_annotations = scipy_io.loadmat(
            Path(options.data_dir) / "train_annotations.mat"
        )
        test_annotations = scipy_io.loadmat(Path(options.data_dir) / "test_annotations.mat")

        self.train_dataset: GazeDataset = GazeDataset(
            training_annotations, "train", options.data_dir, options
        )
        self.test_dataset: GazeDataset = GazeDataset(
            test_annotations, "test", options.data_dir, options
        )
        self.validation_source = getattr(options, "validation_split", "development")
        if self.validation_source == "test":
            training_indices = list(range(len(self.train_dataset)))
            validation_indices = list(range(len(self.test_dataset)))
            self.validation_dataset = self.test_dataset
        else:
            development_dataset = GazeDataset(
                training_annotations, "train", options.data_dir, options
            )
            development_dataset.split = "test"  # Disable all training augmentation.
            validation_fraction = getattr(options, "validation_fraction", 0.05)
            validation_indices = []
            training_indices = []
            for index, image_path in enumerate(self.train_dataset.image_paths):
                relative_path = str(Path(image_path).relative_to(options.data_dir))
                bucket = (
                    int(
                        hashlib.sha256(f"{options.seed}:{relative_path}".encode()).hexdigest()[
                            :16
                        ],
                        16,
                    )
                    / 2**64
                )
                (
                    validation_indices if bucket < validation_fraction else training_indices
                ).append(index)
            self.validation_dataset = Subset(development_dataset, validation_indices)
        self.training_indices = training_indices
        self.validation_indices = validation_indices
        self.train_loader = torch.utils.data.DataLoader(
            Subset(self.train_dataset, training_indices),
            batch_size=options.batch_size,
            shuffle=True,
            num_workers=options.workers,
            worker_init_fn=seed_worker,
            pin_memory=str(options.device).startswith("cuda"),
        )
        self.test_loader: torch.utils.data.DataLoader[GazeSample] = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=options.testbatchsize,
            shuffle=False,
            num_workers=options.workers,
            worker_init_fn=seed_worker,
            pin_memory=str(options.device).startswith("cuda"),
        )
        self.validation_loader = (
            self.test_loader
            if self.validation_source == "test"
            else torch.utils.data.DataLoader(
                self.validation_dataset,
                batch_size=options.testbatchsize,
                shuffle=False,
                num_workers=options.workers,
                worker_init_fn=seed_worker,
                pin_memory=str(options.device).startswith("cuda"),
            )
        )
