"""Load original-format GazeFollow annotations and construct scene/head model inputs.

GazeDataset reads the MATLAB cell arrays for one person-image example at a time,
loads the RGB scene, derives the historical eye-centered head crop from the
person box, and creates a 13x13 one-hot eye-position grid. It returns normalized
scene/head tensors, five 5x5 classification targets, sample metadata, and a
100-slot gaze-coordinate array padded with -1 for unused annotation pairs.
GazeFollow owns the train/test datasets and their seeded PyTorch DataLoaders.

Scene and gaze coordinates use normalized (x,y) positions; tensor images use
channel-first layout. This loader retains the known annotation-aliasing, crop,
and shifted-grid issues pending their separate reproduction fixes. Disable
augmentation with --no-withaug for initial diagnostics. Loading a sample may
mutate the original annotation arrays under the historical augmentation path.
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
import matplotlib.pyplot as plt
import math
from torch.utils.data import Dataset
from torchvision import transforms
import scipy.io as scipy_io
from gaze_augmentations import RandomHorizontalFlip, RandomVerticalFlip, RandomCrop


def getCropped(person_image: ImageArray, eye_position: FloatArray) -> ImageArray:
    """Extract the historical eye-centered 30% crop and resize it to 227×227.

    Args:
        person_image: H×W×3 RGB image, usually an already resized person bounding box.
        eye_position: Normalized (x,y) eye location relative to that image.

    Returns:
        The resized crop, or the input image if resizing fails. The historical
        clipping and fallback behavior is preserved, including its known limits.
    """

    crop_fraction = 0.3
    crop_width = int(math.floor(crop_fraction * person_image.shape[1]))
    crop_height = int(math.floor(crop_fraction * person_image.shape[0]))

    if crop_width % 2 == 0:
        crop_width = crop_width + 1

    if crop_height % 2 == 0:
        crop_height = crop_height + 1

    padding_canvas = np.ones((crop_height, crop_width, 3))
    padding_canvas[:, :, 0] = 123 * np.ones((crop_height, crop_width))
    padding_canvas[:, :, 1] = 117 * np.ones((crop_height, crop_width))
    padding_canvas[:, :, 2] = 104 * np.ones((crop_height, crop_width))

    eye_center_pixels = [
        math.floor(eye_position[0] * person_image.shape[1]),
        math.floor(eye_position[1] * person_image.shape[0]),
    ]
    crop_half_width = math.floor((crop_width - 1) / 2)
    crop_half_height = math.floor((crop_height - 1) / 2)

    crop_left = eye_center_pixels[0] - crop_half_width - 1
    left_padding = 0
    if crop_left < 0:
        left_padding = 1 - crop_left
        crop_left = 0

    crop_right = eye_center_pixels[0] + crop_half_width - 1
    right_padding_boundary = crop_width - 1
    if crop_right > person_image.shape[1] - 1:
        right_padding_boundary = crop_width - (crop_right - person_image.shape[1] + 1)
        crop_right = person_image.shape[1] - 1

    crop_top = eye_center_pixels[1] - crop_half_height - 1
    top_padding = 0
    if crop_top < 0:
        top_padding = 1 - crop_top
        crop_top = 0

    crop_bottom = eye_center_pixels[1] + crop_half_height - 1
    bottom_padding_boundary = crop_height - 1
    if crop_bottom > person_image.shape[0] - 1:
        bottom_padding_boundary = crop_height - (crop_bottom - person_image.shape[0] + 1)
        crop_bottom = person_image.shape[0] - 1

    head_crop = person_image[
        int(crop_top) : int(crop_bottom + 1), int(crop_left) : int(crop_right + 1), :
    ]
    head_crop = np.ascontiguousarray(head_crop)

    try:
        head_crop = transform.resize(head_crop, (227, 227))
        return head_crop
    except:
        # print('begins') #TODO, are there corrupt images?
        # print(x)
        # print(x.shape)
        # print('ends')
        print("corruption")
        return person_image


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

        This historical loader flattens cell contents and rewrites annotation arrays
        in the supplied mapping; callers should load a fresh mapping for each dataset.
        """

        self.split = split
        self.options = options

        if split == "train":
            person_boxes = annotations["train_bbox"][0]
            eye_positions = annotations["train_eyes"][0]
            gaze_points = annotations["train_gaze"][0]
            source_metadata = annotations["train_meta"][0]
            image_paths = annotations["train_path"]

        if split == "test":
            person_boxes = annotations["test_bbox"][0]
            eye_positions = annotations["test_eyes"][0]
            gaze_points = annotations["test_gaze"][0]
            source_metadata = annotations["test_meta"][0]
            image_paths = annotations["test_path"]

        for sample_index in range(image_paths.shape[0]):
            image_paths[sample_index] = image_paths[sample_index][0]
        image_paths = image_paths.flatten()
        image_paths = np.array(
            [
                str(Path(dataset_directory) / str(relative_image_path))
                for relative_image_path in image_paths
            ],
            dtype=object,
        )

        for sample_index in range(person_boxes.shape[0]):
            person_boxes[sample_index] = person_boxes[sample_index].flatten()

        for sample_index in range(eye_positions.shape[0]):
            eye_positions[sample_index] = eye_positions[sample_index].flatten()

        for sample_index in range(gaze_points.shape[0]):
            gaze_points[sample_index] = gaze_points[sample_index].flatten()

        self.image_paths: NDArray[np.object_] = image_paths
        self.person_boxes: NDArray[np.object_] = person_boxes
        self.eye_positions: NDArray[np.object_] = eye_positions
        self.gaze_points: NDArray[np.object_] = gaze_points

    def __len__(self) -> int:
        """Return the number of person-image annotation examples in the split."""
        return len(self.image_paths)

    def __getitem__(self, sample_index: int) -> GazeSample:
        """Build one scene/head pair, eye grid, five grid labels, and gaze annotations.

        Returns:
            Scene and head float tensors (3,227,227); eye grid (13,13); grid labels
            (1,5,25); scene eyes (2,); integer sample index; head-local eyes (2,);
            and flattened gaze coordinates (100,) padded with -1.

        Training augmentation is optional. Existing annotation aliasing and crop/grid
        geometry behavior is retained; disable augmentation for initial diagnostics.
        """
        image_path = self.image_paths[sample_index]
        scene_image = io.imread(image_path)

        image_shape = scene_image.shape
        person_box = self.person_boxes[sample_index]
        person_box[person_box < 0] = 0.0
        head_image = np.copy(
            scene_image[
                int(person_box[1] * image_shape[0]) : int(
                    np.ceil(person_box[1] * image_shape[0] + person_box[3] * image_shape[0])
                ),
                int(person_box[0] * image_shape[1]) : int(
                    np.ceil(person_box[0] * image_shape[1] + person_box[2] * image_shape[1])
                ),
            ]
        )

        head_image = np.ascontiguousarray(head_image)
        head_image = transform.resize(head_image, (227, 227))

        scene_image = np.ascontiguousarray(scene_image)
        scene_image = transform.resize(scene_image, (227, 227))

        gaze_points = self.gaze_points[sample_index]

        eye_position = self.eye_positions[sample_index]
        head_eye_position = (eye_position - person_box[:2]) / person_box[2:]

        if len(scene_image.shape) == 2:
            print("stacking")
            scene_image = np.stack((scene_image,) * 3, axis=-1)

        if len(head_image.shape) == 2:
            print("stacking bbox")
            head_image = np.stack((head_image,) * 3, axis=-1)

        head_image = getCropped(head_image, head_eye_position)
        head_image = np.ascontiguousarray(head_image)
        head_image = transform.resize(head_image, (227, 227))

        eye_grid_resolution = 13
        gaze_grid_resolution = 5

        ######DO DATA AUG HERE###########

        if self.split == "train" and getattr(self.options, "withaug", True):

            augmentation_pipeline = transforms.Compose(
                [RandomHorizontalFlip(), RandomVerticalFlip(), RandomCrop()]
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

        # SHIFTED GRIDS STUFF

        eye_position_grid = np.zeros((eye_grid_resolution, eye_grid_resolution))
        eye_position_grid[int(np.floor(eye_grid_resolution * eye_position[1]))][
            int(np.floor(eye_grid_resolution * eye_position[0]))
        ] = 1

        grid_count = 5

        grid_x_offsets = [0, 1, -1, 0, 0]
        grid_y_offsets = [0, 0, 0, -1, 1]

        gaze_x = gaze_points[0]
        gaze_y = gaze_points[1]

        shifted_grid_targets = np.zeros((grid_count, gaze_grid_resolution, gaze_grid_resolution))
        for grid_index in range(5):

            target_column = int(
                np.floor(
                    gaze_grid_resolution * gaze_points[0]
                    + (grid_x_offsets[grid_index] * (1 / (grid_count * 3.0)))
                )
            )
            target_row = int(
                np.floor(
                    gaze_grid_resolution * gaze_points[1]
                    + (grid_y_offsets[grid_index] * (1 / (grid_count * 3.0)))
                )
            )

            if target_column < 0:
                target_column = 0
            elif target_column > 4:
                target_column = 4
            if target_row < 0:
                target_row = 0
            elif target_row > 4:
                target_row = 4

            try:
                shifted_grid_targets[grid_index][target_row][target_column] = 1
            except:
                print("**************")
                print("eyes: ", eye_position)
                print("eyes_bbox: ", head_eye_position)
                print("gaze: ", gaze_points)
                print("grid vals: ", target_column, target_row)
                plt.imshow(scene_image)
                plt.savefig("a.jpg")
                exit()

        eye_position_grid = torch.from_numpy(eye_position_grid).contiguous()
        shifted_grid_targets = torch.from_numpy(shifted_grid_targets).contiguous()

        shifted_grid_targets = shifted_grid_targets.view(1, 5, 25)
        padded_gaze_annotations = np.ones(100)
        padded_gaze_annotations *= -1
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

        training_annotations = scipy_io.loadmat(Path(options.data_dir) / "train_annotations.mat")
        test_annotations = scipy_io.loadmat(Path(options.data_dir) / "test_annotations.mat")

        self.train_dataset: GazeDataset = GazeDataset(
            training_annotations, "train", options.data_dir, options
        )
        self.train_loader: torch.utils.data.DataLoader[GazeSample] = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=options.batch_size,
            shuffle=True,
            num_workers=options.workers,
            worker_init_fn=seed_worker,
            pin_memory=str(options.device).startswith("cuda"),
        )

        self.test_dataset: GazeDataset = GazeDataset(
            test_annotations, "test", options.data_dir, options
        )
        self.test_loader: torch.utils.data.DataLoader[GazeSample] = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=options.testbatchsize,
            shuffle=False,
            num_workers=options.workers,
            worker_init_fn=seed_worker,
            pin_memory=str(options.device).startswith("cuda"),
        )
