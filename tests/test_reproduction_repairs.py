"""Test the repaired supervision geometry and actual TensorBoard event contents.

These regression cases check annotations remain unchanged during augmentation,
240-to-227 crop scaling, head-local eye invariance, and exact consistency of the
shared shifted-grid convention at cell interiors and image boundaries. Metrics
are tested with ties and known angles. The integration case writes real event
files to confirm both train and validation series are discoverable by TensorBoard.
"""

from __future__ import annotations
from pathlib import Path
import random
import numpy as np
import pytest
import torch
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from gaze_augmentations import RandomCrop, RandomHorizontalFlip
from shifted_grids import encode_points, decode_grids
from gaze_metrics import fixation_auc, angular_error
from test_pytorch_compatibility import synthetic_mat_dataset
import command_line_options
import gazefollow_dataset
import models
from training_engine import Trainer, Validator
from type_definitions import TransformSample


def sample() -> TransformSample:
    """Build an asymmetric, multi-annotation example with a centered head-local eye."""
    return {
        "img": np.zeros((227, 227, 3), dtype=np.float32),
        "bbox": np.zeros((227, 227, 3), dtype=np.float32),
        "eyes": np.array([0.2, 0.3]),
        "eyes_bbox": np.array([0.5, 0.5]),
        "gaze": np.array([0.7, 0.8, 0.6, 0.4]),
    }


def test_transforms_are_pure_and_crop_uses_resized_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check multiple fixations flip without aliasing and crop coordinates use 240 pixels."""
    original = sample()
    monkeypatch.setattr(random, "random", lambda: 0.9)
    flipped = RandomHorizontalFlip()(original)
    np.testing.assert_allclose(flipped["gaze"], [0.3, 0.8, 0.4, 0.4])
    np.testing.assert_allclose(original["gaze"], [0.7, 0.8, 0.6, 0.4])
    monkeypatch.setattr(np.random, "randint", lambda low, high: 7)
    cropped = RandomCrop()(original)
    np.testing.assert_allclose(cropped["eyes"], (original["eyes"] * 240 - 7) / 227)
    np.testing.assert_allclose(cropped["gaze"], (original["gaze"] * 240 - 7) / 227)
    np.testing.assert_array_equal(cropped["eyes_bbox"], original["eyes_bbox"])
    assert np.shares_memory(original["gaze"], cropped["gaze"]) is False


def test_crop_that_cannot_retain_points_falls_back() -> None:
    """Keep labels intact when the eye and gaze cannot fit in any crop window."""
    original = sample()
    original["eyes"] = np.array([0.0, 0.0])
    original["gaze"] = np.array([1.0, 1.0])
    cropped = RandomCrop()(original)
    np.testing.assert_array_equal(original["gaze"], cropped["gaze"])
    assert cropped["img"].shape == (227, 227, 3)


def test_grid_targets_and_decoder_agree_at_all_fine_grid_centers() -> None:
    """Ensure every 15x15 location is a maximum of its synthetic encoded prediction."""
    axis = (torch.arange(15, dtype=torch.float64) + 0.5) / 15
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    points = torch.stack((x, y), -1).reshape(-1, 2)
    labels = encode_points(points)
    logits = torch.full((225, 5, 25), -30.0)
    logits.scatter_(2, labels[..., None], 0.0)
    heatmaps = decode_grids(list(logits.unbind(1)), resolution=15).flatten(1)
    assert torch.allclose(heatmaps.diagonal(), torch.ones(225))
    assert encode_points(torch.tensor([0.15, 0.5]))[1] == 11
    edge_labels = encode_points(torch.tensor([[0.0, 0.0], [1.0, 1.0]]))
    assert ((edge_labels >= 0) & (edge_labels < 25)).all()


def test_metrics_with_ties_perfect_scores_and_known_angle() -> None:
    """Validate chance/perfect AUC and a right-angle prediction with padded annotations."""
    points = torch.tensor([[0.75, 0.5, -1.0, -1.0]])
    heatmaps = torch.zeros(1, 227, 227)
    assert fixation_auc(heatmaps, points) == pytest.approx(0.5)
    heatmaps[0, 113, 170] = 1
    assert fixation_auc(heatmaps, points, resolution=227) == pytest.approx(1.0)
    # Ground-truth vector points right from the eye; prediction points down.
    angle = angular_error(
        torch.tensor([170 * 227 + 113]),
        torch.tensor([[0.75, 113 / 227]]),
        torch.tensor([[113 / 227, 113 / 227]]),
    )
    assert angle == pytest.approx(90, abs=1e-4)


def test_tensorboard_contains_real_train_and_validation_metrics(tmp_path: Path) -> None:
    """Read actual event files after real training and validation and check paired tags."""
    torch.set_num_threads(4)
    synthetic_mat_dataset(tmp_path)
    options = command_line_options.optionargparser().parse_args(
        [
            "--random-init",
            "--backbone",
            "torchvision",
            "--device",
            "cpu",
            "--workers",
            "0",
            "--batch-size",
            "1",
            "--testbatchsize",
            "1",
            "--no-withaug",
            "--validation-fraction",
            "0",
            "--data_dir",
            str(tmp_path),
        ]
    )
    loaders = gazefollow_dataset.GazeFollow(options)
    model, criterion, optimizer = models.setup(models.load_model(options), options)
    directory = tmp_path / "events"
    with SummaryWriter(str(directory)) as writer:
        Trainer(model, criterion, optimizer, options, writer).train(
            loaders.train_loader, 0, options
        )
        Validator(model, criterion, options, writer).validate(loaders.test_loader, 0, options)
    events = EventAccumulator(str(directory)).Reload()
    for metric in ("loss", "grid_accuracy", "mean_distance", "minimum_distance"):
        for split in ("train", "val"):
            scalars = events.Scalars(f"{metric}/{split}")
            assert len(scalars) == 1 and np.isfinite(scalars[0].value)


def test_annotation_storage_is_immutable_under_augmented_reads(tmp_path: Path) -> None:
    """Read one real loader sample repeatedly and ensure stored coordinates cannot change."""
    synthetic_mat_dataset(tmp_path)
    options = command_line_options.optionargparser().parse_args(
        ["--data_dir", str(tmp_path), "--workers", "0", "--validation-fraction", "0"]
    )
    loaders = gazefollow_dataset.GazeFollow(options)
    dataset = loaders.train_dataset
    original_eye = dataset.eye_positions[0].copy()
    original_gaze = dataset.gaze_points[0].copy()
    for _ in range(10):
        dataset[0]
    np.testing.assert_array_equal(dataset.eye_positions[0], original_eye)
    np.testing.assert_array_equal(dataset.gaze_points[0], original_gaze)
    with pytest.raises(ValueError):
        dataset.eye_positions[0][0] = 0.1


def test_development_split_groups_people_from_same_image(tmp_path: Path) -> None:
    """Ensure repeated image paths never cross the deterministic train/validation boundary."""
    import scipy.io

    synthetic_mat_dataset(tmp_path)
    source = scipy.io.loadmat(tmp_path / "train_annotations.mat")
    labels = {}
    for key, value in source.items():
        if key.startswith("train_"):
            if key == "train_path":
                paths = np.empty((128, 1), dtype=object)
                for index in range(128):
                    paths[index, 0] = f"image_{index//2}.jpg"
                labels[key] = paths
            else:
                labels[key] = np.repeat(value, 128, axis=1)
    scipy.io.savemat(tmp_path / "train_annotations.mat", labels)
    options = command_line_options.optionargparser().parse_args(
        ["--data_dir", str(tmp_path), "--workers", "0", "--validation-fraction", ".25"]
    )
    loaders = gazefollow_dataset.GazeFollow(options)
    train = {loaders.train_dataset.image_paths[index] for index in loaders.training_indices}
    val = {loaders.train_dataset.image_paths[index] for index in loaders.validation_indices}
    assert train and val and train.isdisjoint(val)
    assert len(loaders.training_indices) + len(loaders.validation_indices) == 128


def test_official_validation_uses_entire_training_split_and_multiple_fixations(
    tmp_path: Path,
) -> None:
    """Use every training row and preserve official test annotations for epoch validation."""
    import scipy.io

    synthetic_mat_dataset(tmp_path)
    test_annotations = scipy.io.loadmat(tmp_path / "test_annotations.mat")
    test_annotations["test_gaze"][0, 0] = np.array([[0.7, 0.3, 0.4, 0.8]])
    scipy.io.savemat(
        tmp_path / "test_annotations.mat",
        {name: value for name, value in test_annotations.items() if not name.startswith("__")},
    )
    options = command_line_options.optionargparser().parse_args(
        [
            "--data_dir",
            str(tmp_path),
            "--workers",
            "0",
            "--no-withaug",
            "--validation-split",
            "test",
            "--validation-fraction",
            ".99",
        ]
    )
    loaders = gazefollow_dataset.GazeFollow(options)
    assert loaders.training_indices == list(range(len(loaders.train_dataset)))
    assert loaders.validation_indices == list(range(len(loaders.test_dataset)))
    assert loaders.validation_loader is loaders.test_loader
    assert loaders.validation_dataset is loaders.test_dataset
    assert loaders.validation_source == "test"
    batch = next(iter(loaders.validation_loader))
    np.testing.assert_allclose(batch[7][0, :4], [0.7, 0.3, 0.4, 0.8])
    assert (batch[7][0, 4:] == -1).all()
