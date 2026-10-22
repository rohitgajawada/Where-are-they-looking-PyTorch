"""Test device selection, model execution, checkpoints, metrics, and CLI entry points.

The suite generates a tiny original-format MATLAB dataset and RGB image, then
runs the real loader, trainer, validator, and checkpoint resume at batch size
one. Other checks cover boolean options, explicit CUDA failure, Places tensor
state dictionaries and trusted legacy loading, distance definitions, empty
S3FD detections, feature normalization, and import-safe --help commands.

CUDA-specific integration and batch-size-two smoke tests run only when a GPU
is exposed; CPU checks remain useful in a sandbox. Generated weights and data
avoid network downloads and do not establish real GazeFollow convergence.
"""

from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.io
import torch
from PIL import Image

import gazefollow_dataset
import models
import command_line_options
import gaze_metrics
from device_and_initialization import load_places_features, resolve_device, seed_everything
from ssd_detector.s3fd_network import L2Norm
from ssd_detector.face_detection import detect, getFaces
from training_engine import Trainer, Validator


@pytest.fixture(autouse=True)
def deterministic() -> None:
    """Limit CPU threads and reset random seeds before each test."""
    torch.set_num_threads(4)
    seed_everything(235)


def test_boolean_flags_and_seed() -> None:
    """Verify explicit negative boolean flags and integer seed parsing."""
    options = command_line_options.optionargparser().parse_args(
        ["--no-withaug", "--no-nesterov", "--testOnly", "--seed", "7"]
    )
    assert not options.withaug and not options.nesterov and options.testOnly and options.seed == 7


def test_explicit_cuda_never_silently_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify explicit CUDA errors rather than silently switching to CPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto").type == "cpu"
    with pytest.raises(RuntimeError, match="CUDA requested"):
        resolve_device("cuda")


def test_places_state_dict_and_explicit_legacy_opt_in(tmp_path: Path) -> None:
    """Verify tensor-only Places loading and explicit trust for legacy pickled modules."""
    backbone = load_places_features(SimpleNamespace(random_init=True))
    safe_path = tmp_path / "places.pt"
    torch.save(
        {
            "state_dict": {
                "module.features." + parameter_name: parameter_tensor
                for parameter_name, parameter_tensor in backbone.state_dict().items()
            }
        },
        safe_path,
    )
    loaded_backbone = load_places_features(SimpleNamespace(placesmodelpath=safe_path))
    for key, parameter_tensor in backbone.state_dict().items():
        torch.testing.assert_close(parameter_tensor, loaded_backbone.state_dict()[key])
    legacy_path = tmp_path / "legacy.pt"
    torch.save(backbone, legacy_path)
    with pytest.raises(RuntimeError, match="trust-legacy-checkpoint"):
        load_places_features(SimpleNamespace(placesmodelpath=legacy_path))
    loaded_backbone = load_places_features(
        SimpleNamespace(placesmodelpath=legacy_path, trust_legacy_checkpoint=True)
    )
    torch.testing.assert_close(backbone[0].weight, loaded_backbone[0].weight)


def test_metrics_mean_location_and_batch_invariance() -> None:
    """Check mean-location/minimum distances and invariance to batch partitioning."""
    # Prediction at (0,0); annotations at (0,0), (1,0): distance to mean=.5, min=0.
    gaze_annotations = torch.full((2, 100), -1.0)
    gaze_annotations[0, :4] = torch.tensor([0.0, 0.0, 1.0, 0.0])
    gaze_annotations[1, :2] = torch.tensor([0.5, 0.5])
    predicted_flat_indices = torch.tensor([0, 50 * 227 + 100])
    expected_second_distance = ((100 / 227 - 0.5) ** 2 + (50 / 227 - 0.5) ** 2) ** 0.5
    assert gaze_metrics.euclid_dist(predicted_flat_indices, gaze_annotations, 2) == pytest.approx(
        (0.5 + expected_second_distance) / 2
    )
    assert gaze_metrics.euclid_mindist(predicted_flat_indices[:1], gaze_annotations[:1], 1) == 0
    separately_averaged_distance = (
        sum(
            gaze_metrics.euclid_dist(
                predicted_flat_indices[sample_index : sample_index + 1],
                gaze_annotations[sample_index : sample_index + 1],
                1,
            )
            for sample_index in range(2)
        )
        / 2
    )
    assert gaze_metrics.euclid_dist(predicted_flat_indices, gaze_annotations, 2) == pytest.approx(
        separately_averaged_distance
    )


class Writer:
    """Discard TensorBoard scalars while retaining the writer call interface."""

    def add_scalar(self, *args: object) -> None:
        """Accept and discard scalar log arguments during integration tests."""
        pass


def synthetic_mat_dataset(tmp_path: Path) -> None:
    """Create one RGB image and original-format MATLAB train/test annotations."""
    Image.fromarray(np.full((300, 300, 3), 127, dtype=np.uint8)).save(tmp_path / "image.jpg")
    annotation_arrays = {}
    for split in ("train", "test"):
        for annotation_field, annotation_value in [
            ("bbox", [0.0, 0.0, 1.0, 1.0]),
            ("eyes", [0.5, 0.5]),
            ("gaze", [0.7, 0.3]),
            ("meta", [1.0]),
        ]:
            matlab_cells = np.empty((1, 1), dtype=object)
            matlab_cells[0, 0] = np.array([annotation_value])
            annotation_arrays[f"{split}_{annotation_field}"] = matlab_cells
        matlab_cells = np.empty((1, 1), dtype=object)
        matlab_cells[0, 0] = "image.jpg"
        annotation_arrays[f"{split}_path"] = matlab_cells
        scipy.io.savemat(
            tmp_path / f"{split}_annotations.mat",
            {
                parameter_name: parameter_tensor
                for parameter_name, parameter_tensor in annotation_arrays.items()
                if parameter_name.startswith(split)
            },
        )


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_real_data_trainer_validator_checkpoint_batch_one(tmp_path: Path, device_name: str) -> None:
    """Exercise real loader/training/validation/resume code with a single synthetic sample."""
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not exposed in this execution environment")
    synthetic_mat_dataset(tmp_path)
    options = command_line_options.optionargparser().parse_args(
        [
            "--device",
            device_name,
            "--random-init",
            "--no-withaug",
            "--workers",
            "0",
            "--batch-size",
            "1",
            "--testbatchsize",
            "1",
            "--data_dir",
            str(tmp_path),
            "--store",
            str(tmp_path / "checkpoints"),
        ]
    )
    dataset_loaders = gazefollow_dataset.GazeFollow(options)
    gaze_model, loss_function, optimizer = models.setup(models.load_model(options), options)
    trainer = Trainer(gaze_model, loss_function, optimizer, options, Writer())
    trainer.train(dataset_loaders.train_loader, 0, options)
    assert np.isfinite(trainer.loss_meter.avg)
    validation_distance = Validator(gaze_model, loss_function, options, Writer()).validate(
        dataset_loaders.test_loader, 0, options
    )
    assert np.isfinite(validation_distance) and 0 <= validation_distance <= 2**0.5
    models.save_checkpoint(options, gaze_model, optimizer, validation_distance, 0)
    options.resume = str(next((tmp_path / "checkpoints").glob("*.pth.tar")))
    gaze_model, optimizer, options, best_validation_distance = models.resumer(
        options, gaze_model, optimizer
    )
    assert options.start_epoch == 1 and best_validation_distance == validation_distance
    assert next(gaze_model.parameters()).device.type == device_name


def test_s3fd_empty_detections_and_norm() -> None:
    """Check empty face detections and finite learned channel normalization."""

    class EmptyDetector(torch.nn.Module):
        """Produce background-only logits for deterministic empty-detection tests."""

        def __init__(self) -> None:
            """Create a dummy parameter so detector device discovery works."""
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(()))

        def forward(self, detector_images: torch.Tensor) -> list[torch.Tensor]:
            """Return one background-only classification map and zero regression offsets."""
            logits = detector_images.new_tensor([100.0, -100.0]).reshape(1, 2, 1, 1)
            return [logits, detector_images.new_zeros((1, 4, 1, 1))]

    detector = EmptyDetector()
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    assert detect(detector, image).shape == (0, 5)
    face_images, eye_position_grids, eye_positions = getFaces(image, detector=detector)
    assert (
        face_images.shape == (0, 3, 227, 227)
        and eye_position_grids.shape == (0, 13, 13)
        and eye_positions == []
    )
    normalization_layer = L2Norm(3, 10)
    torch.testing.assert_close(normalization_layer.weight, torch.full((3,), 10.0))
    assert torch.isfinite(normalization_layer(torch.zeros(1, 3, 2, 2))).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not exposed")
def test_cuda_smoke_batch_two() -> None:
    """Run the full model smoke test with two examples on CUDA."""
    from check_pytorch_runtime import run

    assert run("cuda", 2)["device"] == "cuda"


@pytest.mark.parametrize(
    "command_arguments",
    [
        ["train_gaze_model.py"],
        ["visualize_gaze_predictions.py"],
        ["predict_gaze_from_faces.py"],
        ["-m", "ssd_detector.evaluate_wider_faces"],
    ],
)
def test_cli_help_has_no_asset_or_cuda_requirement(command_arguments: list[str]) -> None:
    """Verify every supported CLI can show help without model or dataset assets."""
    import subprocess
    import sys

    command_result = subprocess.run(
        [sys.executable, *command_arguments, "--help"], capture_output=True, text=True
    )
    assert command_result.returncode == 0, command_result.stderr
    assert "--device" in command_result.stdout
