"""Verify real gaze-network execution, gradients, and checkpoint reloads on CPU/CUDA.

The run function constructs the actual gaze model with random initialization
and synthetic scene/head images, so no dataset, pretrained weights, or network
download is required. It checks five grid-output shapes, finite loss, nonzero
finite gradients in both branches, an optimizer update, decoded heatmap shape,
and exact model/optimizer checkpoint round-trip predictions.

The CLI defaults to explicit CUDA and fails when CUDA cannot be accessed;
--device cpu is available for diagnostics. The returned JSON records library
versions, device identity, batch size, loss, and completed checks. Passing this
check proves runtime compatibility, not real-data convergence or paper accuracy.
"""

from __future__ import annotations
from type_definitions import SmokeResult
import argparse
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from models.gaze_following_network import Net
from device_and_initialization import resolve_device, seed_everything


def run(device_name: str = "cpu", batch_size: int = 1) -> SmokeResult:
    """Check the real gaze network with synthetic inputs and random initialization.

    Args:
        device_name: CPU, CUDA, or auto device selection. Explicit CUDA must exist.
        batch_size: Number of synthetic 227×227 scene/head pairs.

    Returns:
        Versions, device identity, scalar loss, and the list of completed checks.

    Raises:
        AssertionError: Shapes, gradients, optimizer updates, or reload parity fail.
        RuntimeError: The requested device is unavailable.
    """
    device = resolve_device(device_name)
    seed_everything(235)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    model = Net(SimpleNamespace(random_init=True)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    scene_images = torch.randn(batch_size, 3, 227, 227, device=device)
    head_images = torch.randn_like(scene_images)
    eye_position_grids = torch.zeros(batch_size, 13, 13, device=device)
    eye_position_grids[:, 6, 6] = 1
    target_classes = torch.arange(batch_size, device=device) % 25
    classifier_weights_before = model.fc_0_0.weight.detach().clone()
    grid_log_probabilities = model(scene_images, head_images, eye_position_grids)
    assert len(grid_log_probabilities) == 5
    assert all(
        grid_log_probabilities_for_head.shape == (batch_size, 25)
        for grid_log_probabilities_for_head in grid_log_probabilities
    )
    classification_loss = torch.stack(
        [
            F.nll_loss(grid_log_probabilities_for_head, target_classes)
            for grid_log_probabilities_for_head in grid_log_probabilities
        ]
    ).mean()
    assert torch.isfinite(classification_loss)
    classification_loss.backward()
    for model_branch in (model.saliency_branch, model.gaze_branch):
        gradients = [
            parameter.grad for parameter in model_branch.parameters() if parameter.grad is not None
        ]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
        assert any(gradient.abs().sum() > 0 for gradient in gradients)
    optimizer.step()
    assert not torch.equal(classifier_weights_before, model.fc_0_0.weight)
    model.eval()
    with torch.inference_mode():
        heatmaps = model.predict(scene_images, head_images, eye_position_grids)
        assert heatmaps.shape == (batch_size, 227, 227)
        assert torch.isfinite(heatmaps).all()
        expected_grid_log_probabilities = model(scene_images, head_images, eye_position_grids)
    with tempfile.TemporaryDirectory() as checkpoint_directory:
        checkpoint_path = Path(checkpoint_directory) / "checkpoint.pt"
        torch.save(
            {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint_path
        )
        checkpoint_state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint_state["state_dict"])
        optimizer.load_state_dict(checkpoint_state["optimizer"])
        with torch.inference_mode():
            reloaded_grid_log_probabilities = model(scene_images, head_images, eye_position_grids)
        for reloaded_probabilities, expected_probabilities in zip(
            reloaded_grid_log_probabilities, expected_grid_log_probabilities
        ):
            torch.testing.assert_close(reloaded_probabilities, expected_probabilities)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    smoke_result = {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "device": str(device),
        "batch_size": batch_size,
        "loss": classification_loss.item(),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "checks": [
            "forward",
            "backward",
            "both_branch_gradients",
            "optimizer_step",
            "predict",
            "checkpoint_round_trip",
        ],
    }
    print(json.dumps(smoke_result, indent=2))
    return smoke_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    arguments = parser.parse_args()
    run(arguments.device, arguments.batch_size)
