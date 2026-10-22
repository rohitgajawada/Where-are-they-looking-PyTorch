"""Train and validate gaze models with comparable TensorBoard scalar metrics.

Both loops report sample-weighted five-head NLL, grid accuracy, and normalized
mean/minimum fixation distance. Training charts update during the epoch; epoch
charts allow direct comparison with validation under the same metric definitions.
Training distances describe augmented training examples when augmentation is on.
Validation runs in evaluation mode with gradients disabled. Gradient norms and
per-head accuracy expose optimization failures before an epoch completes.
"""

from __future__ import annotations

import time
import torch
from torch.optim import Optimizer

from type_definitions import GazeLoader, Options, ScalarWriter
from models.gaze_following_network import Net
from training_support import AverageMeter
from shifted_grids import GRID_NAMES, decode_grids
import gaze_metrics


def batch_statistics(
    outputs: list[torch.Tensor],
    targets: torch.Tensor,
    annotations: torch.Tensor,
    loss_function: torch.nn.Module,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute differentiable NLL and detached per-batch accuracy and gaze distances."""
    labels = targets.reshape(-1, 5, 25).argmax(-1)
    head_losses = torch.stack(
        [loss_function(output, labels[:, index]) for index, output in enumerate(outputs)]
    )
    accuracies = torch.stack(
        [
            (output.argmax(-1) == labels[:, index]).float().mean()
            for index, output in enumerate(outputs)
        ]
    )
    with torch.no_grad():
        peaks = decode_grids([output.detach() for output in outputs]).flatten(1).argmax(1)
        distances, minimum_distances = gaze_metrics._distances(
            peaks, annotations, labels.shape[0]
        )
    statistics = {
        "loss": head_losses.mean().item(),
        "grid_accuracy": accuracies.mean().item(),
        "mean_distance": distances.mean().item(),
        "minimum_distance": minimum_distances.mean().item(),
    }
    for index, name in enumerate(GRID_NAMES):
        statistics[f"head_loss/{name}"] = head_losses[index].item()
        statistics[f"head_accuracy/{name}"] = accuracies[index].item()
    return head_losses.mean(), statistics


class Trainer:
    """Optimize the model and stream training metrics and gradient diagnostics."""

    def __init__(
        self,
        model: Net,
        loss_function: torch.nn.Module,
        optimizer: Optimizer,
        options: Options,
        scalar_writer: ScalarWriter,
    ) -> None:
        """Bind training objects and initialize the compatibility loss meter."""
        self.model = model
        self.device = next(model.parameters()).device
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scalar_writer = scalar_writer
        self.loss_meter = AverageMeter()
        self.branch_activations: dict[str, torch.Tensor] = {}
        self.model.saliency_branch.register_forward_hook(self.capture_scene_activation)
        self.model.gaze_branch.register_forward_hook(self.capture_gaze_activation)

    def capture_scene_activation(
        self, module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        """Keep the latest detached scene map for inexpensive activity diagnostics."""
        self.branch_activations["scene"] = output.detach()

    def capture_gaze_activation(
        self, module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        """Keep the latest detached gaze mask for saturation diagnostics."""
        self.branch_activations["gaze"] = output.detach()

    def train(self, train_loader: GazeLoader, epoch: int, options: Options) -> None:
        """Train one epoch with finite checks and matching batch/epoch TensorBoard metrics."""
        self.model.train()
        self.loss_meter.reset()
        meters: dict[str, AverageMeter] = {}
        started = time.monotonic()
        log_interval = max(1, getattr(options, "log_interval", 20))
        for batch_index, batch in enumerate(train_loader):
            scenes, heads, eye_grids, targets = [
                tensor.to(self.device, non_blocking=True) for tensor in batch[:4]
            ]
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(scenes, heads, eye_grids)
            loss, statistics = batch_statistics(outputs, targets, batch[7], self.loss_function)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss at epoch {epoch}, batch {batch_index}"
                )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), float("inf"), error_if_nonfinite=True
            )
            self.optimizer.step()
            batch_size = scenes.shape[0]
            self.loss_meter.update(statistics["loss"], batch_size)
            for metric, value in statistics.items():
                meters.setdefault(metric, AverageMeter()).update(value, batch_size)
            global_step = epoch * len(train_loader) + batch_index
            if batch_index % log_interval == 0 or batch_index + 1 == len(train_loader):
                for metric, value in statistics.items():
                    self.scalar_writer.add_scalar(f"train_batch/{metric}", value, global_step)
                self.scalar_writer.add_scalar(
                    "optimization/gradient_norm", float(gradient_norm), global_step
                )
                for group_index, group in enumerate(self.optimizer.param_groups):
                    self.scalar_writer.add_scalar(
                        f"optimization/learning_rate_{group_index}", group["lr"], global_step
                    )
                for branch_name, activation in self.branch_activations.items():
                    self.scalar_writer.add_scalar(
                        f"activations/{branch_name}_mean", activation.mean().item(), global_step
                    )
                    self.scalar_writer.add_scalar(
                        f"activations/{branch_name}_zero_fraction",
                        (activation == 0).float().mean().item(),
                        global_step,
                    )
                    self.scalar_writer.add_scalar(
                        f"activations/{branch_name}_saturated_fraction",
                        ((activation < 0.01) | (activation > 0.99)).float().mean().item(),
                        global_step,
                    )
                for branch_name, branch in [
                    ("scene", self.model.saliency_branch),
                    ("gaze", self.model.gaze_branch),
                ]:
                    norm = (
                        sum(
                            float(parameter.grad.detach().square().sum())
                            for parameter in branch.parameters()
                            if parameter.grad is not None
                        )
                        ** 0.5
                    )
                    self.scalar_writer.add_scalar(
                        f"optimization/{branch_name}_gradient_norm", norm, global_step
                    )
            if batch_index % options.printfreq == 0 and options.verbose:
                print(
                    f'Epoch {epoch + 1} [{batch_index}/{len(train_loader)}] loss={self.loss_meter.avg:.4f} accuracy={meters["grid_accuracy"].avg:.3f}',
                    flush=True,
                )
        for metric, meter in meters.items():
            self.scalar_writer.add_scalar(f"{metric}/train", meter.avg, epoch + 1)
        self.scalar_writer.add_scalar(
            "runtime/train_seconds", time.monotonic() - started, epoch + 1
        )
        print(f"Train epoch {epoch + 1}: loss={self.loss_meter.avg:.5f}", flush=True)


class Validator:
    """Evaluate loss, classification accuracy, and gaze distances without optimization."""

    def __init__(
        self,
        model: Net,
        loss_function: torch.nn.Module,
        options: Options,
        scalar_writer: ScalarWriter,
    ) -> None:
        """Bind validation objects and initialize compatibility metric meters."""
        self.model = model
        self.device = next(model.parameters()).device
        self.loss_function = loss_function
        self.scalar_writer = scalar_writer
        self.mean_distance_meter = AverageMeter()
        self.minimum_distance_meter = AverageMeter()
        self.loss_meter = AverageMeter()

    def validate(self, validation_loader: GazeLoader, epoch: int, options: Options) -> float:
        """Evaluate all validation batches and log metrics using the same names as training."""
        self.model.eval()
        meters: dict[str, AverageMeter] = {}
        started = time.monotonic()
        with torch.no_grad():
            for batch_index, batch in enumerate(validation_loader):
                scenes, heads, eye_grids, targets = [
                    tensor.to(self.device, non_blocking=True) for tensor in batch[:4]
                ]
                outputs = self.model(scenes, heads, eye_grids)
                loss, statistics = batch_statistics(
                    outputs, targets, batch[7], self.loss_function
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite validation loss")
                heatmaps = decode_grids(outputs)
                peaks = heatmaps.flatten(1).argmax(1)
                statistics["auc_64"] = gaze_metrics.fixation_auc(heatmaps, batch[7])
                statistics["angular_error_degrees"] = gaze_metrics.angular_error(
                    peaks, batch[7], batch[4]
                )
                for metric, value in statistics.items():
                    meters.setdefault(metric, AverageMeter()).update(value, scenes.shape[0])
                if batch_index % options.printfreq == 0 and options.verbose:
                    print(
                        f'Validation [{batch_index}/{len(validation_loader)}] loss={meters["loss"].avg:.4f} distance={meters["mean_distance"].avg:.4f}',
                        flush=True,
                    )
        if not meters:
            raise ValueError("Validation loader is empty")
        for metric, meter in meters.items():
            self.scalar_writer.add_scalar(f"{metric}/val", meter.avg, epoch + 1)
        self.scalar_writer.add_scalar(
            "runtime/val_seconds", time.monotonic() - started, epoch + 1
        )
        self.loss_meter = meters["loss"]
        self.mean_distance_meter = meters["mean_distance"]
        self.minimum_distance_meter = meters["minimum_distance"]
        print(
            f"Validation epoch {epoch + 1}: loss={self.loss_meter.avg:.5f} distance={self.mean_distance_meter.avg:.5f}",
            flush=True,
        )
        return self.mean_distance_meter.avg
