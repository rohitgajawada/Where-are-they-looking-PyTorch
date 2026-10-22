"""Provide the learning-rate schedule and streaming statistics used during training.

adjust_learning_rate applies the historical epoch-decay schedule and configured
weight decay to every optimizer parameter group. AverageMeter tracks a current
scalar, its sample-weighted sum, sample count, and running average for losses,
validation distances, and timing. Its conventional val/avg/sum/count fields are
retained for existing log-format strings.

get_mean_and_std is a legacy helper for CPU RGB (images,labels) batches. It
averages per-batch means and standard deviations equally, so it is not a pooled
pixel-weighted variance estimator. Gaze-specific distance metrics are kept in
gaze_metrics.py instead of mixing them into this support module.
"""

from __future__ import annotations
from collections.abc import Iterable
import copy
import torch
from torch.optim import Optimizer
from type_definitions import Options


def adjust_learning_rate(options: Options, optimizer: Optimizer, epoch: int) -> None:
    """Apply the configured epoch-decay schedule and weight decay to each parameter group."""
    epoch = copy.deepcopy(epoch)
    learning_rate = options.maxlr
    weight_decay = options.weightDecay
    if options.learningratescheduler == "decayschedular":
        while epoch >= options.decayinterval:
            learning_rate = learning_rate / options.decaylevel
            epoch = epoch - options.decayinterval
    learning_rate = max(learning_rate, options.minlr)
    options.lr = learning_rate

    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate * parameter_group.get("lr_scale", 1.0)
        parameter_group["weight_decay"] = weight_decay


def get_mean_and_std(
    data_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average per-batch RGB means and standard deviations from (images, targets) batches.

    Images must have shape B×3×H×W on CPU. This legacy helper averages batch
    statistics equally; it is not a pooled, pixel-weighted variance estimator.
    """
    channel_means = torch.zeros(3)
    channel_standard_deviations = torch.zeros(3)
    batch_count = 0
    print("==> Computing mean and std..")
    for images, labels in data_loader:
        batch_count += 1
        for channel_index in range(len(images[0])):
            channel_means[channel_index] += images[:, channel_index, :, :].mean()
            channel_standard_deviations[channel_index] += images[:, channel_index, :, :].std()
    channel_means.div_(batch_count)
    channel_standard_deviations.div_(batch_count)
    return channel_means, channel_standard_deviations


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self) -> None:
        """Initialize an empty streaming meter."""
        self.reset()

    def reset(self) -> None:
        """Clear the current value, weighted sum, sample count, and average."""
        self.val: float = 0
        self.avg: float = 0
        self.sum: float = 0
        self.count: int = 0

    def update(self, value: float, sample_count: int = 1) -> None:
        """Record a scalar representing sample_count samples and recompute the weighted average."""
        self.val = value
        self.sum += value * sample_count
        self.count += sample_count
        self.avg = self.sum / self.count
