"""Construct the gaze model, training loss, optimizer, and resumable checkpoints.

load_model builds Net and moves it to the configured device. setup creates the
NLL loss for grid log-probabilities and the selected SGD or Adam optimizer.
save_checkpoint writes the model/optimizer states, next epoch, and best distance;
resumer restores those states with explicit device mapping and safe tensor-only
loading, updating the configured start epoch.

Checkpoint field names and historical filename prefixes remain compatible.
These helpers do not create datasets or run training. They are re-exported by
models.__init__ for callers that already use models.setup or models.resumer.
"""

from __future__ import annotations
from type_definitions import Options
from torch.optim import Optimizer
import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import shutil
import numpy as np
from pathlib import Path
from device_and_initialization import resolve_device
import models.gaze_following_network as gaze_network


def setup(
    gaze_model: gaze_network.Net, options: Options
) -> tuple[gaze_network.Net, nn.Module, Optimizer]:
    """Create the configured NLL criterion and SGD/Adam optimizer for the gaze_model."""

    if options.criterion == "crossentropy":
        loss_function = nn.NLLLoss().to(next(gaze_model.parameters()).device)

    pretrained_parameters = []
    new_parameters = []
    for name, parameter in gaze_model.named_parameters():
        (pretrained_parameters if ".features." in name else new_parameters).append(parameter)
    parameter_groups = [
        {
            "params": pretrained_parameters,
            "lr_scale": getattr(options, "pretrained_lr_scale", 0.1),
            "lr": options.maxlr * getattr(options, "pretrained_lr_scale", 0.1),
        },
        {"params": new_parameters, "lr_scale": 1.0, "lr": options.maxlr},
    ]
    if options.optimType == "sgd":
        optimizer = optim.SGD(
            parameter_groups,
            lr=options.maxlr,
            momentum=options.momentum,
            weight_decay=options.weightDecay,
            nesterov=options.nesterov,
        )
    elif options.optimType == "adam":
        optimizer = optim.Adam(
            parameter_groups, lr=options.maxlr, weight_decay=options.weightDecay
        )

    return gaze_model, loss_function, optimizer


def save_checkpoint(
    options: Options,
    gaze_model: gaze_network.Net,
    optimizer: Optimizer,
    best_validation_distance: float,
    epoch: int,
) -> None:
    """Save model/optimizer state, next epoch, and best validation distance under options.store."""
    checkpoint_state = {
        "epoch": epoch + 1,
        "state_dict": gaze_model.state_dict(),
        "best_err": best_validation_distance,
        "optimizer": optimizer.state_dict(),
        "configuration": vars(options),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python_rng_state": random.getstate(),
        "numpy_rng_state": {
            "name": np.random.get_state()[0],
            "keys": torch.tensor(np.random.get_state()[1].astype("int64")),
            "position": np.random.get_state()[2],
            "has_gauss": np.random.get_state()[3],
            "cached_gaussian": np.random.get_state()[4],
        },
    }
    epoch_label = str(epoch)
    Path(options.store).mkdir(parents=True, exist_ok=True)
    checkpoint_path = (
        str(Path(options.store) / "gazenet")
        + "_"
        + options.dataset
        + "_"
        + options.exp
        + "_"
        + epoch_label
        + "epoch.pth.tar"
    )
    temporary_path = str(checkpoint_path) + ".tmp"
    torch.save(checkpoint_state, temporary_path)
    os.replace(temporary_path, checkpoint_path)
    shutil.copyfile(checkpoint_path, Path(options.store) / "last.pth.tar")


def resumer(
    options: Options,
    gaze_model: gaze_network.Net,
    optimizer: Optimizer,
) -> tuple[gaze_network.Net, Optimizer, Options, float] | None:
    """Restore a tensor checkpoint on the model device and update options.start_epoch.

    Returns the restored objects and best distance, or None if the path is absent.
    The caller is responsible for checking that options.resume exists before calling.
    """

    if os.path.isfile(options.resume):
        print("=> loading checkpoint '{}'".format(options.resume))
        checkpoint = torch.load(
            options.resume, map_location=next(gaze_model.parameters()).device, weights_only=True
        )
        options.start_epoch = checkpoint["epoch"]
        best_validation_distance = checkpoint["best_err"]
        gaze_model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
            if torch.cuda.is_available() and checkpoint["cuda_rng_states"]:
                torch.cuda.set_rng_state_all(
                    [state.cpu() for state in checkpoint["cuda_rng_states"]]
                )
            random.setstate(checkpoint["python_rng_state"])
            numpy_state = checkpoint["numpy_rng_state"]
            np.random.set_state(
                (
                    numpy_state["name"],
                    numpy_state["keys"].cpu().numpy().astype("uint32"),
                    numpy_state["position"],
                    numpy_state["has_gauss"],
                    numpy_state["cached_gaussian"],
                )
            )
        print(
            "=> loaded checkpoint '{}' (epoch {})".format(options.resume, checkpoint["epoch"])
        )

        return gaze_model, optimizer, options, best_validation_distance


def load_model(options: Options) -> gaze_network.Net:
    """Construct a gaze network and move it to the explicitly resolved execution device."""
    gaze_model = gaze_network.Net(options)

    return gaze_model.to(resolve_device(getattr(options, "device", "auto")))
