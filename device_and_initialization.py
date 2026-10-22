"""Select execution devices and initialize reproducible runs and scene backbones.

resolve_device handles automatic selection and explicit CPU/CUDA requests;
explicit CUDA fails rather than silently selecting CPU when no GPU is exposed.
seed_everything initializes Python, NumPy, and PyTorch RNGs before constructing
models, while seed_worker initializes the RNGs used inside DataLoader workers.

load_places_features constructs the historical AlexNet scene backbone and loads
matching Places parameters on CPU. It accepts tensor state dictionaries and
requires an explicit trust option for legacy pickled model objects. It checks
backbone keys and shapes strictly. This module does not train models, download
GazeFollow, install drivers, or make a hidden GPU visible to a sandbox.
"""

from __future__ import annotations
from type_definitions import Options
import random
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import alexnet


def resolve_device(requested_device: str = "auto") -> torch.device:
    """Resolve auto/CPU/CUDA selection; raise RuntimeError for unavailable explicit CUDA."""
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested_device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable. Check nvidia-smi and GPU device access; use --device cpu for CPU diagnostics."
            )
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {device.index} does not exist")
    return device


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch before model construction; configure deterministic cuDNN."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python from the PyTorch seed assigned to this DataLoader worker."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def load_places_features(options: Options) -> nn.Sequential:
    """Load the Places AlexNet convolutional backbone on CPU.

    Args:
        options: Options containing placesmodelpath and optional random_init and
            trust_legacy_checkpoint flags. Random initialization skips disk loading.

    Returns:
        AlexNet features through conv5, mapping B×3×227×227 to B×256×13×13.

    Raises:
        FileNotFoundError: The requested weight file does not exist.
        RuntimeError: Deserialization fails or backbone keys/shapes do not match.
        ValueError: The checkpoint does not contain a supported state dictionary.

    Legacy pickled modules execute Python during loading and require explicit trust.
    """
    if getattr(options, "backbone", "torchvision") == "caffe":
        from models.caffe_alexnet_backbone import CaffeAlexNetFeatures

        return CaffeAlexNetFeatures(options, scene=True)
    # Preserve the torchvision variant for controlled comparisons.
    scene_backbone = alexnet(weights=None).features[:-2]
    if getattr(options, "random_init", False):
        return scene_backbone
    checkpoint_path = Path(options.placesmodelpath)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Places weights not found: {checkpoint_path}. Supply --placesmodelpath or use --random-init for diagnostics."
        )
    allow_legacy_pickle = getattr(options, "trust_legacy_checkpoint", False)
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=not allow_legacy_pickle
        )
    except Exception as load_error:
        raise RuntimeError(
            f"Cannot load Places weights {checkpoint_path}. Prefer a tensor state_dict. For a trusted old pickled model only, pass --trust-legacy-checkpoint."
        ) from load_error
    if isinstance(checkpoint, nn.Module):
        checkpoint = checkpoint.state_dict()
    if isinstance(checkpoint, Mapping):
        checkpoint = checkpoint.get("state_dict", checkpoint)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(
            "Places checkpoint must contain an AlexNet state_dict or a trusted AlexNet module"
        )
    expected_parameters = scene_backbone.state_dict()
    backbone_state = {}
    for parameter_name, parameter_tensor in checkpoint.items():
        parameter_name = parameter_name.removeprefix("module.").removeprefix("features.")
        if parameter_name in expected_parameters:
            backbone_state[parameter_name] = parameter_tensor
    scene_backbone.load_state_dict(backbone_state, strict=True)
    return scene_backbone
