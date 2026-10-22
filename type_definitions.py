"""Shared type definitions for dataset samples, arrays, options, and helper interfaces.

This module is the common vocabulary used by the dataset loader, augmentations,
models, trainer, and inference scripts. It defines numeric NumPy array aliases,
the eight-field GazeSample tuple, collated GazeBatch values, the TransformSample
dictionary, supported option/path objects, and the smoke-test result schema.
Protocols describe the small interfaces required of scalar loggers, batch
loaders, and the archival Caffe model without importing those applications.

These are static typing contracts, not runtime validators. torch.Tensor shapes,
coordinate systems, image ranges, and padding conventions are documented at
function boundaries. Heterogeneous MATLAB cell arrays are represented only at
the SciPy loading boundary. Importing this module performs no model inference.
"""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Iterator, Mapping, Sequence
from os import PathLike
from types import SimpleNamespace
from typing import Any, Protocol, TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray
from torch import Tensor

PathInput: TypeAlias = str | PathLike[str]
Options: TypeAlias = Namespace | SimpleNamespace
ImageArray: TypeAlias = NDArray[np.uint8] | NDArray[np.float32] | NDArray[np.float64]
FloatArray: TypeAlias = NDArray[np.float32] | NDArray[np.float64]
MatlabAnnotations: TypeAlias = Mapping[str, NDArray[Any]]
GazeSample: TypeAlias = tuple[
    Tensor, Tensor, Tensor, Tensor, FloatArray, int, FloatArray, FloatArray
]
GazeBatch: TypeAlias = Sequence[Tensor]


class TransformSample(TypedDict):
    """RGB images and normalized eye/gaze coordinates passed to augmentation.

    ``img`` and ``bbox`` are H×W×3 scene and head images. ``eyes`` and ``gaze``
    are scene coordinates; ``eyes_bbox`` is relative to the head crop.
    """

    img: ImageArray
    bbox: ImageArray
    eyes: FloatArray
    eyes_bbox: FloatArray
    gaze: FloatArray


class ScalarWriter(Protocol):
    """Minimal TensorBoard-compatible scalar logging interface."""

    def add_scalar(self, tag: str, scalar_value: float, global_step: int | None = None) -> None:
        """Record a named scalar at an optional training step."""
        ...


class SmokeResult(TypedDict):
    """Versions, device, loss, and completed checks from the model smoke test."""

    torch: str
    cuda_build: str | None
    device: str
    batch_size: int
    loss: float
    gpu: str | None
    checks: list[str]


class CaffeBlob(Protocol):
    """Caffe blob operations used by the archival WIDER evaluator."""

    data: NDArray[np.float32]

    def reshape(self, *shape: int) -> None:
        """Resize the blob to the supplied dimensions."""
        ...


class CaffeNet(Protocol):
    """Small Caffe network interface, without importing Caffe for annotations."""

    blobs: Mapping[str, CaffeBlob]

    def forward(self) -> Mapping[str, FloatArray]:
        """Evaluate the loaded detector and return named output arrays."""
        ...


class GazeLoader(Protocol):
    """Sized iterable of collated gaze batches used by epoch loops."""

    def __iter__(self) -> Iterator[GazeBatch]:
        """Yield collated scene/head/label tensors for each batch."""
        ...

    def __len__(self) -> int:
        """Return the number of batches in one epoch."""
        ...
