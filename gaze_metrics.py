"""Measure predicted gaze locations against one or more human fixation annotations.

_distances accepts one flattened 227x227 heatmap peak index per example and a
padded array of normalized (x,y) annotation pairs. It decodes predictions, masks
negative/non-finite pairs, and returns per-example Euclidean distances to the
mean annotation location and to the nearest individual annotation. Public
wrappers average one of those metrics over the batch and return a Python float.

Distance to the mean location is different from averaging individual distances.
The calculation runs on detached CPU tensors and does not contribute to the
training loss. AUCaccuracy remains an explicit unimplemented placeholder; this
module does not claim that the original-paper AUC evaluator has been reproduced.
"""

from __future__ import annotations
from typing import NoReturn
import torch
from type_definitions import Options


def _distances(
    predicted_flat_indices: torch.Tensor,
    gaze_annotations: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate two Euclidean gaze distances for every sample in a batch.

    Args:
        predicted_flat_indices: Shape (B,). One flattened argmax index per 227×227 predicted
            heatmap, not logits or the full heatmap. Index y*227+x denotes (x,y).
        gaze_annotations: Shape (B, 2*A), normally (B, 100). Human annotations are stored
            as [x1,y1,x2,y2,...] in normalized image coordinates. Negative or
            non-finite pairs are ignored; unused pairs are padded with -1.
        batch_size: Batch size B, which must match both inputs.

    Returns:
        Two CPU tensors of shape (B,): distance to the mean valid annotation
        location, and distance to the closest valid annotation. No batch average
        is performed. Predicted pixel coordinates are normalized by 227.

    Raises:
        ValueError: The batch lengths disagree or a sample has no valid gaze points.

    Example:
        A prediction at index 0 is (0,0). Human points (0.6,0) and (0,0.8)
        have mean (0.3,0.4), yielding mean-location distance 0.5 and minimum
        distance 0.6. Averaging the individual distances would give 0.7, which
        is a different metric.
    """
    predicted_flat_indices = (
        torch.as_tensor(predicted_flat_indices).detach().cpu().reshape(-1).long()
    )
    gaze_annotations = torch.as_tensor(gaze_annotations).detach().cpu().float()
    if predicted_flat_indices.numel() != batch_size or gaze_annotations.shape[0] != batch_size:
        raise ValueError("Expected one flattened prediction per sample")
    annotation_points = gaze_annotations.reshape(batch_size, -1, 2)
    valid_annotation_mask = (annotation_points >= 0).all(-1) & torch.isfinite(
        annotation_points
    ).all(-1)
    if not valid_annotation_mask.any(-1).all():
        raise ValueError("Every sample needs at least one valid gaze annotation")
    predicted_points = (
        torch.stack(
            (
                predicted_flat_indices.remainder(227),
                predicted_flat_indices.div(227, rounding_mode="floor"),
            ),
            dim=-1,
        ).float()
        / 227
    )
    masked_annotation_points = annotation_points.masked_fill(
        ~valid_annotation_mask.unsqueeze(-1), 0
    )
    mean_annotation_points = masked_annotation_points.sum(1) / valid_annotation_mask.sum(
        1, keepdim=True
    )
    mean_annotation_distances = torch.linalg.vector_norm(
        predicted_points - mean_annotation_points, dim=-1
    )
    per_annotation_distances = torch.linalg.vector_norm(
        predicted_points[:, None] - masked_annotation_points, dim=-1
    )
    minimum_annotation_distances = (
        per_annotation_distances.masked_fill(~valid_annotation_mask, float("inf")).min(1).values
    )
    return mean_annotation_distances, minimum_annotation_distances


def euclid_dist(
    predicted_flat_indices: torch.Tensor, gaze_annotations: torch.Tensor, batch_size: int
) -> float:
    """Return the batch average of distances to mean human annotation locations.

    Inputs follow _distances: predicted_flat_indices has shape (B,),
    gaze_annotations has shape (B,2*A), and batch_size is B. Padding is ignored
    before averaging human coordinates.
    """
    return _distances(predicted_flat_indices, gaze_annotations, batch_size)[0].mean().item()


def euclid_mindist(
    predicted_flat_indices: torch.Tensor, gaze_annotations: torch.Tensor, batch_size: int
) -> float:
    """Return the batch average of distances to the nearest valid human annotation.

    Inputs follow _distances: predicted_flat_indices has shape (B,),
    gaze_annotations has shape (B,2*A), and batch_size is B. Each sample may contain several valid human gaze points.
    """
    return _distances(predicted_flat_indices, gaze_annotations, batch_size)[1].mean().item()


def AUCaccuracy(
    predicted_heatmaps: torch.Tensor, gaze_annotations: torch.Tensor, options: Options
) -> NoReturn:
    """Raise NotImplementedError; the original-paper AUC protocol is not implemented."""
    raise NotImplementedError("Original-paper AUC protocol has not yet been implemented")
