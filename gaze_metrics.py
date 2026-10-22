"""Measure predicted gaze locations against one or more human fixation annotations.

_distances accepts one flattened 227x227 heatmap peak index per example and a
padded array of normalized (x,y) annotation pairs. It decodes predictions, masks
negative/non-finite pairs, and returns per-example Euclidean distances to the
mean annotation location and to the nearest individual annotation. Public
wrappers average one of those metrics over the batch and return a Python float.

Distance to the mean location is different from averaging individual distances.
The calculation runs on detached CPU tensors and does not contribute to the
training loss. Angular error follows eye-to-mean gaze directions. Fixation AUC
uses an explicit 64x64 raster protocol; exact 2015 evaluator parity is unverified.
"""

from __future__ import annotations
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
) -> float:
    """Compute explicitly configured fixation AUC; exact 2015 evaluator parity is unverified."""
    return fixation_auc(
        predicted_heatmaps, gaze_annotations, getattr(options, "auc_resolution", 64)
    )


def angular_error(
    predicted_flat_indices: torch.Tensor,
    gaze_annotations: torch.Tensor,
    eye_positions: torch.Tensor,
) -> float:
    """Return mean angle in degrees from eye-to-prediction versus eye-to-mean fixation.

    Zero-length vectors have undefined angle and are excluded from the mean.
    NaN is returned when every vector is undefined, rather than inventing an angle.
    """
    indices = torch.as_tensor(predicted_flat_indices).detach().cpu().long()
    points = (
        torch.as_tensor(gaze_annotations).detach().cpu().float().reshape(len(indices), -1, 2)
    )
    valid = (points >= 0).all(-1) & torch.isfinite(points).all(-1)
    means = points.masked_fill(~valid[..., None], 0).sum(1) / valid.sum(1, keepdim=True)
    predicted = torch.stack((indices % 227, indices // 227), dim=-1).float() / 227
    eyes = torch.as_tensor(eye_positions).detach().cpu().float()
    predicted_vectors, true_vectors = predicted - eyes, means - eyes
    products = predicted_vectors.norm(dim=-1) * true_vectors.norm(dim=-1)
    defined = products > 1e-12
    if not defined.any():
        return float("nan")
    cosines = (predicted_vectors * true_vectors).sum(-1)[defined] / products[defined]
    return torch.rad2deg(torch.acos(cosines.clamp(-1, 1))).mean().item()


def fixation_auc(
    predicted_heatmaps: torch.Tensor, gaze_annotations: torch.Tensor, resolution: int = 64
) -> float:
    """Average tie-correct ROC AUC using all annotated fixation pixels on a fixed grid.

    Each image contributes equally. Heatmaps are bilinearly sampled at resolution
    and annotations become a multi-hot binary map. This explicit auc_64 protocol
    has not been verified against the unavailable original 2015 evaluator.
    """
    import numpy as np

    heatmaps = torch.nn.functional.interpolate(
        predicted_heatmaps.detach().cpu().float()[:, None],
        size=(resolution, resolution),
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    annotation_batches = (
        torch.as_tensor(gaze_annotations).detach().cpu().float().reshape(len(heatmaps), -1, 2)
    )
    values: list[float] = []
    for heatmap, points in zip(heatmaps, annotation_batches):
        valid = (points >= 0).all(-1) & torch.isfinite(points).all(-1)
        pixels = (points[valid] * resolution).floor().clamp(0, resolution - 1).long()
        labels = np.zeros(resolution * resolution, dtype=bool)
        labels[(pixels[:, 1] * resolution + pixels[:, 0]).numpy()] = True
        scores = heatmap.reshape(-1).numpy()
        order = np.argsort(scores, kind="stable")
        sorted_scores = scores[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
        ends = np.r_[starts[1:], len(scores)]
        ranks = np.empty(len(scores), dtype=np.float64)
        ranks[order] = np.repeat((starts + ends - 1) / 2 + 1, ends - starts)
        positive_count = labels.sum()
        if positive_count == 0 or positive_count == len(scores):
            raise ValueError("AUC needs both positive and negative pixels")
        values.append(
            float(
                (ranks[labels].sum() - positive_count * (positive_count + 1) / 2)
                / (positive_count * (len(scores) - positive_count))
            )
        )
    return float(np.mean(values))
