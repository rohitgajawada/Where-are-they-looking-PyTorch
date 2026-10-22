"""Encode, decode, compare, and suppress face-detection bounding boxes.

bboxlog/bboxloginv transform scalar corner boxes and anchor-relative center/log-
size offsets. encode/decode implement the batched PyTorch equivalents with
center/size variance factors. nms greedily retains high-scoring detections using
the historical inclusive-pixel (+1) area convention. IOU provides a scalar
fallback when the optional compiled iou extension cannot be imported.

Callers must respect each helper's documented corner-versus-center box layout.
This module has no checkpoint or dataset dependencies and does not run a face
network; it only transforms supplied arrays and tensors.
"""

from __future__ import print_function
from __future__ import annotations
from collections.abc import Sequence
from type_definitions import FloatArray
import math
import numpy as np
import torch

try:
    from iou import IOU
except:
    # IOU cython speedup 10x
    def IOU(
        first_left: float,
        first_top: float,
        first_right: float,
        first_bottom: float,
        second_left: float,
        second_top: float,
        second_right: float,
        second_bottom: float,
    ) -> float:
        """Compute intersection-over-union for two axis-aligned continuous-coordinate boxes."""
        first_area = abs((first_right - first_left) * (first_bottom - first_top))
        second_area = abs((second_right - second_left) * (second_bottom - second_top))
        intersection_left, intersection_top = max(first_left, second_left), max(
            first_top, second_top
        )
        intersection_right, intersection_bottom = min(first_right, second_right), min(
            first_bottom, second_bottom
        )
        intersection_width = intersection_right - intersection_left
        intersection_height = intersection_bottom - intersection_top
        if intersection_width < 0 or intersection_height < 0:
            return 0.0
        else:
            return (
                1.0
                * intersection_width
                * intersection_height
                / (first_area + second_area - intersection_width * intersection_height)
            )


def bboxlog(
    box_left: float,
    box_top: float,
    box_right: float,
    box_bottom: float,
    anchor_center_x: float,
    anchor_center_y: float,
    anchor_width: float,
    anchor_height: float,
) -> tuple[float, float, float, float]:
    """Encode a corner-coordinate box as anchor-relative center offsets and log width/height."""
    box_center_x, box_center_y, box_width, box_height = (
        (box_right + box_left) / 2,
        (box_bottom + box_top) / 2,
        box_right - box_left,
        box_bottom - box_top,
    )
    center_offset_x, center_offset_y = (box_center_x - anchor_center_x) / anchor_width, (
        box_center_y - anchor_center_y
    ) / anchor_height
    log_width_offset, log_height_offset = math.log(box_width / anchor_width), math.log(
        box_height / anchor_height
    )
    return center_offset_x, center_offset_y, log_width_offset, log_height_offset


def bboxloginv(
    center_offset_x: float,
    center_offset_y: float,
    log_width_offset: float,
    log_height_offset: float,
    anchor_center_x: float,
    anchor_center_y: float,
    anchor_width: float,
    anchor_height: float,
) -> tuple[float, float, float, float]:
    """Decode anchor-relative center/log-size offsets into x1,y1,x2,y2 coordinates."""
    box_center_x, box_center_y = (
        center_offset_x * anchor_width + anchor_center_x,
        center_offset_y * anchor_height + anchor_center_y,
    )
    box_width, box_height = (
        math.exp(log_width_offset) * anchor_width,
        math.exp(log_height_offset) * anchor_height,
    )
    box_left, box_right, box_top, box_bottom = (
        box_center_x - box_width / 2,
        box_center_x + box_width / 2,
        box_center_y - box_height / 2,
        box_center_y + box_height / 2,
    )
    return box_left, box_top, box_right, box_bottom


def nms(detections: FloatArray, iou_threshold: float) -> list[np.intp]:
    """Return retained row indices after greedy score-ordered non-maximum suppression.

    Args:
        detections: (N,5) array containing x1,y1,x2,y2,score rows.
        iou_threshold: Maximum allowed IoU before suppressing a lower-scored box.

    The historical inclusive-pixel (+1) area convention is retained.
    """
    if 0 == len(detections):
        return []
    box_lefts, box_tops, box_rights, box_bottoms, scores = (
        detections[:, 0],
        detections[:, 1],
        detections[:, 2],
        detections[:, 3],
        detections[:, 4],
    )
    box_areas = (box_rights - box_lefts + 1) * (box_bottoms - box_tops + 1)
    score_order = scores.argsort()[::-1]

    retained_indices = []
    while score_order.size > 0:
        selected_index = score_order[0]
        retained_indices.append(selected_index)
        intersection_lefts, intersection_tops = np.maximum(
            box_lefts[selected_index], box_lefts[score_order[1:]]
        ), np.maximum(box_tops[selected_index], box_tops[score_order[1:]])
        intersection_rights, intersection_bottoms = np.minimum(
            box_rights[selected_index], box_rights[score_order[1:]]
        ), np.minimum(box_bottoms[selected_index], box_bottoms[score_order[1:]])

        intersection_widths, intersection_heights = np.maximum(
            0.0, intersection_rights - intersection_lefts + 1
        ), np.maximum(0.0, intersection_bottoms - intersection_tops + 1)
        overlap_ratios = (
            intersection_widths
            * intersection_heights
            / (
                box_areas[selected_index]
                + box_areas[score_order[1:]]
                - intersection_widths * intersection_heights
            )
        )

        remaining_indices = np.where(overlap_ratios <= iou_threshold)[0]
        score_order = score_order[remaining_indices + 1]

    return retained_indices


def encode(
    matched_boxes: torch.Tensor, anchor_boxes: torch.Tensor, variances: Sequence[float]
) -> torch.Tensor:
    """Encode matched (N,4) corner boxes relative to (N,4) center/size anchors.

    Variances contains center and size scale factors. Returns (N,4) normalized
    center offsets and log-size offsets for the regression heads.
    """

    # dist b/t match center and prior's center
    normalized_center_offsets = (matched_boxes[:, :2] + matched_boxes[:, 2:]) / 2 - anchor_boxes[
        :, :2
    ]
    # encode variance
    normalized_center_offsets /= variances[0] * anchor_boxes[:, 2:]
    # match wh / prior wh
    normalized_size_offsets = (matched_boxes[:, 2:] - matched_boxes[:, :2]) / anchor_boxes[:, 2:]
    normalized_size_offsets = torch.log(normalized_size_offsets) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([normalized_center_offsets, normalized_size_offsets], 1)  # [num_priors,4]


def decode(
    box_offsets: torch.Tensor, anchor_boxes: torch.Tensor, variances: Sequence[float]
) -> torch.Tensor:
    """Decode (N,4) regression offsets and center/size anchors into corner decoded_boxes.

    Variances contains center and size scale factors. Returns a tensor of
    x1,y1,x2,y2 rows with shape (N,4).
    """

    decoded_boxes = torch.cat(
        (
            anchor_boxes[:, :2] + box_offsets[:, :2] * variances[0] * anchor_boxes[:, 2:],
            anchor_boxes[:, 2:] * torch.exp(box_offsets[:, 2:] * variances[1]),
        ),
        1,
    )
    decoded_boxes[:, :2] -= decoded_boxes[:, 2:] / 2
    decoded_boxes[:, 2:] += decoded_boxes[:, :2]
    return decoded_boxes
