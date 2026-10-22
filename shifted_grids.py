"""Share the exact grid geometry between target encoding and heatmap decoding.

The five classifiers use center, positive-x, negative-x, negative-y, positive-y
order. Each predicts one of 25 cells; a unit shift moves a boundary by 1/15 of
image width or height, equivalent to 1/3 of a 5x5 cell. Boundary coordinates
are clipped into the outer cells. Decoding evaluates these same cell assignments
at 15x15 pixel centers, averages the five probabilities, and resizes to 227x227.
This convention fixes internal consistency; the unavailable 2015 source has not
established that its undocumented shift convention was numerically identical.
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional

GRID_OFFSETS: tuple[tuple[int, int], ...] = ((0, 0), (1, 0), (-1, 0), (0, -1), (0, 1))
GRID_NAMES: tuple[str, ...] = ("center", "positive_x", "negative_x", "negative_y", "positive_y")


def encode_points(points: torch.Tensor) -> torch.Tensor:
    """Return (...,5) integer cell labels for normalized (...,2) x/y coordinates."""
    points = torch.as_tensor(points)
    offsets = points.new_tensor(GRID_OFFSETS)
    cells = torch.floor(points[..., None, :] * 5 + offsets / 3).clamp(0, 4).long()
    return cells[..., 1] * 5 + cells[..., 0]


def decode_grids(
    grid_log_probabilities: list[torch.Tensor], resolution: int = 227
) -> torch.Tensor:
    """Decode five Bx25 log-probability tensors into Bxresolutionxresolution scores."""
    probabilities = torch.stack(grid_log_probabilities, dim=1).exp()
    pixel_centers = (
        torch.arange(15, device=probabilities.device, dtype=probabilities.dtype) + 0.5
    ) / 15
    row_coordinates, column_coordinates = torch.meshgrid(
        pixel_centers, pixel_centers, indexing="ij"
    )
    labels = encode_points(torch.stack((column_coordinates, row_coordinates), dim=-1))
    indices = labels.reshape(225, 5).T.unsqueeze(0).expand(probabilities.shape[0], -1, -1)
    heatmaps = probabilities.gather(2, indices).mean(1).reshape(-1, 1, 15, 15)
    return functional.interpolate(
        heatmaps, size=(resolution, resolution), mode="bicubic", align_corners=False
    ).squeeze(1)
