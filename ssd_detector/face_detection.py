"""Load S3FD checkpoints, decode face detections, and prepare gaze-model face inputs.

load_detector restores a tensor-only checkpoint on the requested device and
returns an evaluation-mode model. detect accepts a BGR 0..255 image, subtracts
the S3FD mean, evaluates six feature scales, and decodes candidate corner boxes
with face confidence scores before NMS. getFaces accepts RGB images, applies
NMS and confidence filtering, crops faces, and estimates eyes at box centers.

Returned face tensors and one-hot 13x13 eye grids remain on CPU for the caller
to normalize and transfer. Empty detections have explicit empty shapes. This
module performs inference without gradients and has no CLI or model-loading
side effects when imported.
"""

from __future__ import annotations
from type_definitions import FloatArray, ImageArray, PathInput
from .s3fd_network import s3fd
from pathlib import Path

import numpy as np
import torch
from skimage import transform

from .bounding_boxes import decode, nms


def load_detector(model_path: PathInput, device: str | torch.device) -> s3fd:
    """Load a tensor-only S3FD checkpoint on device and return the detector in eval mode."""
    detector = s3fd().to(device)
    checkpoint_state = torch.load(Path(model_path), map_location=device, weights_only=True)
    detector.load_state_dict(checkpoint_state.get("state_dict", checkpoint_state))
    return detector.eval()


@torch.inference_mode()
def detect(detector: torch.nn.Module, bgr_image: ImageArray) -> FloatArray:
    """Detect faces in an H×W×3 BGR image with values in the 0..255 range.

    Returns a float32 (N,5) array of x1,y1,x2,y2,score before NMS, including
    only candidates with probability at least 0.05. No detections yields (0,5).
    """
    device = next(detector.parameters()).device
    detector_input = np.asarray(bgr_image, dtype=np.float32) - np.array(
        [104, 117, 123], dtype=np.float32
    )
    detector_input = (
        torch.from_numpy(detector_input.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
    )
    scale_outputs = detector(detector_input)
    candidate_boxes = []
    for scale_index in range(len(scale_outputs) // 2):
        class_probabilities = scale_outputs[2 * scale_index].softmax(dim=1).cpu()
        box_offsets = scale_outputs[2 * scale_index + 1].cpu()
        _, _, feature_height, feature_width = class_probabilities.shape
        stride = 2 ** (scale_index + 2)
        for feature_row, feature_column in torch.nonzero(
            class_probabilities[0, 1] >= 0.05
        ).tolist():
            face_probability = class_probabilities[0, 1, feature_row, feature_column].item()
            anchor_box = torch.tensor(
                [
                    [
                        stride / 2 + feature_column * stride,
                        stride / 2 + feature_row * stride,
                        stride * 4,
                        stride * 4,
                    ]
                ]
            )
            decoded_box = decode(
                box_offsets[0, :, feature_row, feature_column].reshape(1, 4), anchor_box, [0.1, 0.2]
            )[0]
            candidate_boxes.append([*decoded_box.tolist(), face_probability])
    return np.asarray(candidate_boxes, dtype=np.float32).reshape(-1, 5)


def getFaces(
    rgb_image: ImageArray,
    detector: torch.nn.Module | None = None,
    model_path: PathInput = "s3fd_convert.pth",
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[float, float]]]:
    """Extract detected RGB face crops and eye-location inputs for the gaze model.

    Args:
        rgb_image: H×W×3 RGB uint8 image.
        detector: Existing detector; otherwise model_path is loaded on device.
        model_path: S3FD tensor-state checkpoint used when net is absent.
        device: Device for a newly loaded detector.

    Returns:
        CPU float face tensors (N,3,227,227) in [0,1], one-hot eye grids
        (N,13,13), and normalized (x,y) eye locations estimated at box centers.
        Face tensors have not yet received ImageNet normalization.
    """
    if detector is None:
        detector = load_detector(model_path, device)
    face_detections = detect(detector, np.asarray(rgb_image)[..., ::-1])
    if len(face_detections):
        face_detections = face_detections[nms(face_detections, 0.3)]
    image_height, image_width = rgb_image.shape[:2]
    face_images, eye_position_grids, eye_positions = [], [], []
    for box_left, box_top, box_right, box_bottom, face_probability in face_detections:
        if face_probability < 0.5:
            continue
        crop_left, crop_top = max(0, int(box_left)), max(0, int(box_top))
        crop_right, crop_bottom = min(image_width, int(box_right)), min(
            image_height, int(box_bottom)
        )
        if crop_right <= crop_left or crop_bottom <= crop_top:
            continue
        face_image = transform.resize(
            rgb_image[crop_top:crop_bottom, crop_left:crop_right], (227, 227)
        )
        face_images.append(torch.from_numpy(face_image.transpose(2, 0, 1).copy()).float())
        eye_x, eye_y = np.clip(
            [
                (box_left + box_right) / (2 * image_width),
                (box_top + box_bottom) / (2 * image_height),
            ],
            0,
            np.nextafter(1.0, 0),
        )
        eye_position_grid = torch.zeros(13, 13)
        eye_position_grid[int(eye_y * 13), int(eye_x * 13)] = 1
        eye_position_grids.append(eye_position_grid)
        eye_positions.append((eye_x, eye_y))
    if not face_images:
        return torch.empty(0, 3, 227, 227), torch.empty(0, 13, 13), []
    return torch.stack(face_images), torch.stack(eye_position_grids), eye_positions
