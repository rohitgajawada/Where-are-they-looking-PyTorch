"""Run multi-scale S3FD evaluation on images stored in a WIDER validation ZIP.

The CLI loads event/image annotations and detector weights, reads images without
extracting the ZIP, and combines original, mirrored, reduced, and optionally
enlarged detections. It restores coordinates, applies NMS, retains at most 750
boxes per image, and writes the text files consumed by the WIDER evaluator.

This writes detector predictions; it does not calculate WIDER average precision
or evaluate gaze. Dataset ZIP and S3FD weights must be supplied explicitly.
Run from the repository root with:
python -m ssd_detector.evaluate_wider_faces --help
"""

from __future__ import annotations
import torch
from type_definitions import FloatArray, ImageArray
import argparse
from io import BytesIO
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import scipy.io

from device_and_initialization import resolve_device
from .bounding_boxes import nms
from .face_detection import detect, load_detector


def flip_detect(detector: torch.nn.Module, bgr_image: ImageArray) -> FloatArray:
    """Detect on a horizontally flipped BGR image and map boxes back to original coordinates."""
    boxes = detect(detector, bgr_image[:, ::-1])
    restored_boxes = boxes.copy()
    restored_boxes[:, 0] = bgr_image.shape[1] - boxes[:, 2]
    restored_boxes[:, 2] = bgr_image.shape[1] - boxes[:, 0]
    return restored_boxes


def scale_detect(
    detector: torch.nn.Module,
    bgr_image: ImageArray,
    scale: float = 2.0,
    face_size_threshold: float = 100,
) -> FloatArray:
    """Detect on a resized BGR image, restore coordinates, and filter by original face size."""
    height, width = bgr_image.shape[:2]
    resized_image = np.array(
        Image.fromarray(bgr_image).resize(
            (max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.BILINEAR
        )
    )
    boxes = detect(detector, resized_image)
    boxes[:, :4] /= scale
    box_sizes = boxes[:, 2:4] - boxes[:, :2] + 1
    size_mask = (
        box_sizes.min(1) < face_size_threshold
        if scale > 1
        else box_sizes.max(1) > face_size_threshold
    )
    return boxes[size_mask]


def main() -> None:
    """Run multi-scale WIDER ZIP inference and save one detection text file per image."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="ssd_detector/eval/wider_face_val.mat")
    parser.add_argument("--dataset", required=True, help="WIDER_val.zip")
    parser.add_argument("--model", required=True, help="S3FD tensor state_dict")
    parser.add_argument("--output", default="ssd_detector/eval/sfd_val_pytorch")
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()
    annotations = scipy.io.loadmat(arguments.annotations)
    detector = load_detector(arguments.model, resolve_device(arguments.device))
    with zipfile.ZipFile(arguments.dataset) as archive:
        for event_entry, image_entries in zip(annotations["event_list"], annotations["file_list"]):
            event_name = str(event_entry[0][0])
            event_directory = Path(arguments.output) / event_name
            event_directory.mkdir(parents=True, exist_ok=True)
            for image_entry in image_entries[0]:
                image_name = str(image_entry[0][0])
                with Image.open(
                    BytesIO(archive.read(f"WIDER_val/images/{event_name}/{image_name}.jpg"))
                ) as image:
                    bgr_image = np.array(image.convert("RGB"))[..., ::-1].copy()
                boxes = [
                    detect(detector, bgr_image),
                    flip_detect(detector, bgr_image),
                    scale_detect(detector, bgr_image, 0.5),
                ]
                if bgr_image.shape[0] * bgr_image.shape[1] * 4 <= 3000 * 3000:
                    boxes.append(scale_detect(detector, bgr_image, 2.0))
                boxes = np.concatenate(boxes)
                boxes = boxes[nms(boxes, 0.3)[:750]]
                with (event_directory / f"{image_name}.txt").open("w") as output_file:
                    output_file.write(f"{event_name}/{image_name}.jpg\n{len(boxes)}\n")
                    for box_left, box_top, box_right, box_bottom, face_probability in boxes:
                        output_file.write(
                            f"{box_left:.1f} {box_top:.1f} {box_right-box_left+1:.1f} {box_bottom-box_top+1:.1f} {face_probability:.3f}\n"
                        )
                print(f"{event_name}/{image_name}: {len(boxes)} detections")


if __name__ == "__main__":
    main()
