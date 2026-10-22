"""Preserve the original Caffe-based WIDER evaluation script for historical reference.

The script prepares Caffe detector inputs, performs multi-scale and flipped
inference, combines overlapping detections with box voting, and writes WIDER
prediction files. It relies on Caffe, OpenCV, hardcoded external paths, and
historical MATLAB annotation conventions rather than the modern PyTorch setup.

This archival module changes the working directory and imports Caffe at module
load time; it is not safe to import as an ordinary application helper. Its old
runtime assumptions, including Python-2-era constructs, are retained. Use the
sibling PyTorch evaluator for the supported current detector workflow.
"""

from __future__ import annotations
from typing import TextIO
from type_definitions import CaffeNet, FloatArray, ImageArray
import numpy as np
import cv2
import scipy.io as scipy_io


# Make sure that caffe is on the python path:
caffe_root = "../../"  # this file is expected to be in {caffe_root}/sfd_test_code
import os

os.chdir(caffe_root)
import sys

sys.path.insert(0, "python")
import caffe


def detect_face(detector: CaffeNet, image: ImageArray, resize_scale: float) -> FloatArray:
    """Run the archival Caffe detector on a resized image and restore original box coordinates."""
    if resize_scale != 1:
        image = cv2.resize(
            image, None, None, fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_LINEAR
        )

    detector.blobs["data"].reshape(1, 3, image.shape[0], image.shape[1])
    transformer = caffe.io.Transformer({"data": detector.blobs["data"].data.shape})
    transformer.set_transpose("data", (2, 0, 1))
    transformer.set_mean("data", np.array([104, 117, 123]))
    transformer.set_raw_scale("data", 255)
    transformer.set_channel_swap("data", (2, 1, 0))
    transformed_image = transformer.preprocess("data", image)
    detector.blobs["data"].data[...] = transformed_image

    detections = detector.forward()["detection_out"]
    confidence_scores = detections[0, 0, :, 2]
    box_lefts = image.shape[1] * detections[0, 0, :, 3] / resize_scale
    box_tops = image.shape[0] * detections[0, 0, :, 4] / resize_scale
    box_rights = image.shape[1] * detections[0, 0, :, 5] / resize_scale
    box_bottoms = image.shape[0] * detections[0, 0, :, 6] / resize_scale
    detections = np.column_stack((box_lefts, box_tops, box_rights, box_bottoms, confidence_scores))

    retained_indices = np.where(detections[:, 4] >= 0)[0]
    detections = detections[retained_indices, :]
    return detections


def multi_scale_test(
    detector: CaffeNet,
    image: ImageArray,
    maximum_image_scale: float,
) -> tuple[FloatArray, FloatArray]:
    """Return small-scale and accumulated enlarged-scale detections from the Caffe model."""
    # shrink detecting and shrink only detect big face
    small_image_scale = 0.5 if maximum_image_scale >= 0.75 else 0.5 * maximum_image_scale
    small_scale_detections = detect_face(detector, image, small_image_scale)
    retained_indices = np.where(
        np.maximum(
            small_scale_detections[:, 2] - small_scale_detections[:, 0] + 1,
            small_scale_detections[:, 3] - small_scale_detections[:, 1] + 1,
        )
        > 30
    )[0]
    small_scale_detections = small_scale_detections[retained_indices, :]

    # enlarge one times
    large_image_scale = (
        min(2, maximum_image_scale)
        if maximum_image_scale > 1
        else (small_image_scale + maximum_image_scale) / 2
    )
    large_scale_detections = detect_face(detector, image, large_image_scale)

    # enlarge small iamge x times for small face
    if maximum_image_scale > 2:
        large_image_scale *= 2
        while large_image_scale < maximum_image_scale:
            large_scale_detections = np.row_stack(
                (large_scale_detections, detect_face(detector, image, large_image_scale))
            )
            large_image_scale *= 2
        large_scale_detections = np.row_stack(
            (large_scale_detections, detect_face(detector, image, maximum_image_scale))
        )

    # enlarge only detect small face
    if large_image_scale > 1:
        retained_indices = np.where(
            np.minimum(
                large_scale_detections[:, 2] - large_scale_detections[:, 0] + 1,
                large_scale_detections[:, 3] - large_scale_detections[:, 1] + 1,
            )
            < 100
        )[0]
        large_scale_detections = large_scale_detections[retained_indices, :]
    else:
        retained_indices = np.where(
            np.maximum(
                large_scale_detections[:, 2] - large_scale_detections[:, 0] + 1,
                large_scale_detections[:, 3] - large_scale_detections[:, 1] + 1,
            )
            > 30
        )[0]
        large_scale_detections = large_scale_detections[retained_indices, :]

    return small_scale_detections, large_scale_detections


def flip_test(detector: CaffeNet, image: ImageArray, resize_scale: float) -> FloatArray:
    """Run mirrored-image Caffe detections and map boxes back to original coordinates."""
    flipped_image = cv2.flip(image, 1)
    flipped_detections = detect_face(detector, flipped_image, resize_scale)

    restored_detections = np.zeros(flipped_detections.shape)
    restored_detections[:, 0] = image.shape[1] - flipped_detections[:, 2]
    restored_detections[:, 1] = flipped_detections[:, 1]
    restored_detections[:, 2] = image.shape[1] - flipped_detections[:, 0]
    restored_detections[:, 3] = flipped_detections[:, 3]
    restored_detections[:, 4] = flipped_detections[:, 4]
    return restored_detections


def bbox_vote(detections: FloatArray) -> FloatArray:
    """Merge overlapping score-weighted boxes and retain at most 750 detections.

    This archival routine retains its historical assumptions about nonempty
    merge groups; adding annotations does not modernize its Caffe behavior.
    """
    score_order = detections[:, 4].ravel().argsort()[::-1]
    detections = detections[score_order, :]
    while detections.shape[0] > 0:
        # IOU
        box_areas = (detections[:, 2] - detections[:, 0] + 1) * (
            detections[:, 3] - detections[:, 1] + 1
        )
        intersection_lefts = np.maximum(detections[0, 0], detections[:, 0])
        intersection_tops = np.maximum(detections[0, 1], detections[:, 1])
        intersection_rights = np.minimum(detections[0, 2], detections[:, 2])
        intersection_bottoms = np.minimum(detections[0, 3], detections[:, 3])
        intersection_widths = np.maximum(0.0, intersection_rights - intersection_lefts + 1)
        intersection_heights = np.maximum(0.0, intersection_bottoms - intersection_tops + 1)
        intersection_areas = intersection_widths * intersection_heights
        overlap_ratios = intersection_areas / (box_areas[0] + box_areas[:] - intersection_areas)

        # get needed merge det and delete these det
        merge_indices = np.where(overlap_ratios >= 0.3)[0]
        overlapping_detections = detections[merge_indices, :]
        detections = np.delete(detections, merge_indices, 0)

        if merge_indices.shape[0] <= 1:
            continue
        overlapping_detections[:, 0:4] = overlapping_detections[:, 0:4] * np.tile(
            overlapping_detections[:, -1:], (1, 4)
        )
        max_score = np.max(overlapping_detections[:, 4])
        weighted_detection = np.zeros((1, 5))
        weighted_detection[:, 0:4] = np.sum(overlapping_detections[:, 0:4], axis=0) / np.sum(
            overlapping_detections[:, -1:]
        )
        weighted_detection[:, 4] = max_score
        try:
            merged_detections = np.row_stack((merged_detections, weighted_detection))
        except:
            merged_detections = weighted_detection

    merged_detections = merged_detections[0:750, :]
    return merged_detections


def write_to_txt(output_file: TextIO, detections: FloatArray) -> None:
    """Write WIDER detection rows using the script's current global event and image name."""
    output_file.write("{:s}\n".format(event[0][0].encode("utf-8") + "/" + image_name + ".jpg"))
    output_file.write("{:d}\n".format(detections.shape[0]))
    for detection_index in xrange(detections.shape[0]):
        box_left = detections[detection_index][0]
        box_top = detections[detection_index][1]
        box_right = detections[detection_index][2]
        box_bottom = detections[detection_index][3]
        score = detections[detection_index][4]
        output_file.write(
            "{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n".format(
                box_left, box_top, (box_right - box_left + 1), (box_bottom - box_top + 1), score
            )
        )


if __name__ == "__main__":

    caffe.set_device(0)
    caffe.set_mode_gpu()
    model_def = "models/VGGNet/WIDER_FACE/SFD/deploy.prototxt"
    model_weights = "models/VGGNet/WIDER_FACE/SFD/SFD.caffemodel"
    detector = caffe.Net(model_def, model_weights, caffe.TEST)

    subset = "val"  # val or test
    if subset is "val":
        wider_face = scipy_io.loadmat("./sfd_test_code/WIDER_FACE/wider_face_val.mat")  # Val set
    else:
        wider_face = scipy_io.loadmat("./sfd_test_code/WIDER_FACE/wider_face_test.mat")  # Test set
    event_list = wider_face["event_list"]
    file_list = wider_face["file_list"]
    del wider_face

    # Path = 'Path to the images of WIDER FACE'
    image_directory = "/home/carwin/projects/dataset/face/wider/wider/WIDER_val/"
    save_path = "./sfd_test_code/WIDER_FACE/eval_tools_old-version/sfd" + "_" + subset + "/"
    for index, event in enumerate(event_list):
        image_entries = file_list[index][0]
        if not os.path.exists(save_path + event[0][0].encode("utf-8")):
            os.makedirs(save_path + event[0][0].encode("utf-8"))

        for image_index, image_entry in enumerate(image_entries):
            image_name = image_entry[0][0].encode("utf-8")
            image_path = image_directory + image_name[:] + ".jpg"
            image = caffe.io.load_image(image_path)

            maximum_image_scale = (
                0x7FFFFFFF / 577.0 / (image.shape[0] * image.shape[1])
            ) ** 0.5  # the max size of input image for caffe
            maximum_image_scale = (
                0x7FFFFFFF / 1024.0 / (image.shape[0] * image.shape[1])
            ) ** 0.5  # the max size of input image for caffe
            resize_scale = maximum_image_scale if maximum_image_scale < 1 else 1

            original_detections = detect_face(detector, image, resize_scale)  # origin test
            flipped_detections = flip_test(detector, image, resize_scale)  # flip test
            [small_scale_detections, large_scale_detections] = multi_scale_test(
                detector, image, maximum_image_scale
            )  # multi-scale test

            # merge all test results via bounding box voting
            detections = np.row_stack(
                (
                    original_detections,
                    flipped_detections,
                    small_scale_detections,
                    large_scale_detections,
                )
            )
            merged_detections = bbox_vote(detections)

            output_file = open(
                save_path + event[0][0].encode("utf-8") + "/" + image_name + ".txt", "w"
            )
            write_to_txt(output_file, merged_detections)
            print("event:%d num:%d" % (index + 1, image_index + 1))
