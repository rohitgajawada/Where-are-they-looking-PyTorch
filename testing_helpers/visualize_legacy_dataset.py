"""Preserve the historical manual plotter for the experimental dataset loader.

The script reverses normalization, displays scene/head inputs, and plots the five
shifted target grids for a small selection of examples. Its old experimental
loader getdata_exp is not included in the repository; several sample layouts
and image paths are historical rather than supported by the current loader.

It parses CLI arguments and executes plotting at import time. This file is
retained as a reference for earlier debugging, not advertised as a working
modern entry point. The standard training and visualization CLIs use the
maintained gazefollow_dataset module instead.
"""

from __future__ import annotations
import os
import matplotlib as mpl

mpl.use("Agg")
print("MPL done")
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import command_line_options
import training_engine

import models.__init__ as model_utils
import getdata_exp as experimental_gaze_data
import torchvision
from torchvision import transforms

# import scipy.io as sio
import torch

parser = command_line_options.optionargparser()

global options
options = parser.parse_args()
dataset_loaders = experimental_gaze_data.GazeFollow(options)

undo_standard_deviation = transforms.Compose(
    [transforms.Normalize([0, 0, 0], [1 / (0.229), 1 / (0.224), 1 / (0.225)])]
)
undo_mean = transforms.Compose([transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1])])
sample_iterator = iter(dataset_loaders.train_gaze)
for sample_index in range(8):
    #    print(images, bbox, eye_coords, shifted_grids, eyes2, idx, eyes_bbox, gaze, gaze2)
    (
        scene_image,
        head_image,
        eye_position_grid,
        shifted_grid_targets,
        eye_position,
        annotation_index,
        head_eye_position,
        gaze_annotations,
        gaze_point_pixels,
    ) = next(sample_iterator)
    print(scene_image.shape)
    to_pil = torchvision.transforms.ToPILImage()

    scene_image = undo_standard_deviation(scene_image)
    scene_image = undo_mean(scene_image)
    scene_image = scene_image.numpy()
    scene_image = scene_image.transpose((2, 0, 1))
    scene_image = scene_image.transpose((2, 0, 1))
    print(scene_image.shape)

    head_image = undo_standard_deviation(head_image)
    head_image = undo_mean(head_image)
    head_image = to_pil(head_image)
    eye_position = eye_position * 227

    shifted_grid_targets = shifted_grid_targets.numpy()
    # eyes2 = eyes2.numpy()
    # eyes2 = eyes2 * 227
    # gaze = gaze
    gaze_point_pixels = [int(gaze_point_pixels[0] * 227), int(gaze_point_pixels[1] * 227)]
    print("Eye: ", eye_position, " Gaze: ", gaze_point_pixels)
    # print(image)
    # print(bbox)
    figure = plt.figure()
    axes = figure.add_subplot(331)
    axes.imshow(scene_image)
    axes.plot(
        [gaze_point_pixels[0], int(eye_position[0])], [gaze_point_pixels[1], int(eye_position[1])]
    )
    axes = figure.add_subplot(332)
    axes.imshow(head_image)

    axes = figure.add_subplot(333)
    axes.imshow(shifted_grid_targets[0])

    axes = figure.add_subplot(334)
    axes.imshow(shifted_grid_targets[1])

    axes = figure.add_subplot(335)
    axes.imshow(shifted_grid_targets[2])

    axes = figure.add_subplot(336)
    axes.imshow(shifted_grid_targets[3])

    axes = figure.add_subplot(337)
    axes.imshow(shifted_grid_targets[4])

    figure.savefig("outputs/" + "test" + str(sample_index) + ".jpeg")
    plt.close()
    figure.clf()
    axes.cla()

    """
    name = names[i]
    img = untr(images[i])
    img2 = untr(xis[i])

    img = untr2(img)
    img2 = untr2(img2)

    ey = eyes2[i]
    eye = eye_coords[i].view(1, 169)
    pred = pred_coords[i].view(1, 169)

    ind = pred.max(1)[1]
    step = 1 / 26.0
    y = ((float(ind[0]/ 13)) / 13.0) + step
    x = ((float(ind[0] % 13)) / 13.0) + step

    to_pil = torchvision.transforms.ToPILImage()
    im = to_pil(img)
    im2 = to_pil(img2)
    eye_np = eyes[i].cpu().numpy()
    eye_np2 = eyes2[i].cpu().numpy()
    print(name)
    print(eye_np)
    print(x * 227, y * 227)

    plt.subplot(131)
    plt.plot([x* 227, eye_np[0]* 227],[y* 227, eye_np[1]* 227])
    plt.imshow(im)
    plt.subplot(133)
    plt.plot([0, eye_np2[0]*227], [0, eye_np2[1]*227])
    plt.imshow(im2)
    plt.subplot(132)
    plt.imshow(eye_coords[i].cpu().numpy())
    plt.savefig("outputs/" + "test" + str(i) + ".jpeg")
    """
