"""Display the historical eye-centered head crop for a hardcoded example image.

This manual diagnostic reproduces the original 30-percent crop-window geometry,
boundary calculations, and placeholder padding canvas, then displays the crop
with matplotlib. Eye coordinates and the example image path are embedded in
the script; this is not a general dataset-processing command or a test fixture.

It executes file loading and plotting at import time and retains the old sample
path. It is documented and renamed for clarity, but still needs an appropriate
local image and manual configuration before use.
"""

from __future__ import annotations
import numpy as np
import matplotlib.image as mpimg
import math
import matplotlib.pyplot as plt
from skimage import io, transform


person_image = io.imread("../data/test/00000000/00000116.jpg")

eye_position = [0.37636364, 0.45393939]
# coordinates of the eyes

crop_fraction = 0.3
crop_width = int(math.floor(crop_fraction * person_image.shape[1]))
crop_height = int(math.floor(crop_fraction * person_image.shape[0]))

if crop_width % 2 == 0:
    crop_width = crop_width + 1

if crop_height % 2 == 0:
    crop_height = crop_height + 1

padding_canvas = np.ones((crop_height, crop_width, 3))
padding_canvas[:, :, 0] = 123 * np.ones((crop_height, crop_width))
padding_canvas[:, :, 1] = 117 * np.ones((crop_height, crop_width))
padding_canvas[:, :, 2] = 104 * np.ones((crop_height, crop_width))

eye_center_pixels = [
    math.floor(eye_position[0] * person_image.shape[1]),
    math.floor(eye_position[1] * person_image.shape[0]),
]
crop_half_width = math.floor((crop_width - 1) / 2)
crop_half_height = math.floor((crop_height - 1) / 2)

crop_left = eye_center_pixels[0] - crop_half_width - 1
left_padding = 0
if crop_left < 0:
    left_padding = 1 - crop_left
    crop_left = 0

crop_right = eye_center_pixels[0] + crop_half_width - 1
right_padding_boundary = crop_width - 1
if crop_right > person_image.shape[1] - 1:
    right_padding_boundary = crop_width - (crop_right - person_image.shape[1] + 1)
    crop_right = person_image.shape[1] - 1

crop_top = eye_center_pixels[1] - crop_half_height - 1
top_padding = 0
if crop_top < 0:
    top_padding = 1 - crop_top
    crop_top = 0

crop_bottom = eye_center_pixels[1] + crop_half_height - 1
bottom_padding_boundary = crop_height - 1
if crop_bottom > person_image.shape[0] - 1:
    bottom_padding_boundary = crop_height - (crop_bottom - person_image.shape[0] + 1)
    crop_bottom = person_image.shape[0] - 1

# print delta_b_x, delta_b_y, delta_t_x, delta_t_y
# print top_x, top_y, bottom_x, bottom_y

crop_right_copy = crop_right

head_crop = person_image[
    int(crop_top) : int(crop_bottom + 1), int(crop_left) : int(crop_right + 1), :
]

padding_canvas = padding_canvas[:, :, [1, 0, 2]]
crop_plot = plt.imshow(head_crop)
plt.show()
