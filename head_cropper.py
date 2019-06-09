import numpy as np
import matplotlib.image as mpimg
import math
import matplotlib.pyplot as plt
from skimage import io, transform


img = io.imread('../data/test/00000000/00000116.jpg')

e = [0.37636364, 0.45393939]
# coordinates of the eyes

alpha = 0.3
w_x = int(math.floor(alpha * img.shape[1]))
w_y = int(math.floor(alpha * img.shape[0]))

if w_x % 2 == 0:
    w_x = w_x + 1

if w_y % 2 == 0:
    w_y = w_y + 1

im_face = np.ones((w_y, w_x, 3))
im_face[:, :, 0] = 123 * np.ones((w_y, w_x))
im_face[:, :, 1] = 117 * np.ones((w_y, w_x))
im_face[:, :, 2] = 104 * np.ones((w_y, w_x))

center = [math.floor(e[0] * img.shape[1]), math.floor(e[1] * img.shape[0])]
d_x = math.floor((w_x - 1) / 2)
d_y = math.floor((w_y - 1) / 2)

bottom_x = center[0] - d_x - 1
delta_b_x = 0
if bottom_x < 0:
    delta_b_x = 1 - bottom_x
    bottom_x = 0

top_x = center[0] + d_x - 1
delta_t_x = w_x - 1
if top_x > img.shape[1] - 1:
    delta_t_x = w_x - (top_x - img.shape[1] + 1)
    top_x = img.shape[1] - 1

bottom_y = center[1] - d_y - 1
delta_b_y = 0
if bottom_y < 0:
    delta_b_y = 1 - bottom_y
    bottom_y = 0

top_y = center[1] + d_y - 1
delta_t_y = w_y - 1
if top_y > img.shape[0] - 1:
    delta_t_y = w_y - (top_y - img.shape[0] + 1)
    top_y = img.shape[0] - 1

# print delta_b_x, delta_b_y, delta_t_x, delta_t_y
# print top_x, top_y, bottom_x, bottom_y

topx = top_x

x = img[int(bottom_y): int(top_y + 1), int(bottom_x): int(top_x + 1), :]

im_face = im_face[:,:,[1, 0, 2]]
imgplot = plt.imshow(x)
plt.show()
