import os
import torch.backends.cudnn as cudnn
import opts
import train
import utils
import models.__init__ as init
import datasets.getdata as ld
import matplotlib
import matplotlib.pyplot as plt
import torchvision

parser = opts.myargparser()

global opt
opt = parser.parse_args()

dataloader = ld.GazeFollow(opt)

images, xis, eye_coords, pred_coords = next(iter(dataloader.val_loader))

# utils.visualize_gaze(xhs, xps, labels)

img = images[0]
eye = eye_coords[0].view(1, 169)
pred = pred_coords[0].view(1, 225)

ind = pred.max(1)[1]
step = 1 / 30.0
y = ((float(ind[0]/ 15)) / 15.0) + step
x = ((float(ind[0] % 15)) / 15.0) + step

e = eye.max(1)[1]
ey = ((float(e[0]/ 13)) / 15.0) + step
ex = ((float(e[0] % 13)) / 15.0) + step

print(e, ind)

to_pil = torchvision.transforms.ToPILImage()
im = to_pil(img)
print(x * 227, y * 227, ex * 227, ey * 227)
plt.plot([x, ex],[y, ey])
plt.imshow(im)
plt.show()
