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
import scipy.io as sio
import torch

parser = opts.myargparser()

global opt
opt = parser.parse_args()

dataloader = ld.GazeFollow(opt)

images, xis, eye_coords, pred_coords, eyes, names = next(iter(dataloader.val_loader))

imagenet_mean = sio.loadmat('imagenet_mean_resize.mat')
imagenet_mean = imagenet_mean['image_mean']
places_mean = sio.loadmat('places_mean_resize.mat')
places_mean = places_mean['image_mean']

for i in range(64):
    name = names[i]

    img = xis[i]
    ey = eyes[i]
    eye = eye_coords[i].view(1, 169)
    pred = pred_coords[i].view(1, 225)

    ind = pred.max(1)[1]
    step = 1 / 30.0
    y = ((float(ind[0]/ 15)) / 15.0) + step
    x = ((float(ind[0] % 15)) / 15.0) + step

    to_pil = torchvision.transforms.ToPILImage()
    im = to_pil(img)
    eye_np = eyes[i].cpu().numpy()
    print(name)
    print(eye_np)
    print(x * 227, y * 227)
    plt.subplot(131)
    plt.plot([x* 227, eye_np[0]* 227],[y* 227, eye_np[1]* 227])
    plt.imshow(img)
    plt.subplot(132)
    plt.imshow(eye_coords[i].cpu().numpy())
    plt.show()
