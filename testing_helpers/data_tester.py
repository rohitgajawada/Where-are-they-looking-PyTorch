import os
import matplotlib as mpl
mpl.use('Agg')
print("MPL done")
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import opts
import train
import utils
import models.__init__ as init
import getdata_exp as ld
import torchvision
from torchvision import transforms
#import scipy.io as sio
import torch

parser = opts.optionargparser()

global opt
opt = parser.parse_args()
dataloader = ld.GazeFollow(opt)

untr = transforms.Compose([
        transforms.Normalize([0, 0, 0], [1/(0.229), 1/(0.224), 1/(0.225)])])
untr2 = transforms.Compose([
        transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1])])
it = iter(dataloader.train_gaze)
for i in range(8):
#    print(images, bbox, eye_coords, shifted_grids, eyes2, idx, eyes_bbox, gaze, gaze2)
    image, bbox, eye_coords, shifted_grids, eyes2, idx, eyes_bbox, gaze, gaze2 = next(it)
    print(image.shape)
    to_pil = torchvision.transforms.ToPILImage()
    
    image = untr(image)
    image = untr2(image)
    image = image.numpy()
    image = image.transpose((2, 0, 1))
    image = image.transpose((2, 0, 1))
    print(image.shape)

    bbox = untr(bbox)
    bbox = untr2(bbox)
    bbox = to_pil(bbox)
    eyes2 = eyes2 * 227

    shifted_grids = shifted_grids.numpy()
    # eyes2 = eyes2.numpy()
    # eyes2 = eyes2 * 227
    # gaze = gaze
    gaze2 = [int(gaze2[0] * 227), int(gaze2[1] * 227)]
    print("Eye: ",eyes2," Gaze: ", gaze2)
    # print(image)
    # print(bbox)
    fig = plt.figure()
    ax = fig.add_subplot(331)
    ax.imshow(image)
    ax.plot([gaze2[0], int(eyes2[0])], [gaze2[1], int(eyes2[1])])
    ax = fig.add_subplot(332)
    ax.imshow(bbox)
    
    ax = fig.add_subplot(333)
    ax.imshow(shifted_grids[0])

    ax = fig.add_subplot(334)
    ax.imshow(shifted_grids[1])

    ax = fig.add_subplot(335)
    ax.imshow(shifted_grids[2])

    ax = fig.add_subplot(336)
    ax.imshow(shifted_grids[3])

    ax = fig.add_subplot(337)
    ax.imshow(shifted_grids[4])

    fig.savefig("outputs/" + "test" + str(i) + ".jpeg")
    plt.close()
    fig.clf()
    ax.cla()
    
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
