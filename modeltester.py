import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import datasets.getdata as ld
import utils
import os
import opts
from PIL import Image
import models.gazenet as gazenet
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib

parser = opts.myargparser()
opt = parser.parse_args()

checkpoint = torch.load('./savedmodels/adamodels/gazenet_gazefollow_99epoch.pth.tar')
print("Loading pretrained model: ")
start_epoch = checkpoint['epoch']
best_err = checkpoint['best_prec1']

model = gazenet.Net(opt).cuda()
model.load_state_dict(checkpoint['state_dict'])

dataloader = ld.GazeFollow(opt)

images, xis, eye_coords, pred_coords, eyes, names, eyes2 = next(iter(dataloader.val_loader))

images, xis, eye_coords, pred_coords = Variable(images.cuda()), Variable(xis.cuda()), Variable(eye_coords.cuda()), Variable(pred_coords.cuda())

# print(eye_coords)
print(images[0])
exit()
outputs = model(images, xis, eye_coords)

untr = transforms.Compose([
        transforms.Normalize([0, 0, 0], [1/(0.229), 1/(0.224), 1/(0.225)])])
untr2 = transforms.Compose([
        transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1])])

for i in range(64):

    name = names[i]
    img = untr(images[i].data.contiguous().cpu())
    img2 = untr(xis[i].data.contiguous().cpu())

    img = untr2(img)
    img2 = untr2(img2)

    ey = eyes2[i]
    eye = eye_coords[i].view(1, 169)
    pred = outputs[i].data.view(1, 169)

    ind = pred.max(1)[1]
    step = 1 / 26.0
    y = ((float(ind[0]/ 13.0)) / 13.0) + step
    x = ((float(ind[0] % 13.0)) / 13.0) + step

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
    plt.imshow(im2)

    plt.subplot(132)
    plt.imshow(eye_coords[i].data.cpu().numpy())
    plt.show()
