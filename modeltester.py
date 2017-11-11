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
import matplotlib.pyplot as plt
import matplotlib

parser = opts.myargparser()
opt = parser.parse_args()

checkpoint = torch.load('../adamodels/gazenet_gazefollow_best.pth.tar')
print("Loading pretrained model: ")
start_epoch = checkpoint['epoch']
best_err = checkpoint['best_prec1']

model = gazenet.Net(opt).cuda()
model.load_state_dict(checkpoint['state_dict'])

dataloader = ld.GazeFollow(opt)

images, xis, eye_coords, pred_coords, eyes, names = next(iter(dataloader.val_loader))

images, xis, eye_coords, pred_coords, eyes = Variable(images.cuda()), Variable(xis.cuda()), Variable(eye_coords.cuda()), Variable(pred_coords.cuda()), eyes

outputs = model(images, xis, eye_coords)

for i in range(64):
    name = names[i]
    img = images[i].data.cpu()
    pred = outputs[i].data.view(1, 225)

    ind = pred.max(1)[1]
    step = 1 / 30.0
    y = ((float(ind[0]/ 15)) / 15.0) + step
    x = ((float(ind[0] % 15)) / 15.0) + step

    to_pil = torchvision.transforms.ToPILImage()
    im = to_pil(img)
    eye_np = eyes[i].cpu().numpy()

    print(name)
    print(eye_np * 227)
    print(x * 227, y * 227)
    print(pred_coords[i] * 227)


    plt.subplot(131)
    plt.plot([x* 227, eye_np[0]* 227],[y* 227, eye_np[1]* 227])
    plt.imshow(im)
    plt.subplot(132)
    plt.imshow(eye_coords[i].data.cpu().numpy())
    plt.show()
