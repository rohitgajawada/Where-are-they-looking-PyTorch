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
import sys
import sfd.tester as facedetect
import scipy.io as sio
from skimage import io, transform
import numpy as np

sys.path.append('./sfd')

parser = opts.myargparser()
opt = parser.parse_args()

checkpoint = torch.load('./gazenet_gazefollow_28epoch.pth.tar')
# checkpoint = torch.load('./savedmodels/adamodels/gazenet_gazefollow_99epoch.pth.tar')

print("Loading our pretrained gazenet model: ")
start_epoch = checkpoint['epoch']
best_err = checkpoint['best_prec1']

model = gazenet.Net(opt).cuda()
model.load_state_dict(checkpoint['state_dict'])

im = io.imread(opt.testpic)

faces, eye_coords, eyes = facedetect.getFaces(im)

normtransform = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

untr = transforms.Compose([
        transforms.Normalize([0, 0, 0], [1/(0.229), 1/(0.224), 1/(0.225)])])
untr2 = transforms.Compose([
        transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1])])

im = np.ascontiguousarray(im)
im = transform.resize(im,(227, 227))
im = im.transpose((2, 0, 1))
img = torch.from_numpy(im).contiguous()
img = img.float()
img = normtransform(img)
img = img.view(1, 3, 227, 227)

num = faces.size(0)
imgs = torch.FloatTensor()
for i in range(num):
    imgs = torch.cat((imgs, img))
    faces[i] = normtransform(faces[i])

xis = faces


imgs = Variable(imgs.cuda())
xis = Variable(faces.cuda())
eye_coords = Variable(eye_coords.cuda())
eye_coords = eye_coords.view(-1, 13, 13)

print(imgs)
print(xis)
print(eye_coords)

outputs = model(imgs, xis, eye_coords)

for i in range(num):

    img = untr(imgs[i].data.contiguous().cpu())
    img2 = untr(xis[i].data.contiguous().cpu())

    img = untr2(img)
    img2 = untr2(img2)

    eye = eye_coords[i].view(1, 169)

    pred = outputs[i].data.view(1, 169)

    ind = pred.max(1)[1]
    step = 1 / 26.0
    y = ((float(ind[0]/ 13)) / 13.0) + step
    x = ((float(ind[0] % 13)) / 13.0) + step

    to_pil = torchvision.transforms.ToPILImage()
    im = to_pil(img.float())
    im2 = to_pil(img2.float())

    eye_np = eyes[i]

    # print(eye_np)
    # print(x * 227, y * 227)

    plt.subplot(131)
    plt.plot([x* 227, eye_np[0]* 227],[y* 227, eye_np[1]* 227])
    plt.imshow(im)

    plt.subplot(133)
    plt.imshow(im2)

    plt.subplot(132)
    plt.imshow(eye_coords[i].data.cpu().numpy())
    plt.show()
