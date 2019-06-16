import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import getdata as ld
import utils
import os
import opts
from PIL import Image
import models.gazenet as gazenet
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib

parser = opts.optionargparser()
opt = parser.parse_args()

opt.testbatchsize = 16

checkpoint = torch.load('./savedmodels/gazenet_gazefollow_softmaxnesterovsgd_70epoch.pth.tar')
print("Loading pretrained model: ")
start_epoch = checkpoint['epoch']
best_err = checkpoint['best_err']

model = gazenet.Net(opt).cuda()
model.load_state_dict(checkpoint['state_dict'])

dataloader = ld.GazeFollow(opt)

images, xis, eye_coords, pred_coords, eyes, names, eyes2, gaze_final = next(iter(dataloader.val_loader))

images, xis, eye_coords, pred_coords, gaze_final = images.cuda(), xis.cuda(), eye_coords.cuda(), pred_coords.cuda(), gaze_final.cuda()

outputs = model.predict(images, xis, eye_coords)

untr = transforms.Compose([
        transforms.Normalize([0, 0, 0], [1/(0.229), 1/(0.224), 1/(0.225)])])
untr2 = transforms.Compose([
        transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1])])

to_pil = torchvision.transforms.ToPILImage()

for i in range(16):

    name = names[i]
    img = untr(images[i].data.contiguous().cpu())
    img = untr2(img)

    print(torch.max(outputs[i]))

    heat_in = torch.clamp(outputs[i].data.cpu(), 0, 1)

    heatmap = to_pil(heat_in)
    plt.imshow(heatmap)
    plt.savefig('./model_outputs/' + str(i) + '_heat.jpg')
    plt.clf()


    pred = outputs[i].data.view(1, 227 * 227)
    ind = pred.max(1)[1]

    y = ((float(ind[0]/ 227.0)) / 227.0)
    x = ((float(ind[0] % 227.0)) / 227.0)

    print(x, y)

    im = to_pil(img)
    eye_np = eyes[i].cpu().numpy()

    print(eye_np)
    print(x * 227, y * 227)

    plt.plot([x* 227, eye_np[0]* 227],[y* 227, eye_np[1]* 227])
    plt.imshow(im)
    plt.savefig('./model_outputs/' + str(i) + '.jpg')
    plt.clf()

#     exit()
