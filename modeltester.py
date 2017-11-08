import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import utils
import os
import opts
from PIL import Image
import models.gazenet as gazenet

def load_input():
    img_name = ''
    img = Image.open(img_name)
    pass

parser = opts.myargparser()
opt = parser.parse_args()

checkpoint = torch.load('savedmodels/gazenet_gazefollow_best.pth.tar')
start_epoch = checkpoint['epoch']
best_err = checkpoint['best_prec1']

model = gazenet.Net(opt).cuda()
model.load_state_dict(checkpoint['state_dict'])

inputs = load_input()

outputs = model(inputs)
