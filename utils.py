import torch
import torch.nn as nn
from torch.nn import init
import copy
import random
import math
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def adjust_learning_rate(opt, optimizer, epoch):
    epoch = copy.deepcopy(epoch)
    lr = opt.maxlr
    wd = opt.weightDecay
    if opt.learningratescheduler == 'decayschedular':
        while epoch >= opt.decayinterval:
            lr = lr/opt.decaylevel
            epoch = epoch - opt.decayinterval
    lr = max(lr, opt.minlr)
    opt.lr = lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['weight_decay'] = wd

def get_mean_and_std(dataloader):
    '''Compute the mean and std value of dataset.'''
    mean = torch.zeros(3)
    std = torch.zeros(3)
    len_dataset = 0
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        len_dataset += 1
        for i in range(len(inputs[0])):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len_dataset)
    std.div_(len_dataset)
    return mean, std

class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


#Metric functions 
def euclid_dist(output, target, l):
    total = 0
    fulltotal = 0

    output = output.float()
    target = target.float()
    for i in range(l):

        predy = ((output[i] / 227.0) / 227.0) 
        predx = ((output[i] % 227.0) / 227.0) 

        ct = 0
        for j in range(100):
            ground_x = target[i][2*j]
            ground_y = target[i][2*j + 1]

            if ground_x == -1 or ground_y == -1:
                break

            temp = np.sqrt(np.power((ground_x - predx), 2) + np.power((ground_y - predy), 2))
            total += temp
            ct += 1

        total = total / float(ct * 1.0)

        fulltotal += total

    fulltotal = fulltotal / float(l * 1.0)

    return fulltotal


def euclid_mindist(output, target, l):

    fulltotal = 0
    output = output.float()
    target = target.float()
    for i in range(l):

        best = 1000000000
        
        predy = ((output[i] / 227.0) / 227.0) 
        predx = ((output[i] % 227.0) / 227.0) 

        ct = 0
        for j in range(100):
            ground_x = target[i][2*j]
            ground_y = target[i][2*j + 1]
            if ground_x == -1 or ground_y == -1:
                break

            temp = np.sqrt(np.power((ground_x - predx), 2) + np.power((ground_y - predy), 2))

            if temp < best:
                best = temp
            ct += 1

        fulltotal += best

    fulltotal = fulltotal / float(l * 1.0)
    return fulltotal


def AUCaccuracy(output, target, opt):
    pass


