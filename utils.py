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

def euclid_dist(output, target, l):
    total = 0
    for i in range(l):
        ground_x = target[2*i]
        ground_y = target[2*i + 1]

        step = 1 / 26.0
        pred = output[i]
        predy = (output[i] / 13.0) + step
        predx = (output[i] % 13) + step

        temp = np.sqrt(np.power((ground_x - predx), 2) + np.power((ground_y - predy), 2))
        total += temp

    total = total / float(l * 1.0)
    return total


def euclid_mindist(output, target, l):
    best = 1000000000
    for i in range(l):
        ground_x = target[2*i]
        ground_y = target[2*i + 1]

        step = 1 / 26.0
        pred = output[i]
        predy = (output[i] / 13.0) + step
        predx = (output[i] % 13) + step

        temp = np.sqrt(np.power((ground_x - predx), 2) + np.power((ground_y - predy), 2))
        if temp < best:
            best = temp

    return best

def AUCaccuracy(output, target, opt):
    pass

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
