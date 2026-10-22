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


# Metrics consume one flattened heatmap index per sample.
def _distances(output, target, length):
    output = torch.as_tensor(output).detach().cpu().reshape(-1).long()
    target = torch.as_tensor(target).detach().cpu().float()
    if output.numel() != length or target.shape[0] != length:
        raise ValueError('Expected one flattened prediction per sample')
    points = target.reshape(length, -1, 2)
    valid = (points >= 0).all(-1) & torch.isfinite(points).all(-1)
    if not valid.any(-1).all():
        raise ValueError('Every sample needs at least one valid gaze annotation')
    predicted = torch.stack((output.remainder(227), output.div(227, rounding_mode='floor')), dim=-1).float() / 227
    clean_points = points.masked_fill(~valid.unsqueeze(-1), 0)
    mean_target = clean_points.sum(1) / valid.sum(1, keepdim=True)
    mean_dist = torch.linalg.vector_norm(predicted - mean_target, dim=-1)
    all_dist = torch.linalg.vector_norm(predicted[:, None] - clean_points, dim=-1)
    min_dist = all_dist.masked_fill(~valid, float('inf')).min(1).values
    return mean_dist, min_dist


def euclid_dist(output, target, l):
    return _distances(output, target, l)[0].mean().item()


def euclid_mindist(output, target, l):
    return _distances(output, target, l)[1].mean().item()


def AUCaccuracy(output, target, opt):
    raise NotImplementedError('Original-paper AUC protocol has not yet been implemented')
