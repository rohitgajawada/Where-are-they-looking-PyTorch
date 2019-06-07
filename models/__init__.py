import torch
import torch.nn as nn
import torch.optim as optim
import utils
import os
import models.gazenet as gazenet

def setup(model, opt):

    if opt.criterion == "mse":
        criterion = nn.MSELoss().cuda()
    elif opt.criterion == "crossentropy":
        criterion = nn.NLLLoss().cuda()

    if opt.optimType == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr = opt.lr, momentum = opt.momentum, nesterov = opt.nesterov, weight_decay = opt.weightDecay)
    elif opt.optimType == 'adam':
        optimizer = optim.Adam(model.parameters(), lr = opt.maxlr, weight_decay = opt.weightDecay)

    return model, criterion, optimizer

def save_checkpoint(opt, model, optimizer, best_acc, epoch):
    state = {
        'epoch': epoch + 1,
        'arch': opt.model_def,
        'state_dict': model.state_dict(),
        'best_prec1': best_acc,
        'optimizer' : optimizer.state_dict(),
    }
    epochnum = str(epoch)
    filename = "savedmodels/" + opt.model_def + '_' + opt.dataset + '_' + epochnum + 'epoch.pth.tar'
    torch.save(state, filename)

def resumer(opt, model, optimizer):

    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        opt.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(opt.resume, checkpoint['epoch']))

        return model, optimizer, opt, best_prec1


def load_model(opt):
    if opt.model_def == 'gazenet':
        model = gazenet.Net(opt)
        if opt.cuda:
            model = model.cuda()
    return model
