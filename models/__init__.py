import torch
import torch.nn as nn
import torch.optim as optim
import utils
import os
import models.gazenet as gazenet

def setup(model, opt):

    if opt.criterion == "crossentropy":
        criterion = nn.NLLLoss().cuda()

    if opt.optimType == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=opt.maxlr, momentum=opt.momentum, weight_decay=opt.weightDecay, nesterov=opt.nesterov)
    elif opt.optimType == 'adam':
        optimizer = optim.Adam(model.parameters(), lr = opt.maxlr, weight_decay = opt.weightDecay)

    return model, criterion, optimizer

def save_checkpoint(opt, model, optimizer, best_err, epoch):
    state = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_err': best_err,
        'optimizer' : optimizer.state_dict(),
    }
    epochnum = str(epoch)
    filename = opt.store + "gazenet" + '_' + opt.dataset + '_' + opt.exp + '_' + epochnum +  'epoch.pth.tar'
    torch.save(state, filename)

def resumer(opt, model, optimizer):

    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        opt.start_epoch = checkpoint['epoch']
        best_err = checkpoint['best_err']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(opt.resume, checkpoint['epoch']))

        return model, optimizer, opt, best_err


def load_model(opt):
    model = gazenet.Net(opt)
    
    return model.cuda()
