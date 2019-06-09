import torch
from utils import AverageMeter
from copy import deepcopy
import time
import models.__init__ as init
import utils


class Trainer():
    def __init__(self, model, criterion, optimizer, opt):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.losses = AverageMeter()

    def train(self, trainloader, epoch, opt):
        self.data_time.reset()
        self.batch_time.reset()
        self.model.train()
        self.losses.reset()

        end = time.time()
        for i, data in enumerate(trainloader, 0):

            self.optimizer.zero_grad()

            if opt.cuda:
                xh, xi, xp, targets, eyes, names, eyes2, gcorrs = data
                xh = xh.cuda()  
                xi = xi.cuda()
                xp = xp.cuda()
                targets = targets.cuda().squeeze()

            xh, xi, xp, targets = xh, xi, xp, targets

            self.data_time.update(time.time() - end)

            outputs = self.model(xh, xi, xp)
            loss = self.criterion(outputs, targets.max(1)[1])

            loss.backward()
            self.optimizer.step()

            inputs_size = xh.size(0)
            self.losses.update(loss.item(), inputs_size)
            self.batch_time.update(time.time() - end)
            end = time.time()

            if i % opt.printfreq == 0 and opt.verbose == True:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.avg:.3f} ({batch_time.sum:.3f})\t'
                      'Data {data_time.avg:.3f} ({data_time.sum:.3f})\t'
                      'Loss {loss.avg:.3f}\t'.format(
                       epoch, i, len(trainloader), batch_time=self.batch_time,
                       data_time= self.data_time, loss=self.losses))

        print('Train: [{0}]\t'
              'Time {batch_time.sum:.3f}\t'
              'Data {data_time.sum:.3f}\t'
              'Loss {loss.avg:.3f}\t'.format(
               epoch, batch_time=self.batch_time,
               data_time= self.data_time, loss=self.losses))


class Validator():
    def __init__(self, model, criterion, opt):

        self.model = model
        self.criterion = criterion
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.dist = AverageMeter()
        self.mindist = AverageMeter()

    def validate(self, valloader, epoch, opt):

        self.model.eval()
        self.dist.reset()
        self.mindist.reset()
        self.data_time.reset()
        self.batch_time.reset()
        end = time.time()

        for i, data in enumerate(valloader, 0):  #follow new practices for gradient calculation removal
            if opt.cuda:
                xh, xi, xp, targets, eyes, names, eyes2, gcorrs = data
                xh = xh.cuda()
                xi = xi.cuda()
                xp = xp.cuda()
                targets = targets.cuda().squeeze()
                gcorrs = gcorrs.cuda()

            xh, xi, xp, targets = xh, xi, xp, targets

            self.data_time.update(time.time() - end)
            outputs = self.model.predict(xh, xi, xp)

            ground_labels = gcorrs
            pred_labels = outputs.max(1)[1]
            inputs_size = xh.size(0)

            distval = utils.euclid_dist(pred_labels.data.cpu(), ground_labels.cpu(), inputs_size)
            mindistval = utils.euclid_mindist(pred_labels.data.cpu(), ground_labels.cpu(), inputs_size)

            self.dist.update(distval, inputs_size)
            self.mindist.update(mindistval, inputs_size)
            self.batch_time.update(time.time() - end)
            end = time.time()

            if i % opt.printfreq == 0 and opt.verbose == True:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Dist {dist.avg:.3f}\t'
                      'MinDist {mindist.avg:.3f}\t'
                      .format(
                       epoch, i, len(valloader), batch_time=self.batch_time,
                       data_time= self.data_time, dist=self.dist, mindist=self.mindist))

        print('Val: [{0}]\t'
              'Time {batch_time.sum:.3f}\t'
              'Data {data_time.sum:.3f}\t'
              'Dist {dist.avg:.3f}\t' 'MinDist {mindist.avg:.3f}\t'.format(
               epoch, batch_time=self.batch_time,
               data_time= self.data_time, dist=self.dist, mindist=self.mindist))

        return self.dist.avg
