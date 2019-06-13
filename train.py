import torch
from utils import AverageMeter
import time
import models.__init__ as init
import utils
import sys


class Trainer():
    def __init__(self, model, criterion, optimizer, opt, writer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.losses = AverageMeter()
        self.writer = writer

    def train(self, trainloader, epoch, opt):
        self.data_time.reset()
        self.batch_time.reset()
        self.model.train()
        self.losses.reset()

        end = time.time()
        for i, data in enumerate(trainloader, 0):

            self.optimizer.zero_grad()

            xh, xi, xp, shifted_targets, eyes, names, eyes2, gcorrs = data
            xh = xh.cuda()  
            xi = xi.cuda()
            xp = xp.cuda()
            shifted_targets = shifted_targets.cuda().squeeze()

            self.data_time.update(time.time() - end)

            outputs = self.model(xh, xi, xp)
            total_loss = self.criterion(outputs[0], shifted_targets[:, 0, :].max(1)[1])
            for j in range(1, len(outputs)):
                total_loss += self.criterion(outputs[j], shifted_targets[:, j, :].max(1)[1])
            
            total_loss = total_loss / (len(outputs) * 1.0)

            total_loss.backward()
            self.optimizer.step()

            inputs_size = xh.size(0)
            self.losses.update(total_loss.item(), inputs_size)
            self.batch_time.update(time.time() - end)
            end = time.time()

            if i % opt.printfreq == 0 and opt.verbose:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.avg:.3f} ({batch_time.sum:.3f})\t'
                      'Data {data_time.avg:.3f} ({data_time.sum:.3f})\t'
                      'Loss {loss.avg:.3f}\t'.format(
                       epoch, i, len(trainloader), batch_time=self.batch_time,
                       data_time= self.data_time, loss=self.losses))

            sys.stdout.flush()


        self.writer.add_scalar('Train Loss', self.losses.avg, epoch)
        print('Train: [{0}]\t'
              'Time {batch_time.sum:.3f}\t'
              'Data {data_time.sum:.3f}\t'
              'Loss {loss.avg:.3f}\t'.format(
               epoch, batch_time=self.batch_time,
               data_time= self.data_time, loss=self.losses))


class Validator():
    def __init__(self, model, criterion, opt, writer):

        self.model = model
        self.criterion = criterion
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.dist = AverageMeter()
        self.mindist = AverageMeter()
        self.writer = writer

    def validate(self, valloader, epoch, opt):

        self.model.eval()
        self.dist.reset()
        self.mindist.reset()
        self.data_time.reset()
        self.batch_time.reset()
        end = time.time()

        with torch.no_grad():
            for i, data in enumerate(valloader, 0): 

                xh, xi, xp, targets, eyes, names, eyes2, ground_labels = data
                xh = xh.cuda()
                xi = xi.cuda()
                xp = xp.cuda()

                self.data_time.update(time.time() - end)
                outputs = self.model.predict(xh, xi, xp)

                pred_labels = outputs.max(1)[1]
                inputs_size = xh.size(0)

                distval = utils.euclid_dist(pred_labels.data.cpu(), ground_labels, inputs_size)
                mindistval = utils.euclid_mindist(pred_labels.data.cpu(), ground_labels, inputs_size)

                self.dist.update(distval, inputs_size)
                self.mindist.update(mindistval, inputs_size)
                self.batch_time.update(time.time() - end)
                end = time.time()

                if i % opt.printfreq == 0 and opt.verbose:
                    print('Epoch: [{0}][{1}/{2}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                            'Dist {dist.avg:.3f}\t'
                            'MinDist {mindist.avg:.3f}\t'
                            .format(
                            epoch, i, len(valloader), batch_time=self.batch_time,
                            data_time= self.data_time, dist=self.dist, mindist=self.mindist))

                sys.stdout.flush()


            self.writer.add_scalar('Val Dist', self.dist.avg, epoch)
            self.writer.add_scalar('Val Min Dist', self.mindist.avg, epoch)

            print('Val: [{0}]\t'
                    'Time {batch_time.sum:.3f}\t'
                    'Data {data_time.sum:.3f}\t'
                    'Dist {dist.avg:.3f}\t' 'MinDist {mindist.avg:.3f}\t'.format(
                    epoch, batch_time=self.batch_time,
                    data_time= self.data_time, dist=self.dist, mindist=self.mindist))

        return self.dist.avg
