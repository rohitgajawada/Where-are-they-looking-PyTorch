from torch.autograd import Variable
from utils import AverageMeter
from utils import accuracy
from copy import deepcopy
import time

class Trainer():
    def __init__(self, model, criterion, optimizer, opt):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.losses = AverageMeter()
        self.acc = AverageMeter()

    def train(self, trainloader, epoch, opt):
        self.data_time.reset()
        self.batch_time.reset()
        self.model.train()
        self.losses.reset()
        self.acc.reset()

        end = time.time()
        for i, data in enumerate(trainloader, 0):

            self.optimizer.zero_grad()
            if opt.cuda:
                inputs, targets = data
                inputs = inputs.cuda(async=True)
                targets = targets.cuda(async=True)

            inputs, targets = Variable(inputs), Variable(targets)

            self.data_time.update(time.time() - end)

            outputs = self.model(inputs)
            #TODO: add get max prob coordinates

            loss = self.criterion(outputs, targets)
            acc = accuracy(outputs.data.max(1)[1], targets.data, opt)

            loss.backward()
            self.optimizer.step()

            #TODO:To be changed to AUC
            inputs_size = inputs.size(0)
            self.acc.update(acc, inputs_size)
            self.losses.update(loss.data[0], inputs_size)

            # measure elapsed time
            self.batch_time.update(time.time() - end)
            end = time.time()

            if i % opt.printfreq == 0 and opt.verbose == True:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.avg:.3f} ({batch_time.sum:.3f})\t'
                      'Data {data_time.avg:.3f} ({data_time.sum:.3f})\t'
                      'Loss {loss.avg:.3f}\t'
                      'Accuracy {acc.avg:.4f}\t'.format(
                       epoch, i, len(trainloader), batch_time=self.batch_time,
                       data_time= self.data_time, loss=self.losses, acc=self.acc))

        print('Train: [{0}]\t'
              'Time {batch_time.sum:.3f}\t'
              'Data {data_time.sum:.3f}\t'
              'Loss {loss.avg:.3f}\t'
              'Accuracy {acc.avg:.4f}\t'.format(
               epoch, batch_time=self.batch_time,
               data_time= self.data_time, loss=self.losses,
               acc=self.acc))


class Validator():
    def __init__(self, model, criterion, opt):

        self.model = model
        self.realparams = deepcopy(model.parameters)
        self.criterion = criterion
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.losses = AverageMeter()
        self.acc = AverageMeter()

    def validate(self, valloader, epoch, opt):

        self.model.eval()
        self.losses.reset()
        self.acc.reset()
        self.data_time.reset()
        self.batch_time.reset()
        end = time.time()

        for i, data in enumerate(valloader, 0):
            if opt.cuda:
                inputs, targets = data
                inputs = inputs.cuda(async=True)
                targets = targets.cuda(async=True)

            inputs, targets = Variable(inputs), Variable(targets)

            self.data_time.update(time.time() - end)
            outputs = self.model(inputs)
            #TODO: add get max prob coordinates

            loss = self.criterion(outputs, targets)
            acc = accuracy(outputs.data.max(1)[1], targets.data, opt)

            #TODO:To be changed to AUC
            inputs_size = inputs.size(0)
            self.acc.update(acc, inputs_size)
            self.losses.update(loss.data[0], inputs_size)

            # measure elapsed time
            self.batch_time.update(time.time() - end)
            end = time.time()

            if i % opt.printfreq == 0 and opt.verbose == True:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy {acc.val:.4f} ({acc.avg:.4f})\t'.format(
                       epoch, i, len(valloader), batch_time=self.batch_time,
                       data_time= self.data_time, loss=self.losses,
                       acc=self.acc))

        finalacc = self.acc.avg

        print('Val: [{0}]\t'
              'Time {batch_time.sum:.3f}\t'
              'Data {data_time.sum:.3f}\t'
              'Loss {loss.avg:.3f}\t'
              'Accuracy {acc:.4f}\t'.format(
               epoch, batch_time=self.batch_time,
               data_time= self.data_time, loss=self.losses,
               acc=finalacc))

        return self.top1.avg
