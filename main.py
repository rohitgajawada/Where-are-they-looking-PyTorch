import os
import torch.backends.cudnn as cudnn
import opts
import train
import utils
import models.__init__ as init
import datasets.load_data as ld

parser = opts.myargparser()

def main():
    global opt, best_prec1
    opt = parser.parse_args()
    best_prec1 = 0
    print(opt)

    model = init.load_model(opt)
    model, criterion, optimizer = init.setup(model,opt)
    print(model)

    trainer = train.Trainer(model, criterion, optimizer, opt)
    validator = train.Validator(model, criterion, opt)

    if opt.resume:
        if os.path.isfile(opt.resume):
            model, optimizer, opt, best_prec1 = init.resumer(opt, model, optimizer)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    cudnn.benchmark = True

    dataloader = ld.LoadGazeFollow(opt)
    train_loader = dataloader.train_loader
    val_loader = dataloader.val_loader

    for epoch in range(opt.start_epoch, opt.epochs):
        utils.adjust_learning_rate(opt, optimizer, epoch)
        print("Starting epoch number:",epoch+1,"Learning rate:", optimizer.param_groups[0]["lr"])

        if opt.testOnly == False:
            trainer.train(train_loader, epoch, opt)

        acc = validator.validate(val_loader, epoch, opt)
        best_prec1 = max(acc, best_prec1)
        if best_prec1 == acc:
            init.save_checkpoint(opt, model, optimizer, best_prec1, epoch)

        print('Best accuracy: [{0:.3f}]\t'.format(best_prec1))


if __name__ == '__main__':
    main()
