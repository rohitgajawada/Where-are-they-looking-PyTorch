from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

import getdata as ld
import models as init
import opts
import train
import utils
from runtime import resolve_device, seed_everything


def main():
    opt = opts.optionargparser().parse_args()
    opt.device = str(resolve_device(opt.device))
    seed_everything(opt.seed)
    print(opt)
    for split in ('train', 'test'):
        annotation = Path(opt.data_dir) / f'{split}_annotations.mat'
        if not annotation.is_file():
            raise FileNotFoundError(f'GazeFollow annotation file not found: {annotation}')
    if opt.resume != "none":
        if not Path(opt.resume).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {opt.resume}")
        opt.random_init = True  # The resume checkpoint supplies all model weights.
    model = init.load_model(opt)
    model, criterion, optimizer = init.setup(model, opt)
    best_err = float('inf')
    if opt.resume != 'none':
        if not Path(opt.resume).is_file():
            raise FileNotFoundError(f'Checkpoint not found: {opt.resume}')
        model, optimizer, opt, best_err = init.resumer(opt, model, optimizer)
    data = ld.GazeFollow(opt)
    with SummaryWriter() as writer:
        trainer = train.Trainer(model, criterion, optimizer, opt, writer)
        validator = train.Validator(model, criterion, opt, writer)
        if opt.testOnly:
            validator.validate(data.val_loader, opt.start_epoch, opt)
            return
        for epoch in range(opt.start_epoch, opt.epochs):
            utils.adjust_learning_rate(opt, optimizer, epoch)
            print(f'Starting epoch {epoch + 1}, learning rate {optimizer.param_groups[0]["lr"]}')
            trainer.train(data.train_loader, epoch, opt)
            error = validator.validate(data.val_loader, epoch, opt)
            best_err = min(best_err, error)
            if epoch % 10 == 0 or epoch == opt.epochs - 1:
                init.save_checkpoint(opt, model, optimizer, best_err, epoch)
            print(f'Best error: {best_err:.3f}')


if __name__ == '__main__':
    main()
