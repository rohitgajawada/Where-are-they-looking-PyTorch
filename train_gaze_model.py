"""Command-line entry point for training or evaluating the gaze-following model.

main parses shared CLI options, resolves the device, seeds initialization,
checks annotation paths, builds or resumes the model and optimizer, and opens
the GazeFollow loaders and TensorBoard writer. It runs the outer training and
validation loop, applies the learning-rate schedule, and saves periodic/final
checkpoints. --testOnly evaluates once without starting training epochs.

A complete resume checkpoint supplies both backbones and avoids unnecessary
pretrained-weight downloads. New pretrained runs still need Places weights and
may download ImageNet weights. No work runs on import. Example from the root:
python train_gaze_model.py --device cuda --data_dir data/gazefollow --no-withaug
Add --placesmodelpath for pretrained training or --random-init for diagnostics.
"""

from __future__ import annotations
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

import gazefollow_dataset as gaze_data
import models as model_utils
import command_line_options
import training_engine
import training_support
from device_and_initialization import resolve_device, seed_everything


def main() -> None:
    """Parse options, seed the run, load assets, and train or evaluate on the selected device."""
    options = command_line_options.optionargparser().parse_args()
    options.device = str(resolve_device(options.device))
    seed_everything(options.seed)
    print(options)
    for split in ("train", "test"):
        annotation_path = Path(options.data_dir) / f"{split}_annotations.mat"
        if not annotation_path.is_file():
            raise FileNotFoundError(f"GazeFollow annotation file not found: {annotation_path}")
    if options.resume != "none":
        if not Path(options.resume).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {options.resume}")
        options.random_init = True  # The resume checkpoint supplies all model weights.
    gaze_model = model_utils.load_model(options)
    gaze_model, loss_function, optimizer = model_utils.setup(gaze_model, options)
    best_validation_distance = float("inf")
    if options.resume != "none":
        if not Path(options.resume).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {options.resume}")
        gaze_model, optimizer, options, best_validation_distance = model_utils.resumer(
            options, gaze_model, optimizer
        )
    dataset_loaders = gaze_data.GazeFollow(options)
    with SummaryWriter() as scalar_writer:
        trainer = training_engine.Trainer(
            gaze_model, loss_function, optimizer, options, scalar_writer
        )
        validator = training_engine.Validator(gaze_model, loss_function, options, scalar_writer)
        if options.testOnly:
            validator.validate(dataset_loaders.test_loader, options.start_epoch, options)
            return
        for epoch in range(options.start_epoch, options.epochs):
            training_support.adjust_learning_rate(options, optimizer, epoch)
            print(f'Starting epoch {epoch + 1}, learning rate {optimizer.param_groups[0]["lr"]}')
            trainer.train(dataset_loaders.train_loader, epoch, options)
            validation_distance = validator.validate(dataset_loaders.test_loader, epoch, options)
            best_validation_distance = min(best_validation_distance, validation_distance)
            if epoch % 10 == 0 or epoch == options.epochs - 1:
                model_utils.save_checkpoint(
                    options, gaze_model, optimizer, best_validation_distance, epoch
                )
            print(f"Best error: {best_validation_distance:.3f}")


if __name__ == "__main__":
    main()
