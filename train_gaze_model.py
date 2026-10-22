"""Command-line entry point for training or evaluating the gaze-following model.

main parses shared CLI options, resolves the device, seeds initialization,
checks annotation paths, builds or resumes the model and optimizer, and opens
the GazeFollow loaders and TensorBoard writer. It runs the outer training and
validation loop, applies the learning-rate schedule, and saves periodic/final
checkpoints. Epoch validation uses either an image-disjoint development split or the
corrected test2 split when --validation-split test is selected.
--testOnly evaluates the official test split without starting training epochs.

A complete resume checkpoint supplies both backbones and avoids unnecessary
pretrained-weight downloads. New pretrained runs still need Places weights and
need the downloaded ImageNet weights. No work runs on import. Example from the root:
python train_gaze_model.py --device cuda --data_dir data/gazefollow --no-withaug
Add --placesmodelpath for pretrained training or --random-init for diagnostics.
"""

from __future__ import annotations
from pathlib import Path
import json
import torch
import shutil
from datetime import datetime

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
    if not 0 <= options.validation_fraction < 1:
        raise ValueError("validation-fraction must be in [0,1)")
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
        checkpoint_configuration = torch.load(
            options.resume, map_location="cpu", weights_only=True
        ).get("configuration", {})
        for name in ("architecture", "backbone"):
            if name in checkpoint_configuration:
                setattr(options, name, checkpoint_configuration[name])
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
    run_directory = Path(
        options.log_dir or f"runs/{options.exp}_{datetime.now():%Y%m%d_%H%M%S}"
    )
    run_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "configuration.json").write_text(
        json.dumps(vars(options), indent=2) + "\n"
    )
    (run_directory / "split_indices.json").write_text(
        json.dumps(
            {
                "train": dataset_loaders.training_indices,
                "validation": dataset_loaders.validation_indices,
                "validation_source": dataset_loaders.validation_source,
                "training_annotations": str(Path(options.data_dir) / "train_annotations.mat"),
                "validation_annotations": str(
                    Path(options.data_dir)
                    / (
                        "test_annotations.mat"
                        if dataset_loaders.validation_source == "test"
                        else "train_annotations.mat"
                    )
                ),
            }
        )
    )
    print(
        f"Training examples: {len(dataset_loaders.training_indices)}; validation examples: {len(dataset_loaders.validation_indices)}; validation source: {dataset_loaders.validation_source}",
        flush=True,
    )
    with SummaryWriter(log_dir=str(run_directory), flush_secs=10) as scalar_writer:
        trainer = training_engine.Trainer(
            gaze_model, loss_function, optimizer, options, scalar_writer
        )
        validator = training_engine.Validator(gaze_model, loss_function, options, scalar_writer)
        if options.testOnly:
            validator.validate(dataset_loaders.test_loader, options.start_epoch, options)
            return
        if not dataset_loaders.validation_indices:
            raise ValueError("No validation images in the selected validation split")
        validator.validate(dataset_loaders.validation_loader, options.start_epoch - 1, options)
        scalar_writer.flush()
        for epoch in range(options.start_epoch, options.epochs):
            training_support.adjust_learning_rate(options, optimizer, epoch)
            print(
                f'Starting epoch {epoch + 1}, learning rate {optimizer.param_groups[0]["lr"]}'
            )
            trainer.train(dataset_loaders.train_loader, epoch, options)
            validation_distance = validator.validate(
                dataset_loaders.validation_loader, epoch, options
            )
            improved = validation_distance < best_validation_distance
            best_validation_distance = min(best_validation_distance, validation_distance)
            model_utils.save_checkpoint(
                options, gaze_model, optimizer, best_validation_distance, epoch
            )
            if improved:
                shutil.copyfile(
                    Path(options.store) / "last.pth.tar", Path(options.store) / "best.pth.tar"
                )
            scalar_writer.flush()
            print(f"Best error: {best_validation_distance:.3f}")


if __name__ == "__main__":
    main()
