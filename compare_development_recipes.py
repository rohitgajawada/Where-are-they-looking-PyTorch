"""Compare short controlled training recipes on an image-disjoint development set.

Each candidate starts from the same seed and official pretrained backbones,
uses the same training images and number of optimizer updates, and is evaluated
on the same held-out images. Candidate learning rates, original versus paper
FC layers, and augmentations are compared sequentially with real TensorBoard
metrics. The selected JSON recipe is an initial development choice, not proof
that the full reproduction matches the paper or is optimal across random seeds.
"""

from __future__ import annotations
import copy
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

import command_line_options
import gazefollow_dataset
import models
from device_and_initialization import seed_everything
from training_engine import Trainer, Validator
from type_definitions import Options


def candidate(
    base: Options,
    name: str,
    learning_rate: float,
    architecture: str,
    augmentation: bool,
    crop: bool,
) -> dict[str, str | float | bool]:
    """Train one controlled short candidate and record held-out distance and loss."""
    options = copy.deepcopy(base)
    options.maxlr = learning_rate
    options.architecture = architecture
    options.withaug = augmentation
    options.random_crop = crop
    seed_everything(options.seed)
    loaders = gazefollow_dataset.GazeFollow(options)
    train = DataLoader(
        Subset(loaders.train_dataset, loaders.training_indices[:2048]),
        batch_size=32,
        shuffle=True,
        num_workers=4,
        worker_init_fn=gazefollow_dataset.seed_worker,
    )
    validation = DataLoader(
        Subset(
            loaders.validation_dataset, list(range(min(512, len(loaders.validation_dataset))))
        ),
        batch_size=64,
        num_workers=4,
    )
    model, criterion, optimizer = models.setup(models.load_model(options), options)
    run = Path("runs") / name
    run.mkdir(parents=True, exist_ok=True)
    run.joinpath("configuration.json").write_text(json.dumps(vars(options), indent=2) + "\n")
    with SummaryWriter(str(run), flush_secs=5) as writer:
        trainer = Trainer(model, criterion, optimizer, options, writer)
        validator = Validator(model, criterion, options, writer)
        validator.validate(validation, -1, options)
        trainer.train(train, 0, options)
        distance = validator.validate(validation, 0, options)
        result = {
            "name": name,
            "learning_rate": learning_rate,
            "architecture": architecture,
            "augmentation": augmentation,
            "random_crop": crop,
            "validation_distance": distance,
            "validation_loss": validator.loss_meter.avg,
            "training_loss": trainer.loss_meter.avg,
        }
    run.joinpath("result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("CANDIDATE", json.dumps(result), flush=True)
    del model, optimizer, trainer, validator
    torch.cuda.empty_cache()
    return result


def main() -> None:
    """Calibrate two learning rates, then compare legacy layers and augmentations."""
    options = command_line_options.optionargparser().parse_args()
    options.device = "cuda"
    options.workers = 4
    options.optimType = "adam"
    options.pretrained_lr_scale = 0.1
    options.verbose = True
    options.printfreq = 20
    torch.set_num_threads(4)
    results = [
        candidate(options, "calibration_paper_lr_0003", 0.0003, "paper", False, False),
        candidate(options, "calibration_paper_lr_001", 0.001, "paper", False, False),
    ]
    best = min(results, key=lambda result: result["validation_distance"])
    rate = float(best["learning_rate"])
    results.append(candidate(options, "calibration_legacy", rate, "legacy", False, False))
    results.append(
        candidate(options, "calibration_horizontal_flip", rate, "paper", True, False)
    )
    results.append(candidate(options, "calibration_flip_crop", rate, "paper", True, True))
    selected = min(results, key=lambda result: result["validation_distance"])
    Path("runs/development_comparison.json").write_text(
        json.dumps(
            {
                "results": results,
                "selected": selected,
                "training_examples": 2048,
                "validation_examples": 512,
                "steps": 64,
            },
            indent=2,
        )
        + "\n"
    )
    print("SELECTED", json.dumps(selected), flush=True)


if __name__ == "__main__":
    main()
