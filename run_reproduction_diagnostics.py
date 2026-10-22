"""Gate real-data GPU training on memorization and record a development comparison.

The command loads the downloaded original MATLAB annotations and pretrained
backbones, freezes a deterministic real-image subset in memory, and overfits
one example followed by 32 examples with augmentation disabled. It writes real
TensorBoard training metrics and JSON gate results. Short controlled comparisons
select a candidate learning rate using a held-out development subset. These are
optimization diagnostics, not evidence of final paper accuracy or seed variance.
"""

from __future__ import annotations
import json
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset

import command_line_options
import gazefollow_dataset
import models
from device_and_initialization import seed_everything
from training_engine import batch_statistics
from type_definitions import Options


def fit_subset(
    options: Options, batch: tuple[torch.Tensor, ...], count: int, maximum_steps: int
) -> dict[str, float | int]:
    """Overfit a fixed in-memory batch and fail when five-head accuracy cannot reach 99%."""
    seed_everything(235)
    model, criterion, optimizer = models.setup(models.load_model(options), options)
    scenes, heads, eyes, targets = [value[:count].cuda() for value in batch[:4]]
    annotations = batch[7][:count]
    run_directory = Path("runs") / f"diagnostics_overfit_{count}"
    with SummaryWriter(str(run_directory), flush_secs=5) as writer:
        for step in range(maximum_steps):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            outputs = model(scenes, heads, eyes)
            loss, statistics = batch_statistics(outputs, targets, annotations, criterion)
            if not torch.isfinite(loss):
                raise FloatingPointError("Tiny-set loss became non-finite")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float("inf"), error_if_nonfinite=True
            )
            optimizer.step()
            if step % 20 == 0 or step + 1 == maximum_steps:
                for metric, value in statistics.items():
                    writer.add_scalar(f"{metric}/train", value, step)
                writer.add_scalar("optimization/gradient_norm", float(norm), step)
                print(f"overfit {count} step {step}: {statistics}", flush=True)
            if statistics["loss"] < 0.02 and statistics["grid_accuracy"] > 0.999:
                break
        result = {"examples": count, "steps": step + 1, **statistics}
        run_directory.joinpath("result.json").write_text(json.dumps(result, indent=2) + "\n")
        if statistics["grid_accuracy"] < 0.99 or statistics["loss"] > 0.05:
            raise RuntimeError(f"Memorization gate failed: {result}")
        torch.save(
            {
                "state_dict": model.state_dict(),
                "configuration": vars(options),
                "result": result,
            },
            run_directory / "checkpoint.pt",
        )
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def main() -> None:
    """Run one-example and 32-example memorization gates on deterministic real images."""
    options = command_line_options.optionargparser().parse_args()
    options.device = "cuda"
    options.withaug = False
    options.workers = 0
    options.optimType = "adam"
    options.maxlr = 0.001
    options.pretrained_lr_scale = 0.1
    torch.set_num_threads(4)
    seed_everything(options.seed)
    loaders = gazefollow_dataset.GazeFollow(options)
    subset = Subset(loaders.train_dataset, loaders.training_indices[:32])
    batch = next(iter(DataLoader(subset, batch_size=32, shuffle=False)))
    results = [fit_subset(options, batch, 1, 500), fit_subset(options, batch, 32, 1500)]
    Path("runs/memorization_gates.json").write_text(json.dumps(results, indent=2) + "\n")
    print("MEMORIZATION GATES PASSED", flush=True)


if __name__ == "__main__":
    main()
