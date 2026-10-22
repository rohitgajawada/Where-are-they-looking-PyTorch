"""Run gaze-model training epochs and evaluate decoded fixation predictions.

Trainer moves collated scene/head images and eye grids to the model device,
averages NLL across the five shifted-grid classification heads, backpropagates,
and updates the optimizer. It records sample-weighted loss and timing through
a TensorBoard-compatible scalar writer. Batch size one is supported.

Validator runs without gradients, finds each heatmap's global maximum, and
reports normalized distance to the mean human fixation and to the closest human
fixation. It returns the mean-location distance used for checkpoint selection.
Model construction, dataset loading, CLI parsing, and the outer epoch loop live
in separate modules; importing this engine does not start training.
"""

from __future__ import annotations
from type_definitions import GazeLoader, Options, ScalarWriter
from models.gaze_following_network import Net
from torch.optim import Optimizer
import torch
from training_support import AverageMeter
import time
import gaze_metrics
import sys


class Trainer:
    """Optimize the gaze model while tracking sample-weighted loss and timing."""

    def __init__(
        self,
        model: Net,
        loss_function: torch.nn.Module,
        optimizer: Optimizer,
        options: Options,
        scalar_writer: ScalarWriter,
    ) -> None:
        """Bind the model, optimizer, loss, scalar writer, device, and epoch meters."""
        self.model = model
        self.device = next(model.parameters()).device
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.batch_time_meter = AverageMeter()
        self.data_time_meter = AverageMeter()
        self.loss_meter = AverageMeter()
        self.scalar_writer = scalar_writer

    def train(self, train_loader: GazeLoader, epoch: int, options: Options) -> None:
        """Train one epoch by averaging NLL across five shifted-grid heads.

        Batches contain scene images, head crops, eye grids, five one-hot targets,
        eye coordinates, sample indices, head-local eyes, and padded gaze labels.
        This method updates model weights and logs the sample-weighted training loss.
        """
        self.data_time_meter.reset()
        self.batch_time_meter.reset()
        self.model.train()
        self.loss_meter.reset()

        previous_batch_end_time = time.time()
        for batch_index, batch in enumerate(train_loader, 0):

            self.optimizer.zero_grad()

            (
                scene_images,
                head_images,
                eye_position_grids,
                shifted_grid_targets,
                eye_positions,
                sample_indices,
                head_eye_positions,
                gaze_annotations,
            ) = batch
            scene_images = scene_images.to(self.device, non_blocking=True)
            head_images = head_images.to(self.device, non_blocking=True)
            eye_position_grids = eye_position_grids.to(self.device, non_blocking=True)
            shifted_grid_targets = shifted_grid_targets.to(self.device, non_blocking=True).reshape(
                scene_images.size(0), 5, 25
            )

            self.data_time_meter.update(time.time() - previous_batch_end_time)

            grid_log_probabilities = self.model(scene_images, head_images, eye_position_grids)
            total_loss = self.loss_function(
                grid_log_probabilities[0], shifted_grid_targets[:, 0, :].max(1)[1]
            )
            for grid_index in range(1, len(grid_log_probabilities)):
                total_loss += self.loss_function(
                    grid_log_probabilities[grid_index],
                    shifted_grid_targets[:, grid_index, :].max(1)[1],
                )

            total_loss = total_loss / (len(grid_log_probabilities) * 1.0)

            total_loss.backward()
            self.optimizer.step()

            batch_size = scene_images.size(0)
            self.loss_meter.update(total_loss.item(), batch_size)
            self.batch_time_meter.update(time.time() - previous_batch_end_time)
            previous_batch_end_time = time.time()

            if batch_index % options.printfreq == 0 and options.verbose:
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Time {batch_time.avg:.3f} ({batch_time.sum:.3f})\t"
                    "Data {data_time.avg:.3f} ({data_time.sum:.3f})\t"
                    "Loss {loss.avg:.3f}\t".format(
                        epoch,
                        batch_index,
                        len(train_loader),
                        batch_time=self.batch_time_meter,
                        data_time=self.data_time_meter,
                        loss=self.loss_meter,
                    )
                )

            sys.stdout.flush()

        self.scalar_writer.add_scalar("Train Loss", self.loss_meter.avg, epoch)
        print(
            "Train: [{0}]\t"
            "Time {batch_time.sum:.3f}\t"
            "Data {data_time.sum:.3f}\t"
            "Loss {loss.avg:.3f}\t".format(
                epoch,
                batch_time=self.batch_time_meter,
                data_time=self.data_time_meter,
                loss=self.loss_meter,
            )
        )


class Validator:
    """Evaluate global heatmap peaks against the human gaze annotations."""

    def __init__(
        self,
        model: Net,
        loss_function: torch.nn.Module,
        options: Options,
        scalar_writer: ScalarWriter,
    ) -> None:
        """Bind the model, writer, device, and validation distance/timing meters."""

        self.model = model
        self.device = next(model.parameters()).device
        self.loss_function = loss_function
        self.batch_time_meter = AverageMeter()
        self.data_time_meter = AverageMeter()
        self.mean_distance_meter = AverageMeter()
        self.minimum_distance_meter = AverageMeter()
        self.scalar_writer = scalar_writer

    def validate(self, validation_loader: GazeLoader, epoch: int, options: Options) -> float:
        """Evaluate decoded gaze predictions without gradient tracking.

        Returns the sample-weighted distance to each person's mean annotation,
        in normalized image coordinates. Also logs minimum annotation distance.
        """

        self.model.eval()
        self.mean_distance_meter.reset()
        self.minimum_distance_meter.reset()
        self.data_time_meter.reset()
        self.batch_time_meter.reset()
        previous_batch_end_time = time.time()

        with torch.no_grad():
            for batch_index, batch in enumerate(validation_loader, 0):

                (
                    scene_images,
                    head_images,
                    eye_position_grids,
                    shifted_grid_targets,
                    eye_positions,
                    sample_indices,
                    head_eye_positions,
                    gaze_annotations,
                ) = batch
                scene_images = scene_images.to(self.device, non_blocking=True)
                head_images = head_images.to(self.device, non_blocking=True)
                eye_position_grids = eye_position_grids.to(self.device, non_blocking=True)

                self.data_time_meter.update(time.time() - previous_batch_end_time)
                heatmaps = self.model.predict(scene_images, head_images, eye_position_grids)

                predicted_flat_indices = heatmaps.flatten(1).argmax(1)
                batch_size = scene_images.size(0)

                mean_distance = gaze_metrics.euclid_dist(
                    predicted_flat_indices.cpu(), gaze_annotations, batch_size
                )
                minimum_distance = gaze_metrics.euclid_mindist(
                    predicted_flat_indices.cpu(), gaze_annotations, batch_size
                )

                self.mean_distance_meter.update(mean_distance, batch_size)
                self.minimum_distance_meter.update(minimum_distance, batch_size)
                self.batch_time_meter.update(time.time() - previous_batch_end_time)
                previous_batch_end_time = time.time()

                if batch_index % options.printfreq == 0 and options.verbose:
                    print(
                        "Epoch: [{0}][{1}/{2}]\t"
                        "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                        "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                        "Dist {dist.avg:.3f}\t"
                        "MinDist {mindist.avg:.3f}\t".format(
                            epoch,
                            batch_index,
                            len(validation_loader),
                            batch_time=self.batch_time_meter,
                            data_time=self.data_time_meter,
                            dist=self.mean_distance_meter,
                            mindist=self.minimum_distance_meter,
                        )
                    )

                sys.stdout.flush()

            self.scalar_writer.add_scalar("Val Dist", self.mean_distance_meter.avg, epoch)
            self.scalar_writer.add_scalar("Val Min Dist", self.minimum_distance_meter.avg, epoch)

            print(
                "Val: [{0}]\t"
                "Time {batch_time.sum:.3f}\t"
                "Data {data_time.sum:.3f}\t"
                "Dist {dist.avg:.3f}\t"
                "MinDist {mindist.avg:.3f}\t".format(
                    epoch,
                    batch_time=self.batch_time_meter,
                    data_time=self.data_time_meter,
                    dist=self.mean_distance_meter,
                    mindist=self.minimum_distance_meter,
                )
            )

        return self.mean_distance_meter.avg
