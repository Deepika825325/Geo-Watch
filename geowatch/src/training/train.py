"""Configuration-driven training runtime for GeoWatch.

This module builds the Week 4 training components and provides a one-batch
dry run. Full multi-epoch training, checkpointing, early stopping and W&B
logging are added after this runtime contract has passed.

Evaluation discipline
---------------------
Only regions listed under ``dataset.train_regions`` and
``dataset.validation_regions`` are opened. Both lists must contain official
OSCD training regions. Official test-region images and labels are not used
during training or hyperparameter selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import logging
import random
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
import yaml
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.data import DataLoader

from src.data.oscd_dataset import OSCDTrainingDataset
from src.evaluation.metrics import ChangeMetrics, calculate_change_metrics
from src.models.siamese_unet import SiameseUNet, count_parameters
from src.training.augmentation import (
    PairedGeometricAugmentation,
    build_augmentation_from_config,
)
from src.training.losses import DiceFocalLoss, build_loss_from_config


LOGGER = logging.getLogger("geowatch.training")


class TrainingConfigurationError(ValueError):
    """Raised when training configuration or runtime state is invalid."""


@dataclass
class TrainingRuntime:
    """Objects required by the GeoWatch training loop."""

    config: Mapping[str, Any]
    device: torch.device
    mixed_precision_enabled: bool
    train_dataset: OSCDTrainingDataset
    validation_dataset: OSCDTrainingDataset
    train_loader: DataLoader
    validation_loader: DataLoader
    model: SiameseUNet
    criterion: DiceFocalLoss
    optimizer: Optimizer
    scheduler: LRScheduler
    scaler: torch.amp.GradScaler


@dataclass(frozen=True)
class DryRunResult:
    """Diagnostic values produced by one training and validation batch."""

    training_total_loss: float
    training_dice_loss: float
    training_focal_loss: float
    gradient_norm: float
    validation_total_loss: float
    validation_dice_loss: float
    validation_focal_loss: float
    validation_metrics: ChangeMetrics
    training_batch_shape: tuple[int, ...]
    validation_batch_shape: tuple[int, ...]


def require_mapping(
    value: Any,
    name: str,
) -> Mapping[str, Any]:
    """Return a validated mapping from the configuration."""
    if not isinstance(value, Mapping):
        raise TrainingConfigurationError(
            f"{name} must be a YAML mapping."
        )

    return value


def load_training_config(
    path: Path,
) -> Mapping[str, Any]:
    """Load and validate the root YAML configuration."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Training configuration does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as config_file:
        config = yaml.safe_load(config_file)

    return require_mapping(
        config,
        "root configuration",
    )


def validate_training_config(
    config: Mapping[str, Any],
) -> None:
    """Validate the Week 4 training and evaluation contract."""
    protocol = require_mapping(
        config.get("protocol"),
        "protocol",
    )
    dataset = require_mapping(
        config.get("dataset"),
        "dataset",
    )
    loader = require_mapping(
        dataset.get("loader"),
        "dataset.loader",
    )
    model = require_mapping(
        config.get("model"),
        "model",
    )
    optimizer = require_mapping(
        config.get("optimizer"),
        "optimizer",
    )
    scheduler = require_mapping(
        config.get("scheduler"),
        "scheduler",
    )
    training = require_mapping(
        config.get("training"),
        "training",
    )
    metrics = require_mapping(
        config.get("metrics"),
        "metrics",
    )

    if protocol.get("training_only") is not True:
        raise TrainingConfigurationError(
            "protocol.training_only must be true."
        )

    if protocol.get("official_test_regions_sealed") is not True:
        raise TrainingConfigurationError(
            "Official test regions must remain sealed."
        )

    if protocol.get("mask_binarization") != "greater_than_zero":
        raise TrainingConfigurationError(
            "Mask binarization must remain 'greater_than_zero'."
        )

    if protocol.get("positive_class") != "change":
        raise TrainingConfigurationError(
            "The positive evaluation class must remain 'change'."
        )

    bands = tuple(dataset.get("bands", ()))
    input_channels = int(dataset.get("input_channels", 0))

    if len(bands) not in {4, 6}:
        raise TrainingConfigurationError(
            "Dataset must contain four or six ordered Sentinel-2 bands."
        )

    if input_channels != len(bands):
        raise TrainingConfigurationError(
            "dataset.input_channels must equal the number of bands."
        )

    patch_size = int(dataset.get("patch_size", 0))
    stride = int(dataset.get("stride", 0))

    if patch_size <= 0 or patch_size % 32 != 0:
        raise TrainingConfigurationError(
            "dataset.patch_size must be positive and divisible by 32."
        )

    if stride <= 0:
        raise TrainingConfigurationError(
            "dataset.stride must be positive."
        )

    train_regions = tuple(dataset.get("train_regions", ()))
    validation_regions = tuple(
        dataset.get("validation_regions", ())
    )

    if not train_regions:
        raise TrainingConfigurationError(
            "At least one training region is required."
        )

    if not validation_regions:
        raise TrainingConfigurationError(
            "At least one validation region is required."
        )

    if len(set(train_regions)) != len(train_regions):
        raise TrainingConfigurationError(
            "Training regions contain duplicates."
        )

    if len(set(validation_regions)) != len(validation_regions):
        raise TrainingConfigurationError(
            "Validation regions contain duplicates."
        )

    overlap = set(train_regions).intersection(
        validation_regions
    )

    if overlap:
        raise TrainingConfigurationError(
            "Training and validation regions overlap: "
            f"{sorted(overlap)}"
        )

    for field_name in (
        "batch_size",
        "validation_batch_size",
    ):
        if int(loader.get(field_name, 0)) <= 0:
            raise TrainingConfigurationError(
                f"dataset.loader.{field_name} must be positive."
            )

    if int(loader.get("num_workers", -1)) < 0:
        raise TrainingConfigurationError(
            "dataset.loader.num_workers cannot be negative."
        )

    if model.get("architecture") != "siamese_unet":
        raise TrainingConfigurationError(
            "model.architecture must be 'siamese_unet'."
        )

    if model.get("shared_encoder") is not True:
        raise TrainingConfigurationError(
            "The Siamese encoder must remain weight-shared."
        )

    if model.get("fusion") != "absolute_difference":
        raise TrainingConfigurationError(
            "Week 4 requires absolute-difference feature fusion."
        )

    if int(model.get("output_channels", 0)) != 1:
        raise TrainingConfigurationError(
            "Binary change detection requires one output channel."
        )

    if model.get("output_activation") != "none":
        raise TrainingConfigurationError(
            "The model must return raw logits without an activation."
        )

    decoder_channels = tuple(
        int(value)
        for value in model.get("decoder_channels", ())
    )

    if len(decoder_channels) != 4:
        raise TrainingConfigurationError(
            "model.decoder_channels must contain four values."
        )

    if optimizer.get("name") != "adamw":
        raise TrainingConfigurationError(
            "Week 4 optimizer must be AdamW."
        )

    if scheduler.get("name") != "cosine_annealing":
        raise TrainingConfigurationError(
            "Week 4 scheduler must be cosine annealing."
        )

    if int(training.get("epochs", 0)) <= 0:
        raise TrainingConfigurationError(
            "training.epochs must be positive."
        )

    if int(
        training.get(
            "gradient_accumulation_steps",
            0,
        )
    ) <= 0:
        raise TrainingConfigurationError(
            "gradient_accumulation_steps must be positive."
        )

    if float(
        training.get(
            "maximum_gradient_norm",
            0.0,
        )
    ) <= 0:
        raise TrainingConfigurationError(
            "maximum_gradient_norm must be positive."
        )

    threshold = float(metrics.get("threshold", -1.0))

    if not 0.0 <= threshold <= 1.0:
        raise TrainingConfigurationError(
            "metrics.threshold must be between 0 and 1."
        )

    # These builders perform detailed validation of their own sections.
    build_loss_from_config(config)
    build_augmentation_from_config(config)


def resolve_device(
    requested_device: str,
) -> torch.device:
    """Resolve ``auto``, ``cpu`` or ``cuda`` to an available device."""
    normalized = requested_device.strip().lower()

    if normalized == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise TrainingConfigurationError(
                "CUDA was requested but is unavailable."
            )

        return torch.device("cuda")

    if normalized == "cpu":
        return torch.device("cpu")

    raise TrainingConfigurationError(
        "Device must be one of: auto, cpu or cuda."
    )


def configure_reproducibility(
    seed: int,
    deterministic_algorithms: bool,
    cudnn_benchmark: bool,
) -> None:
    """Configure Python, NumPy and PyTorch random generators."""
    if seed < 0:
        raise TrainingConfigurationError(
            "The project seed must be non-negative."
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(
        deterministic_algorithms,
        warn_only=True,
    )

    torch.backends.cudnn.deterministic = (
        deterministic_algorithms
    )
    torch.backends.cudnn.benchmark = bool(
        cudnn_benchmark
    )


def seed_data_worker(
    worker_id: int,
) -> None:
    """Seed NumPy and Python RNGs inside one DataLoader worker."""
    del worker_id

    worker_seed = torch.initial_seed() % (2**32)

    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_datasets(
    config: Mapping[str, Any],
) -> tuple[
    OSCDTrainingDataset,
    OSCDTrainingDataset,
]:
    """Build geographically separated training and validation datasets."""
    paths = require_mapping(
        config.get("paths"),
        "paths",
    )
    dataset_config = require_mapping(
        config.get("dataset"),
        "dataset",
    )

    augmentation: PairedGeometricAugmentation = (
        build_augmentation_from_config(config)
    )

    common_arguments = {
        "raw_root": Path(
            str(paths["oscd_raw_root"])
        ),
        "band_names": tuple(
            dataset_config["bands"]
        ),
        "patch_size": int(
            dataset_config["patch_size"]
        ),
        "stride": int(
            dataset_config["stride"]
        ),
        "reflectance_scale": float(
            dataset_config["reflectance_scale"]
        ),
        "clip_minimum": float(
            dataset_config["clip_minimum"]
        ),
        "clip_maximum": float(
            dataset_config["clip_maximum"]
        ),
    }

    training_dataset = OSCDTrainingDataset(
        **common_arguments,
        region_names=tuple(
            dataset_config["train_regions"]
        ),
        transform=augmentation,
    )

    validation_dataset = OSCDTrainingDataset(
        **common_arguments,
        region_names=tuple(
            dataset_config["validation_regions"]
        ),
        transform=None,
    )

    if set(training_dataset.region_names).intersection(
        validation_dataset.region_names
    ):
        raise TrainingConfigurationError(
            "Training and validation datasets overlap geographically."
        )

    return (
        training_dataset,
        validation_dataset,
    )


def create_data_loader(
    dataset: OSCDTrainingDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    drop_last: bool,
) -> DataLoader:
    """Create a deterministic OSCD DataLoader."""
    if batch_size <= 0:
        raise TrainingConfigurationError(
            "DataLoader batch size must be positive."
        )

    if num_workers < 0:
        raise TrainingConfigurationError(
            "DataLoader worker count cannot be negative."
        )

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader_arguments: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "generator": generator,
        "worker_init_fn": seed_data_worker,
    }

    if num_workers > 0:
        loader_arguments["persistent_workers"] = (
            persistent_workers
        )
        loader_arguments["prefetch_factor"] = (
            prefetch_factor
        )

    return DataLoader(
        **loader_arguments
    )


def build_data_loaders(
    config: Mapping[str, Any],
    device: torch.device,
    batch_size_override: int | None = None,
    validation_batch_size_override: int | None = None,
    num_workers_override: int | None = None,
) -> tuple[
    OSCDTrainingDataset,
    OSCDTrainingDataset,
    DataLoader,
    DataLoader,
]:
    """Build deterministic train and validation DataLoaders."""
    project = require_mapping(
        config.get("project"),
        "project",
    )
    dataset_config = require_mapping(
        config.get("dataset"),
        "dataset",
    )
    loader_config = require_mapping(
        dataset_config.get("loader"),
        "dataset.loader",
    )

    training_dataset, validation_dataset = build_datasets(
        config
    )

    seed = int(project["seed"])

    batch_size = (
        int(batch_size_override)
        if batch_size_override is not None
        else int(loader_config["batch_size"])
    )
    validation_batch_size = (
        int(validation_batch_size_override)
        if validation_batch_size_override is not None
        else int(
            loader_config["validation_batch_size"]
        )
    )
    num_workers = (
        int(num_workers_override)
        if num_workers_override is not None
        else int(loader_config["num_workers"])
    )

    use_pinned_memory = (
        bool(loader_config["pin_memory"])
        and device.type == "cuda"
    )

    training_loader = create_data_loader(
        dataset=training_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        seed=seed,
        pin_memory=use_pinned_memory,
        persistent_workers=bool(
            loader_config["persistent_workers"]
        ),
        prefetch_factor=int(
            loader_config["prefetch_factor"]
        ),
        drop_last=bool(
            loader_config[
                "drop_last_training_batch"
            ]
        ),
    )

    validation_loader = create_data_loader(
        dataset=validation_dataset,
        batch_size=validation_batch_size,
        num_workers=num_workers,
        shuffle=False,
        seed=seed + 1,
        pin_memory=use_pinned_memory,
        persistent_workers=bool(
            loader_config["persistent_workers"]
        ),
        prefetch_factor=int(
            loader_config["prefetch_factor"]
        ),
        drop_last=False,
    )

    return (
        training_dataset,
        validation_dataset,
        training_loader,
        validation_loader,
    )


def build_model(
    config: Mapping[str, Any],
    device: torch.device,
    disable_pretrained: bool = False,
) -> SiameseUNet:
    """Build the configured Siamese U-Net."""
    dataset_config = require_mapping(
        config.get("dataset"),
        "dataset",
    )
    model_config = require_mapping(
        config.get("model"),
        "model",
    )

    model = SiameseUNet(
        input_channels=int(
            dataset_config["input_channels"]
        ),
        band_names=tuple(
            dataset_config["bands"]
        ),
        pretrained_encoder=(
            bool(model_config["pretrained_encoder"])
            and not disable_pretrained
        ),
        decoder_channels=tuple(
            int(value)
            for value in model_config[
                "decoder_channels"
            ]
        ),
        head_channels=int(
            model_config["head_channels"]
        ),
        preferred_norm_groups=int(
            model_config["preferred_norm_groups"]
        ),
        dropout_probability=float(
            model_config["dropout_probability"]
        ),
    )

    return model.to(device)


def build_optimizer_and_scheduler(
    model: nn.Module,
    config: Mapping[str, Any],
) -> tuple[
    Optimizer,
    LRScheduler,
]:
    """Build AdamW and cosine annealing from YAML values."""
    optimizer_config = require_mapping(
        config.get("optimizer"),
        "optimizer",
    )
    scheduler_config = require_mapping(
        config.get("scheduler"),
        "scheduler",
    )
    training_config = require_mapping(
        config.get("training"),
        "training",
    )

    beta_values = tuple(
        float(value)
        for value in optimizer_config["betas"]
    )

    if len(beta_values) != 2:
        raise TrainingConfigurationError(
            "optimizer.betas must contain two values."
        )

    optimizer = AdamW(
        model.parameters(),
        lr=float(
            optimizer_config["learning_rate"]
        ),
        weight_decay=float(
            optimizer_config["weight_decay"]
        ),
        betas=(
            beta_values[0],
            beta_values[1],
        ),
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=int(
            training_config["epochs"]
        ),
        eta_min=float(
            scheduler_config[
                "minimum_learning_rate"
            ]
        ),
    )

    return (
        optimizer,
        scheduler,
    )


def build_training_runtime(
    config: Mapping[str, Any],
    device_override: str | None = None,
    batch_size_override: int | None = None,
    validation_batch_size_override: int | None = None,
    num_workers_override: int | None = None,
    disable_pretrained: bool = False,
) -> TrainingRuntime:
    """Construct every component required by the training loop."""
    validate_training_config(config)

    project_config = require_mapping(
        config.get("project"),
        "project",
    )
    training_config = require_mapping(
        config.get("training"),
        "training",
    )
    reproducibility_config = require_mapping(
        config.get("reproducibility"),
        "reproducibility",
    )

    configure_reproducibility(
        seed=int(project_config["seed"]),
        deterministic_algorithms=bool(
            reproducibility_config[
                "deterministic_algorithms"
            ]
        ),
        cudnn_benchmark=bool(
            reproducibility_config["cudnn_benchmark"]
        ),
    )

    requested_device = (
        device_override
        if device_override is not None
        else str(training_config["device"])
    )
    device = resolve_device(requested_device)

    (
        training_dataset,
        validation_dataset,
        training_loader,
        validation_loader,
    ) = build_data_loaders(
        config=config,
        device=device,
        batch_size_override=batch_size_override,
        validation_batch_size_override=(
            validation_batch_size_override
        ),
        num_workers_override=num_workers_override,
    )

    model = build_model(
        config=config,
        device=device,
        disable_pretrained=disable_pretrained,
    )
    criterion = build_loss_from_config(config).to(
        device
    )
    optimizer, scheduler = build_optimizer_and_scheduler(
        model=model,
        config=config,
    )

    mixed_precision_enabled = (
        bool(training_config["mixed_precision"])
        and device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        device.type,
        enabled=mixed_precision_enabled,
    )

    return TrainingRuntime(
        config=config,
        device=device,
        mixed_precision_enabled=mixed_precision_enabled,
        train_dataset=training_dataset,
        validation_dataset=validation_dataset,
        train_loader=training_loader,
        validation_loader=validation_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )


def move_batch_to_device(
    batch: Mapping[str, Any],
    device: torch.device,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
]:
    """Move one collated OSCD batch to the selected device."""
    before = batch["before"].to(
        device,
        non_blocking=device.type == "cuda",
    )
    after = batch["after"].to(
        device,
        non_blocking=device.type == "cuda",
    )
    mask = batch["mask"].to(
        device,
        non_blocking=device.type == "cuda",
    )

    return (
        before,
        after,
        mask,
    )


def calculate_tensor_metrics(
    logits: Tensor,
    targets: Tensor,
    threshold: float,
) -> ChangeMetrics:
    """Calculate positive-class metrics using the frozen Week 2 protocol."""
    predictions = (
        torch.sigmoid(logits) >= threshold
    ).to(
        dtype=torch.uint8
    )

    binary_targets = targets.to(
        dtype=torch.uint8
    )

    image_width = int(
        binary_targets.shape[-1]
    )

    target_array = (
        binary_targets
        .detach()
        .cpu()
        .numpy()
        .reshape(
            -1,
            image_width,
        )
    )
    prediction_array = (
        predictions
        .detach()
        .cpu()
        .numpy()
        .reshape(
            -1,
            image_width,
        )
    )

    return calculate_change_metrics(
        ground_truth=target_array,
        prediction=prediction_array,
        ground_truth_change_values=(1,),
        prediction_change_values=(1,),
        zero_division=0.0,
    )


def run_training_dry_run(
    runtime: TrainingRuntime,
) -> DryRunResult:
    """Run one optimizer update and one validation forward pass."""
    training_config = require_mapping(
        runtime.config.get("training"),
        "training",
    )
    metrics_config = require_mapping(
        runtime.config.get("metrics"),
        "metrics",
    )

    maximum_gradient_norm = float(
        training_config["maximum_gradient_norm"]
    )
    threshold = float(
        metrics_config["threshold"]
    )

    training_batch = next(
        iter(runtime.train_loader)
    )
    before, after, targets = move_batch_to_device(
        training_batch,
        runtime.device,
    )

    runtime.model.train()
    runtime.optimizer.zero_grad(
        set_to_none=True
    )

    with torch.amp.autocast(
        device_type=runtime.device.type,
        enabled=runtime.mixed_precision_enabled,
    ):
        logits = runtime.model(
            before,
            after,
        )
        training_breakdown = (
            runtime.criterion.compute_breakdown(
                logits=logits,
                targets=targets,
            )
        )

    runtime.scaler.scale(
        training_breakdown.total
    ).backward()

    runtime.scaler.unscale_(
        runtime.optimizer
    )

    gradient_norm_tensor = (
        torch.nn.utils.clip_grad_norm_(
            runtime.model.parameters(),
            max_norm=maximum_gradient_norm,
        )
    )

    runtime.scaler.step(
        runtime.optimizer
    )
    runtime.scaler.update()

    validation_batch = next(
        iter(runtime.validation_loader)
    )
    (
        validation_before,
        validation_after,
        validation_targets,
    ) = move_batch_to_device(
        validation_batch,
        runtime.device,
    )

    runtime.model.eval()

    with torch.inference_mode():
        with torch.amp.autocast(
            device_type=runtime.device.type,
            enabled=runtime.mixed_precision_enabled,
        ):
            validation_logits = runtime.model(
                validation_before,
                validation_after,
            )
            validation_breakdown = (
                runtime.criterion.compute_breakdown(
                    logits=validation_logits,
                    targets=validation_targets,
                )
            )

    validation_metrics = calculate_tensor_metrics(
        logits=validation_logits,
        targets=validation_targets,
        threshold=threshold,
    )

    return DryRunResult(
        training_total_loss=float(
            training_breakdown.total.detach().item()
        ),
        training_dice_loss=float(
            training_breakdown.dice.detach().item()
        ),
        training_focal_loss=float(
            training_breakdown.focal.detach().item()
        ),
        gradient_norm=float(
            gradient_norm_tensor.detach().cpu().item()
        ),
        validation_total_loss=float(
            validation_breakdown.total.detach().item()
        ),
        validation_dice_loss=float(
            validation_breakdown.dice.detach().item()
        ),
        validation_focal_loss=float(
            validation_breakdown.focal.detach().item()
        ),
        validation_metrics=validation_metrics,
        training_batch_shape=tuple(before.shape),
        validation_batch_shape=tuple(
            validation_before.shape
        ),
    )



@dataclass(frozen=True)
class ChangeClassMetrics:
    """Globally aggregated positive-class segmentation metrics."""

    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    precision: float
    recall: float
    f1: float
    iou: float
    accuracy: float


@dataclass(frozen=True)
class EpochResult:
    """Losses, metrics and optimizer diagnostics for one epoch."""

    total_loss: float
    dice_loss: float
    focal_loss: float
    metrics: ChangeClassMetrics
    batches: int
    samples: int
    optimizer_steps: int
    mean_gradient_norm: float
    maximum_gradient_norm: float


class BinaryMetricAccumulator:
    """Accumulate binary change-detection confusion counts globally.

    Metrics are calculated after all selected batches have been processed.
    This is preferable to averaging per-batch F1 or IoU because batches can
    contain very different numbers of changed pixels.
    """

    def __init__(
        self,
        threshold: float,
    ) -> None:
        """Initialize an empty positive-class metric accumulator."""
        if not 0.0 <= threshold <= 1.0:
            raise TrainingConfigurationError(
                "Metric threshold must be between 0 and 1."
            )

        self.threshold = float(
            threshold
        )

        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0
        self.true_negative = 0

    def update(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> None:
        """Add confusion counts from one batch of raw logits."""
        if logits.shape != targets.shape:
            raise TrainingConfigurationError(
                "Metric logits and targets must have identical shapes; "
                f"received {tuple(logits.shape)} and {tuple(targets.shape)}."
            )

        if not logits.is_floating_point():
            raise TrainingConfigurationError(
                "Metric logits must be floating-point."
            )

        if not targets.is_floating_point():
            raise TrainingConfigurationError(
                "Metric targets must be floating-point."
            )

        if not bool(
            torch.isfinite(
                logits
            ).all().item()
        ):
            raise TrainingConfigurationError(
                "Metric logits contain non-finite values."
            )

        if not bool(
            torch.isfinite(
                targets
            ).all().item()
        ):
            raise TrainingConfigurationError(
                "Metric targets contain non-finite values."
            )

        valid_targets = torch.logical_or(
            targets == 0,
            targets == 1,
        )

        if not bool(
            valid_targets.all().item()
        ):
            raise TrainingConfigurationError(
                "Metric targets must contain only 0 and 1."
            )

        predictions = (
            torch.sigmoid(
                logits
            )
            >= self.threshold
        )

        ground_truth = targets > 0

        self.true_positive += int(
            torch.logical_and(
                predictions,
                ground_truth,
            ).sum().item()
        )

        self.false_positive += int(
            torch.logical_and(
                predictions,
                torch.logical_not(
                    ground_truth
                ),
            ).sum().item()
        )

        self.false_negative += int(
            torch.logical_and(
                torch.logical_not(
                    predictions
                ),
                ground_truth,
            ).sum().item()
        )

        self.true_negative += int(
            torch.logical_and(
                torch.logical_not(
                    predictions
                ),
                torch.logical_not(
                    ground_truth
                ),
            ).sum().item()
        )

    @staticmethod
    def _safe_divide(
        numerator: int | float,
        denominator: int | float,
    ) -> float:
        """Divide safely using the frozen zero-division value of zero."""
        if denominator == 0:
            return 0.0

        return float(
            numerator
            / denominator
        )

    def compute(
        self,
    ) -> ChangeClassMetrics:
        """Calculate positive change-class metrics from global counts."""
        precision = self._safe_divide(
            self.true_positive,
            (
                self.true_positive
                + self.false_positive
            ),
        )

        recall = self._safe_divide(
            self.true_positive,
            (
                self.true_positive
                + self.false_negative
            ),
        )

        f1 = self._safe_divide(
            2.0
            * precision
            * recall,
            precision + recall,
        )

        iou = self._safe_divide(
            self.true_positive,
            (
                self.true_positive
                + self.false_positive
                + self.false_negative
            ),
        )

        total_pixels = (
            self.true_positive
            + self.false_positive
            + self.false_negative
            + self.true_negative
        )

        accuracy = self._safe_divide(
            (
                self.true_positive
                + self.true_negative
            ),
            total_pixels,
        )

        return ChangeClassMetrics(
            true_positive=self.true_positive,
            false_positive=self.false_positive,
            false_negative=self.false_negative,
            true_negative=self.true_negative,
            precision=precision,
            recall=recall,
            f1=f1,
            iou=iou,
            accuracy=accuracy,
        )


def resolve_batch_limit(
    loader: DataLoader,
    maximum_batches: int | None,
) -> int:
    """Resolve an optional smoke-test batch limit."""
    available_batches = len(
        loader
    )

    if available_batches <= 0:
        raise TrainingConfigurationError(
            "The DataLoader contains no batches."
        )

    if maximum_batches is None:
        return available_batches

    if maximum_batches <= 0:
        raise TrainingConfigurationError(
            "maximum_batches must be positive when provided."
        )

    return min(
        available_batches,
        maximum_batches,
    )


def train_one_epoch(
    runtime: TrainingRuntime,
    maximum_batches: int | None = None,
) -> EpochResult:
    """Train the model for one complete or limited epoch.

    Gradient accumulation is implemented in exact groups. A final partial
    group is divided by its actual number of batches rather than by the full
    configured accumulation count.
    """
    training_config = require_mapping(
        runtime.config.get(
            "training"
        ),
        "training",
    )
    metrics_config = require_mapping(
        runtime.config.get(
            "metrics"
        ),
        "metrics",
    )

    accumulation_steps = int(
        training_config[
            "gradient_accumulation_steps"
        ]
    )
    maximum_norm = float(
        training_config[
            "maximum_gradient_norm"
        ]
    )
    threshold = float(
        metrics_config[
            "threshold"
        ]
    )

    if accumulation_steps <= 0:
        raise TrainingConfigurationError(
            "gradient_accumulation_steps must be positive."
        )

    if maximum_norm <= 0:
        raise TrainingConfigurationError(
            "maximum_gradient_norm must be positive."
        )

    selected_batches = resolve_batch_limit(
        loader=runtime.train_loader,
        maximum_batches=maximum_batches,
    )

    metric_accumulator = BinaryMetricAccumulator(
        threshold=threshold
    )

    runtime.model.train()
    runtime.optimizer.zero_grad(
        set_to_none=True
    )

    weighted_total_loss = 0.0
    weighted_dice_loss = 0.0
    weighted_focal_loss = 0.0

    samples_processed = 0
    batches_processed = 0
    optimizer_steps = 0
    gradient_norms: list[float] = []

    for batch_index, batch in enumerate(
        runtime.train_loader
    ):
        if batch_index >= selected_batches:
            break

        before, after, targets = move_batch_to_device(
            batch=batch,
            device=runtime.device,
        )

        batch_size = int(
            before.shape[0]
        )

        if batch_size <= 0:
            raise TrainingConfigurationError(
                "Encountered an empty training batch."
            )

        group_start = (
            batch_index
            // accumulation_steps
        ) * accumulation_steps

        group_end = min(
            group_start
            + accumulation_steps,
            selected_batches,
        )

        group_size = (
            group_end
            - group_start
        )

        with torch.amp.autocast(
            device_type=runtime.device.type,
            enabled=runtime.mixed_precision_enabled,
        ):
            logits = runtime.model(
                before,
                after,
            )

            breakdown = (
                runtime.criterion.compute_breakdown(
                    logits=logits,
                    targets=targets,
                )
            )

            backward_loss = (
                breakdown.total
                / float(
                    group_size
                )
            )

        if not bool(
            torch.isfinite(
                breakdown.total
            ).item()
        ):
            raise TrainingConfigurationError(
                "Non-finite training loss detected."
            )

        runtime.scaler.scale(
            backward_loss
        ).backward()

        weighted_total_loss += (
            float(
                breakdown.total.detach().item()
            )
            * batch_size
        )
        weighted_dice_loss += (
            float(
                breakdown.dice.detach().item()
            )
            * batch_size
        )
        weighted_focal_loss += (
            float(
                breakdown.focal.detach().item()
            )
            * batch_size
        )

        metric_accumulator.update(
            logits=logits.detach(),
            targets=targets,
        )

        samples_processed += batch_size
        batches_processed += 1

        should_step = (
            batch_index + 1
            == group_end
        )

        if should_step:
            runtime.scaler.unscale_(
                runtime.optimizer
            )

            gradient_norm_tensor = (
                torch.nn.utils.clip_grad_norm_(
                    runtime.model.parameters(),
                    max_norm=maximum_norm,
                )
            )

            gradient_norm = float(
                gradient_norm_tensor
                .detach()
                .cpu()
                .item()
            )

            if not np.isfinite(
                gradient_norm
            ):
                raise TrainingConfigurationError(
                    "Non-finite gradient norm detected."
                )

            runtime.scaler.step(
                runtime.optimizer
            )
            runtime.scaler.update()

            runtime.optimizer.zero_grad(
                set_to_none=True
            )

            gradient_norms.append(
                gradient_norm
            )
            optimizer_steps += 1

    if samples_processed <= 0:
        raise TrainingConfigurationError(
            "No training samples were processed."
        )

    if optimizer_steps <= 0:
        raise TrainingConfigurationError(
            "No optimizer step was completed."
        )

    return EpochResult(
        total_loss=(
            weighted_total_loss
            / samples_processed
        ),
        dice_loss=(
            weighted_dice_loss
            / samples_processed
        ),
        focal_loss=(
            weighted_focal_loss
            / samples_processed
        ),
        metrics=metric_accumulator.compute(),
        batches=batches_processed,
        samples=samples_processed,
        optimizer_steps=optimizer_steps,
        mean_gradient_norm=float(
            sum(
                gradient_norms
            )
            / len(
                gradient_norms
            )
        ),
        maximum_gradient_norm=max(
            gradient_norms
        ),
    )


def validate_one_epoch(
    runtime: TrainingRuntime,
    maximum_batches: int | None = None,
) -> EpochResult:
    """Evaluate one complete or limited validation epoch.

    Validation never applies augmentation, never computes gradients and never
    updates model, optimizer, scaler or scheduler state.
    """
    metrics_config = require_mapping(
        runtime.config.get(
            "metrics"
        ),
        "metrics",
    )

    threshold = float(
        metrics_config[
            "threshold"
        ]
    )

    selected_batches = resolve_batch_limit(
        loader=runtime.validation_loader,
        maximum_batches=maximum_batches,
    )

    metric_accumulator = BinaryMetricAccumulator(
        threshold=threshold
    )

    weighted_total_loss = 0.0
    weighted_dice_loss = 0.0
    weighted_focal_loss = 0.0

    samples_processed = 0
    batches_processed = 0

    runtime.model.eval()

    with torch.inference_mode():
        for batch_index, batch in enumerate(
            runtime.validation_loader
        ):
            if batch_index >= selected_batches:
                break

            before, after, targets = move_batch_to_device(
                batch=batch,
                device=runtime.device,
            )

            batch_size = int(
                before.shape[0]
            )

            if batch_size <= 0:
                raise TrainingConfigurationError(
                    "Encountered an empty validation batch."
                )

            with torch.amp.autocast(
                device_type=runtime.device.type,
                enabled=runtime.mixed_precision_enabled,
            ):
                logits = runtime.model(
                    before,
                    after,
                )

                breakdown = (
                    runtime.criterion.compute_breakdown(
                        logits=logits,
                        targets=targets,
                    )
                )

            if not bool(
                torch.isfinite(
                    breakdown.total
                ).item()
            ):
                raise TrainingConfigurationError(
                    "Non-finite validation loss detected."
                )

            weighted_total_loss += (
                float(
                    breakdown.total.item()
                )
                * batch_size
            )
            weighted_dice_loss += (
                float(
                    breakdown.dice.item()
                )
                * batch_size
            )
            weighted_focal_loss += (
                float(
                    breakdown.focal.item()
                )
                * batch_size
            )

            metric_accumulator.update(
                logits=logits,
                targets=targets,
            )

            samples_processed += batch_size
            batches_processed += 1

    if samples_processed <= 0:
        raise TrainingConfigurationError(
            "No validation samples were processed."
        )

    return EpochResult(
        total_loss=(
            weighted_total_loss
            / samples_processed
        ),
        dice_loss=(
            weighted_dice_loss
            / samples_processed
        ),
        focal_loss=(
            weighted_focal_loss
            / samples_processed
        ),
        metrics=metric_accumulator.compute(),
        batches=batches_processed,
        samples=samples_processed,
        optimizer_steps=0,
        mean_gradient_norm=0.0,
        maximum_gradient_norm=0.0,
    )


@dataclass(frozen=True)
class EarlyStoppingDecision:
    """Result of evaluating one validation metric."""

    improved: bool
    should_stop: bool
    best_metric: float
    epochs_without_improvement: int


@dataclass(frozen=True)
class ResumeState:
    """Training position restored from a checkpoint."""

    next_epoch: int
    best_metric: float
    epochs_without_improvement: int
    monitor_name: str
    monitor_value: float


@dataclass(frozen=True)
class CheckpointUpdate:
    """Files and state produced after one completed epoch."""

    last_checkpoint: Path
    best_checkpoint: Path | None
    monitor_name: str
    monitor_value: float
    improved: bool
    should_stop: bool
    best_metric: float
    epochs_without_improvement: int


class EarlyStopping:
    """Track validation improvement using a fixed metric and mode."""

    def __init__(
        self,
        monitor: str,
        mode: str,
        patience: int,
        minimum_delta: float,
        best_metric: float | None = None,
        epochs_without_improvement: int = 0,
    ) -> None:
        """Initialize early stopping.

        Args:
            monitor: Name of the validation metric.
            mode: ``max`` for F1/IoU or ``min`` for validation loss.
            patience: Consecutive non-improving epochs allowed.
            minimum_delta: Minimum change required to count as improvement.
            best_metric: Optional metric restored from a checkpoint.
            epochs_without_improvement: Optional restored counter.
        """
        normalized_mode = mode.strip().lower()

        if normalized_mode not in {
            "max",
            "min",
        }:
            raise TrainingConfigurationError(
                "Early-stopping mode must be 'max' or 'min'."
            )

        if patience <= 0:
            raise TrainingConfigurationError(
                "Early-stopping patience must be positive."
            )

        if minimum_delta < 0:
            raise TrainingConfigurationError(
                "Early-stopping minimum_delta cannot be negative."
            )

        if epochs_without_improvement < 0:
            raise TrainingConfigurationError(
                "epochs_without_improvement cannot be negative."
            )

        if best_metric is not None and not np.isfinite(
            best_metric
        ):
            raise TrainingConfigurationError(
                "Restored best metric must be finite."
            )

        self.monitor = monitor
        self.mode = normalized_mode
        self.patience = int(
            patience
        )
        self.minimum_delta = float(
            minimum_delta
        )
        self.best_metric = best_metric
        self.epochs_without_improvement = int(
            epochs_without_improvement
        )

    def _is_improvement(
        self,
        metric_value: float,
    ) -> bool:
        """Return whether a value exceeds the frozen improvement rule."""
        if self.best_metric is None:
            return True

        if self.mode == "max":
            return (
                metric_value
                > self.best_metric
                + self.minimum_delta
            )

        return (
            metric_value
            < self.best_metric
            - self.minimum_delta
        )

    def update(
        self,
        metric_value: float,
    ) -> EarlyStoppingDecision:
        """Update state after one validation epoch."""
        value = float(
            metric_value
        )

        if not np.isfinite(
            value
        ):
            raise TrainingConfigurationError(
                "Early-stopping metric must be finite."
            )

        improved = self._is_improvement(
            value
        )

        if improved:
            self.best_metric = value
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        if self.best_metric is None:
            raise TrainingConfigurationError(
                "Early stopping failed to initialize its best metric."
            )

        should_stop = (
            self.epochs_without_improvement
            >= self.patience
        )

        return EarlyStoppingDecision(
            improved=improved,
            should_stop=should_stop,
            best_metric=float(
                self.best_metric
            ),
            epochs_without_improvement=(
                self.epochs_without_improvement
            ),
        )


def build_early_stopping_from_config(
    config: Mapping[str, Any],
    best_metric: float | None = None,
    epochs_without_improvement: int = 0,
) -> EarlyStopping:
    """Build early stopping from the frozen training configuration."""
    training_config = require_mapping(
        config.get(
            "training"
        ),
        "training",
    )
    early_config = require_mapping(
        training_config.get(
            "early_stopping"
        ),
        "training.early_stopping",
    )
    checkpoint_config = require_mapping(
        config.get(
            "checkpointing"
        ),
        "checkpointing",
    )

    if early_config.get(
        "enabled"
    ) is not True:
        raise TrainingConfigurationError(
            "Week 4 requires early stopping to remain enabled."
        )

    early_monitor = str(
        early_config[
            "monitor"
        ]
    )
    checkpoint_monitor = str(
        checkpoint_config[
            "monitor"
        ]
    )

    if early_monitor != checkpoint_monitor:
        raise TrainingConfigurationError(
            "Early stopping and checkpointing must monitor the same metric."
        )

    early_mode = str(
        early_config[
            "mode"
        ]
    )
    checkpoint_mode = str(
        checkpoint_config[
            "mode"
        ]
    )

    if early_mode != checkpoint_mode:
        raise TrainingConfigurationError(
            "Early stopping and checkpointing must use the same mode."
        )

    return EarlyStopping(
        monitor=early_monitor,
        mode=early_mode,
        patience=int(
            early_config[
                "patience"
            ]
        ),
        minimum_delta=float(
            early_config[
                "minimum_delta"
            ]
        ),
        best_metric=best_metric,
        epochs_without_improvement=(
            epochs_without_improvement
        ),
    )


def extract_validation_monitor(
    result: EpochResult,
    monitor: str,
) -> float:
    """Extract one supported validation value from an epoch result."""
    monitor_values = {
        "validation_total_loss": result.total_loss,
        "validation_loss": result.total_loss,
        "validation_dice_loss": result.dice_loss,
        "validation_focal_loss": result.focal_loss,
        "validation_precision": result.metrics.precision,
        "validation_recall": result.metrics.recall,
        "validation_f1": result.metrics.f1,
        "validation_iou": result.metrics.iou,
        "validation_accuracy": result.metrics.accuracy,
    }

    if monitor not in monitor_values:
        raise TrainingConfigurationError(
            "Unsupported checkpoint monitor "
            f"'{monitor}'. Supported values are "
            f"{sorted(monitor_values)}."
        )

    value = float(
        monitor_values[
            monitor
        ]
    )

    if not np.isfinite(
        value
    ):
        raise TrainingConfigurationError(
            f"Validation monitor '{monitor}' is not finite."
        )

    return value


def create_config_fingerprint(
    config: Mapping[str, Any],
) -> str:
    """Create a deterministic SHA-256 fingerprint for the YAML config."""
    canonical_config = json.dumps(
        config,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        default=str,
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        canonical_config
    ).hexdigest()


def capture_random_states() -> dict[str, Any]:
    """Capture Python, NumPy, CPU and optional CUDA RNG states."""
    states: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }

    if torch.cuda.is_available():
        states[
            "torch_cuda"
        ] = torch.cuda.get_rng_state_all()

    return states


def normalize_rng_state_tensor(
    state: Any,
    field_name: str,
) -> Tensor:
    if not isinstance(
        state,
        Tensor,
    ):
        raise TrainingConfigurationError(
            f"{field_name} RNG state must be a tensor."
        )

    normalized = (
        state
        .detach()
        .to(
            device="cpu",
            dtype=torch.uint8,
        )
        .contiguous()
    )

    if normalized.ndim != 1:
        raise TrainingConfigurationError(
            f"{field_name} RNG state must be one-dimensional."
        )

    return normalized


def restore_random_states(
    states: Mapping[str, Any],
) -> None:
    random.setstate(
        states[
            "python"
        ]
    )
    np.random.set_state(
        states[
            "numpy"
        ]
    )

    torch.set_rng_state(
        normalize_rng_state_tensor(
            states[
                "torch_cpu"
            ],
            "torch_cpu",
        )
    )

    cuda_states = states.get(
        "torch_cuda"
    )

    if (
        cuda_states is not None
        and torch.cuda.is_available()
    ):
        if not isinstance(
            cuda_states,
            (
                list,
                tuple,
            ),
        ):
            raise TrainingConfigurationError(
                "torch_cuda RNG states must be a sequence."
            )

        normalized_cuda_states = [
            normalize_rng_state_tensor(
                state,
                f"torch_cuda[{index}]",
            )
            for index, state
            in enumerate(
                cuda_states
            )
        ]

        if (
            len(
                normalized_cuda_states
            )
            != torch.cuda.device_count()
        ):
            raise TrainingConfigurationError(
                "The checkpoint CUDA RNG-state count does not match "
                "the available CUDA device count."
            )

        torch.cuda.set_rng_state_all(
            normalized_cuda_states
        )


def resolve_checkpoint_paths(
    config: Mapping[str, Any],
) -> tuple[
    Path,
    Path,
]:
    """Resolve the configured best and last checkpoint paths."""
    paths_config = require_mapping(
        config.get(
            "paths"
        ),
        "paths",
    )
    checkpoint_config = require_mapping(
        config.get(
            "checkpointing"
        ),
        "checkpointing",
    )

    checkpoint_directory = Path(
        str(
            paths_config[
                "checkpoint_directory"
            ]
        )
    )

    best_filename = str(
        checkpoint_config[
            "filename"
        ]
    ).strip()

    if not best_filename:
        raise TrainingConfigurationError(
            "checkpointing.filename cannot be empty."
        )

    return (
        checkpoint_directory
        / best_filename,
        checkpoint_directory
        / "last_model.pt",
    )


def save_training_checkpoint(
    path: Path,
    runtime: TrainingRuntime,
    epoch: int,
    training_result: EpochResult,
    validation_result: EpochResult,
    monitor_name: str,
    monitor_value: float,
    best_metric: float,
    epochs_without_improvement: int,
) -> None:
    """Atomically save complete training state for exact resumption."""
    if epoch <= 0:
        raise TrainingConfigurationError(
            "Checkpoint epoch must be positive."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "schema_version": 1,
        "epoch": int(
            epoch
        ),
        "monitor_name": monitor_name,
        "monitor_value": float(
            monitor_value
        ),
        "best_metric": float(
            best_metric
        ),
        "epochs_without_improvement": int(
            epochs_without_improvement
        ),
        "config": dict(
            runtime.config
        ),
        "config_fingerprint": create_config_fingerprint(
            runtime.config
        ),
        "model_state_dict": runtime.model.state_dict(),
        "optimizer_state_dict": runtime.optimizer.state_dict(),
        "scheduler_state_dict": runtime.scheduler.state_dict(),
        "scaler_state_dict": runtime.scaler.state_dict(),
        "training_result": asdict(
            training_result
        ),
        "validation_result": asdict(
            validation_result
        ),
        "random_states": capture_random_states(),
    }

    temporary_file = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temporary_path = Path(
        temporary_file.name
    )
    temporary_file.close()

    try:
        torch.save(
            payload,
            temporary_path,
        )

        os.replace(
            temporary_path,
            path,
        )

    finally:
        temporary_path.unlink(
            missing_ok=True
        )


def update_epoch_checkpoints(
    runtime: TrainingRuntime,
    epoch: int,
    training_result: EpochResult,
    validation_result: EpochResult,
    early_stopping: EarlyStopping,
) -> CheckpointUpdate:
    """Update early stopping and save last/best checkpoints."""
    checkpoint_config = require_mapping(
        runtime.config.get(
            "checkpointing"
        ),
        "checkpointing",
    )

    monitor_name = str(
        checkpoint_config[
            "monitor"
        ]
    )
    monitor_value = extract_validation_monitor(
        result=validation_result,
        monitor=monitor_name,
    )

    decision = early_stopping.update(
        monitor_value
    )

    best_path, last_path = resolve_checkpoint_paths(
        runtime.config
    )

    if bool(
        checkpoint_config.get(
            "save_last",
            True,
        )
    ):
        save_training_checkpoint(
            path=last_path,
            runtime=runtime,
            epoch=epoch,
            training_result=training_result,
            validation_result=validation_result,
            monitor_name=monitor_name,
            monitor_value=monitor_value,
            best_metric=decision.best_metric,
            epochs_without_improvement=(
                decision.epochs_without_improvement
            ),
        )

    saved_best_path: Path | None = None

    if decision.improved:
        save_training_checkpoint(
            path=best_path,
            runtime=runtime,
            epoch=epoch,
            training_result=training_result,
            validation_result=validation_result,
            monitor_name=monitor_name,
            monitor_value=monitor_value,
            best_metric=decision.best_metric,
            epochs_without_improvement=(
                decision.epochs_without_improvement
            ),
        )
        saved_best_path = best_path

    return CheckpointUpdate(
        last_checkpoint=last_path,
        best_checkpoint=saved_best_path,
        monitor_name=monitor_name,
        monitor_value=monitor_value,
        improved=decision.improved,
        should_stop=decision.should_stop,
        best_metric=decision.best_metric,
        epochs_without_improvement=(
            decision.epochs_without_improvement
        ),
    )


def load_training_checkpoint(
    path: Path,
    runtime: TrainingRuntime,
    strict_config: bool = True,
    restore_rng: bool = True,
) -> ResumeState:
    """Restore complete runtime state from a GeoWatch checkpoint."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {path}"
        )

    payload = torch.load(
        path,
        map_location=runtime.device,
        weights_only=False,
    )

    if not isinstance(
        payload,
        Mapping,
    ):
        raise TrainingConfigurationError(
            "Checkpoint payload must be a mapping."
        )

    required_keys = {
        "schema_version",
        "epoch",
        "monitor_name",
        "monitor_value",
        "best_metric",
        "epochs_without_improvement",
        "config_fingerprint",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "random_states",
    }

    missing_keys = required_keys.difference(
        payload
    )

    if missing_keys:
        raise TrainingConfigurationError(
            "Checkpoint is missing required fields: "
            f"{sorted(missing_keys)}"
        )

    if int(
        payload[
            "schema_version"
        ]
    ) != 1:
        raise TrainingConfigurationError(
            "Unsupported checkpoint schema version."
        )

    expected_fingerprint = create_config_fingerprint(
        runtime.config
    )
    stored_fingerprint = str(
        payload[
            "config_fingerprint"
        ]
    )

    if (
        strict_config
        and stored_fingerprint
        != expected_fingerprint
    ):
        raise TrainingConfigurationError(
            "Checkpoint configuration does not match the active "
            "training configuration."
        )

    runtime.model.load_state_dict(
        payload[
            "model_state_dict"
        ],
        strict=True,
    )
    runtime.optimizer.load_state_dict(
        payload[
            "optimizer_state_dict"
        ]
    )
    runtime.scheduler.load_state_dict(
        payload[
            "scheduler_state_dict"
        ]
    )
    runtime.scaler.load_state_dict(
        payload[
            "scaler_state_dict"
        ]
    )

    if restore_rng:
        restore_random_states(
            payload[
                "random_states"
            ]
        )

    epoch = int(
        payload[
            "epoch"
        ]
    )

    if epoch <= 0:
        raise TrainingConfigurationError(
            "Checkpoint epoch must be positive."
        )

    return ResumeState(
        next_epoch=epoch + 1,
        best_metric=float(
            payload[
                "best_metric"
            ]
        ),
        epochs_without_improvement=int(
            payload[
                "epochs_without_improvement"
            ]
        ),
        monitor_name=str(
            payload[
                "monitor_name"
            ]
        ),
        monitor_value=float(
            payload[
                "monitor_value"
            ]
        ),
    )


@dataclass(frozen=True)
class TrainingLoopSummary:
    """Summary of one fresh or resumed multi-epoch run."""

    start_epoch: int
    final_epoch: int
    epochs_completed: int
    stopped_early: bool
    best_metric: float
    last_checkpoint: Path
    best_checkpoint: Path | None
    history_path: Path
    resolved_config_path: Path
    wandb_enabled: bool
    wandb_run_id: str | None
    wandb_run_name: str | None


def resolve_run_log_directory(
    config: Mapping[str, Any],
) -> Path:
    """Resolve an experiment-specific log directory."""
    paths_config = require_mapping(
        config.get(
            "paths"
        ),
        "paths",
    )
    project_config = require_mapping(
        config.get(
            "project"
        ),
        "project",
    )

    experiment_name = str(
        project_config.get(
            "experiment_name",
            "training",
        )
    ).strip()

    if not experiment_name:
        raise TrainingConfigurationError(
            "project.experiment_name cannot be empty."
        )

    return (
        Path(
            str(
                paths_config[
                    "log_directory"
                ]
            )
        )
        / experiment_name
    )


def write_resolved_config(
    config: Mapping[str, Any],
    output_path: Path,
) -> None:
    """Atomically save the exact configuration used by a run."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    temporary_path.write_text(
        yaml.safe_dump(
            dict(
                config
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    os.replace(
        temporary_path,
        output_path,
    )


def append_json_record(
    path: Path,
    record: Mapping[str, Any],
) -> None:
    """Append one deterministic JSON record to a JSONL history."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    serialized_record = json.dumps(
        dict(
            record
        ),
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    )

    with path.open(
        "a",
        encoding="utf-8",
        newline="\n",
    ) as history_file:
        history_file.write(
            serialized_record
            + "\n"
        )
        history_file.flush()
        os.fsync(
            history_file.fileno()
        )


def build_epoch_history_record(
    epoch: int,
    learning_rate_before_scheduler: float,
    learning_rate_after_scheduler: float,
    training_result: EpochResult,
    validation_result: EpochResult,
    checkpoint_update: CheckpointUpdate,
) -> dict[str, Any]:
    """Build one serializable epoch-history record."""
    return {
        "epoch": int(
            epoch
        ),
        "learning_rate_before_scheduler": float(
            learning_rate_before_scheduler
        ),
        "learning_rate_after_scheduler": float(
            learning_rate_after_scheduler
        ),
        "training": asdict(
            training_result
        ),
        "validation": asdict(
            validation_result
        ),
        "checkpoint": {
            "monitor_name": (
                checkpoint_update.monitor_name
            ),
            "monitor_value": (
                checkpoint_update.monitor_value
            ),
            "best_metric": (
                checkpoint_update.best_metric
            ),
            "improved": (
                checkpoint_update.improved
            ),
            "should_stop": (
                checkpoint_update.should_stop
            ),
            "epochs_without_improvement": (
                checkpoint_update
                .epochs_without_improvement
            ),
            "last_checkpoint": str(
                checkpoint_update.last_checkpoint
            ),
            "best_checkpoint": (
                str(
                    checkpoint_update.best_checkpoint
                )
                if checkpoint_update.best_checkpoint
                is not None
                else None
            ),
        },
        "evaluation_protocol": {
            "positive_class": "change",
            "official_test_regions_accessed": False,
            "official_test_labels_accessed": False,
        },
    }




def resolve_wandb_mode(
    config: Mapping[str, Any],
    mode_override: str | None = None,
) -> str:
    """Resolve whether W&B runs online, offline or disabled."""
    tracking_value = config.get(
        "tracking"
    )

    if tracking_value is None:
        return "disabled"

    tracking_config = require_mapping(
        tracking_value,
        "tracking",
    )

    if tracking_config.get(
        "enabled",
        False,
    ) is not True:
        return "disabled"

    selected_mode = (
        mode_override
        if mode_override is not None
        else str(
            tracking_config.get(
                "mode",
                "online",
            )
        )
    ).strip().lower()

    if selected_mode not in {
        "online",
        "offline",
        "disabled",
    }:
        raise TrainingConfigurationError(
            "W&B mode must be online, offline or disabled."
        )

    return selected_mode


def initialize_wandb_run(
    config: Mapping[str, Any],
    run_log_directory: Path,
    mode_override: str | None = None,
    resume_path: Path | None = None,
) -> Any | None:
    """Initialize one W&B run from the tracking configuration."""
    mode = resolve_wandb_mode(
        config=config,
        mode_override=mode_override,
    )

    if mode == "disabled":
        return None

    tracking_config = require_mapping(
        config.get(
            "tracking"
        ),
        "tracking",
    )
    project_config = require_mapping(
        config.get(
            "project"
        ),
        "project",
    )

    project_name = str(
        tracking_config.get(
            "project",
            "",
        )
    ).strip()

    if not project_name:
        raise TrainingConfigurationError(
            "tracking.project cannot be empty."
        )

    experiment_name = str(
        project_config.get(
            "experiment_name",
            "geowatch-training",
        )
    ).strip()

    entity_value = tracking_config.get(
        "entity"
    )

    entity = (
        str(
            entity_value
        )
        if entity_value not in {
            None,
            "",
        }
        else None
    )

    tracking_snapshot = dict(
        config
    )
    tracking_snapshot[
        "runtime"
    ] = {
        "resume_checkpoint": (
            str(
                resume_path
            )
            if resume_path is not None
            else None
        ),
        "official_test_regions_accessed": False,
        "official_test_labels_accessed": False,
    }

    run_log_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    run = wandb.init(
        project=project_name,
        entity=entity,
        name=experiment_name,
        group=str(
            tracking_config.get(
                "group",
                "week4",
            )
        ),
        job_type=str(
            tracking_config.get(
                "job_type",
                "training",
            )
        ),
        mode=mode,
        dir=str(
            run_log_directory
        ),
        config=tracking_snapshot,
        tags=[
            "week4",
            "oscd",
            "siamese-unet",
            "dice-focal",
            "positive-change-class",
        ],
    )

    if run is None:
        raise TrainingConfigurationError(
            "wandb.init() did not return a run."
        )

    run.define_metric(
        "epoch"
    )
    run.define_metric(
        "*",
        step_metric="epoch",
    )

    return run


def build_epoch_wandb_payload(
    epoch: int,
    learning_rate_before_scheduler: float,
    learning_rate_after_scheduler: float,
    training_result: EpochResult,
    validation_result: EpochResult,
    checkpoint_update: CheckpointUpdate,
) -> dict[str, Any]:
    """Build the scalar W&B payload for one epoch."""
    return {
        "epoch": int(
            epoch
        ),
        "optimizer/learning_rate": float(
            learning_rate_before_scheduler
        ),
        "optimizer/next_learning_rate": float(
            learning_rate_after_scheduler
        ),
        "training/total_loss": (
            training_result.total_loss
        ),
        "training/dice_loss": (
            training_result.dice_loss
        ),
        "training/focal_loss": (
            training_result.focal_loss
        ),
        "training/precision": (
            training_result.metrics.precision
        ),
        "training/recall": (
            training_result.metrics.recall
        ),
        "training/f1": (
            training_result.metrics.f1
        ),
        "training/iou": (
            training_result.metrics.iou
        ),
        "training/accuracy_secondary": (
            training_result.metrics.accuracy
        ),
        "training/mean_gradient_norm": (
            training_result.mean_gradient_norm
        ),
        "training/maximum_gradient_norm": (
            training_result.maximum_gradient_norm
        ),
        "training/batches": (
            training_result.batches
        ),
        "training/samples": (
            training_result.samples
        ),
        "training/optimizer_steps": (
            training_result.optimizer_steps
        ),
        "validation/total_loss": (
            validation_result.total_loss
        ),
        "validation/dice_loss": (
            validation_result.dice_loss
        ),
        "validation/focal_loss": (
            validation_result.focal_loss
        ),
        "validation/precision": (
            validation_result.metrics.precision
        ),
        "validation/recall": (
            validation_result.metrics.recall
        ),
        "validation/f1": (
            validation_result.metrics.f1
        ),
        "validation/iou": (
            validation_result.metrics.iou
        ),
        "validation/accuracy_secondary": (
            validation_result.metrics.accuracy
        ),
        "validation/true_positive": (
            validation_result.metrics.true_positive
        ),
        "validation/false_positive": (
            validation_result.metrics.false_positive
        ),
        "validation/false_negative": (
            validation_result.metrics.false_negative
        ),
        "validation/true_negative": (
            validation_result.metrics.true_negative
        ),
        "checkpoint/monitor_value": (
            checkpoint_update.monitor_value
        ),
        "checkpoint/best_metric": (
            checkpoint_update.best_metric
        ),
        "checkpoint/improved": int(
            checkpoint_update.improved
        ),
        "checkpoint/epochs_without_improvement": (
            checkpoint_update.epochs_without_improvement
        ),
        "checkpoint/early_stop": int(
            checkpoint_update.should_stop
        ),
        "protocol/official_test_regions_accessed": 0,
        "protocol/official_test_labels_accessed": 0,
    }


def normalize_rgb_composite(
    image: Tensor,
    band_names: tuple[str, ...],
) -> np.ndarray:
    """Convert multispectral channel-first data to displayable RGB."""
    required_bands = (
        "B04",
        "B03",
        "B02",
    )

    missing_bands = [
        band
        for band in required_bands
        if band not in band_names
    ]

    if missing_bands:
        raise TrainingConfigurationError(
            "RGB prediction logging requires B04, B03 and B02; "
            f"missing {missing_bands}."
        )

    channel_indices = [
        band_names.index(
            band
        )
        for band in required_bands
    ]

    rgb = image[
        channel_indices
    ].detach().float().cpu()

    minimum = rgb.amin(
        dim=(
            1,
            2,
        ),
        keepdim=True,
    )
    maximum = rgb.amax(
        dim=(
            1,
            2,
        ),
        keepdim=True,
    )

    rgb = (
        rgb
        - minimum
    ) / (
        maximum
        - minimum
    ).clamp_min(
        1.0e-6
    )

    return (
        rgb
        .clamp(
            0.0,
            1.0,
        )
        .permute(
            1,
            2,
            0,
        )
        .numpy()
    )


def mask_to_rgb(
    mask: Tensor,
) -> np.ndarray:
    """Convert a single-channel mask or probability map to RGB."""
    mask_array = (
        mask
        .detach()
        .float()
        .cpu()
        .squeeze(
            0
        )
        .clamp(
            0.0,
            1.0,
        )
        .numpy()
    )

    return np.repeat(
        mask_array[
            :,
            :,
            None,
        ],
        repeats=3,
        axis=2,
    )


def extract_batch_metadata(
    batch: Mapping[str, Any],
    key: str,
    sample_index: int,
    default: str,
) -> str:
    """Extract one collated metadata value from a DataLoader batch."""
    value = batch.get(
        key
    )

    if value is None:
        return default

    if isinstance(
        value,
        Tensor,
    ):
        if value.ndim == 0:
            return str(
                value.item()
            )

        return str(
            value[
                sample_index
            ].item()
        )

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return str(
            value[
                sample_index
            ]
        )

    return str(
        value
    )


def collect_validation_prediction_samples(
    runtime: TrainingRuntime,
    maximum_samples: int,
) -> list[Any]:
    """Create W&B panels from validation-only predictions.

    Each panel contains:

    1. Before RGB composite
    2. After RGB composite
    3. Ground-truth change mask
    4. Predicted change mask
    5. Change probability map
    """
    if maximum_samples <= 0:
        raise TrainingConfigurationError(
            "maximum_prediction_samples must be positive."
        )

    dataset_config = require_mapping(
        runtime.config.get(
            "dataset"
        ),
        "dataset",
    )
    metrics_config = require_mapping(
        runtime.config.get(
            "metrics"
        ),
        "metrics",
    )

    band_names = tuple(
        str(
            value
        )
        for value in dataset_config[
            "bands"
        ]
    )
    threshold = float(
        metrics_config[
            "threshold"
        ]
    )

    prediction_images: list[Any] = []
    was_training = runtime.model.training

    runtime.model.eval()

    try:
        with torch.inference_mode():
            for batch in runtime.validation_loader:
                before, after, targets = move_batch_to_device(
                    batch=batch,
                    device=runtime.device,
                )

                with torch.amp.autocast(
                    device_type=runtime.device.type,
                    enabled=(
                        runtime.mixed_precision_enabled
                    ),
                ):
                    logits = runtime.model(
                        before,
                        after,
                    )

                probabilities = torch.sigmoid(
                    logits
                )
                predictions = (
                    probabilities
                    >= threshold
                ).float()

                for sample_index in range(
                    int(
                        before.shape[
                            0
                        ]
                    )
                ):
                    before_rgb = normalize_rgb_composite(
                        image=before[
                            sample_index
                        ],
                        band_names=band_names,
                    )
                    after_rgb = normalize_rgb_composite(
                        image=after[
                            sample_index
                        ],
                        band_names=band_names,
                    )
                    target_rgb = mask_to_rgb(
                        targets[
                            sample_index
                        ]
                    )
                    prediction_rgb = mask_to_rgb(
                        predictions[
                            sample_index
                        ]
                    )
                    probability_rgb = mask_to_rgb(
                        probabilities[
                            sample_index
                        ]
                    )

                    panel = np.concatenate(
                        (
                            before_rgb,
                            after_rgb,
                            target_rgb,
                            prediction_rgb,
                            probability_rgb,
                        ),
                        axis=1,
                    )

                    region = extract_batch_metadata(
                        batch=batch,
                        key="region",
                        sample_index=sample_index,
                        default="unknown-region",
                    )
                    patch_id = extract_batch_metadata(
                        batch=batch,
                        key="patch_id",
                        sample_index=sample_index,
                        default="unknown-patch",
                    )

                    prediction_images.append(
                        wandb.Image(
                            panel,
                            caption=(
                                f"{region} | {patch_id} | "
                                "before, after, target, "
                                "prediction, probability"
                            ),
                        )
                    )

                    if (
                        len(
                            prediction_images
                        )
                        >= maximum_samples
                    ):
                        return prediction_images

    finally:
        runtime.model.train(
            was_training
        )

    return prediction_images


def should_log_prediction_samples(
    config: Mapping[str, Any],
    epoch: int,
) -> bool:
    """Return whether validation images should be logged this epoch."""
    tracking_config = require_mapping(
        config.get(
            "tracking"
        ),
        "tracking",
    )

    interval = int(
        tracking_config.get(
            "log_validation_predictions_every_epochs",
            0,
        )
    )

    if interval <= 0:
        return False

    return (
        epoch == 1
        or epoch
        % interval
        == 0
    )


def log_epoch_to_wandb(
    run: Any,
    runtime: TrainingRuntime,
    epoch: int,
    learning_rate_before_scheduler: float,
    learning_rate_after_scheduler: float,
    training_result: EpochResult,
    validation_result: EpochResult,
    checkpoint_update: CheckpointUpdate,
) -> None:
    """Log one completed epoch and optional prediction panels."""
    payload = build_epoch_wandb_payload(
        epoch=epoch,
        learning_rate_before_scheduler=(
            learning_rate_before_scheduler
        ),
        learning_rate_after_scheduler=(
            learning_rate_after_scheduler
        ),
        training_result=training_result,
        validation_result=validation_result,
        checkpoint_update=checkpoint_update,
    )

    if should_log_prediction_samples(
        runtime.config,
        epoch,
    ):
        tracking_config = require_mapping(
            runtime.config.get(
                "tracking"
            ),
            "tracking",
        )

        maximum_samples = int(
            tracking_config.get(
                "maximum_prediction_samples",
                4,
            )
        )

        prediction_samples = (
            collect_validation_prediction_samples(
                runtime=runtime,
                maximum_samples=maximum_samples,
            )
        )

        if prediction_samples:
            payload[
                "validation/prediction_samples"
            ] = prediction_samples

    run.log(
        payload,
        step=epoch,
    )

    run.summary[
        "best_validation_metric"
    ] = checkpoint_update.best_metric
    run.summary[
        "checkpoint_monitor"
    ] = checkpoint_update.monitor_name
    run.summary[
        "last_completed_epoch"
    ] = epoch
    run.summary[
        "official_test_regions_accessed"
    ] = False
    run.summary[
        "official_test_labels_accessed"
    ] = False


def run_training_loop(
    runtime: TrainingRuntime,
    epochs_override: int | None = None,
    resume_path: Path | None = None,
    maximum_training_batches: int | None = None,
    maximum_validation_batches: int | None = None,
    tracking_mode_override: str | None = None,
) -> TrainingLoopSummary:
    """Run fresh or resumed training with checkpointing and W&B."""
    training_config = require_mapping(
        runtime.config.get(
            "training"
        ),
        "training",
    )
    reproducibility_config = require_mapping(
        runtime.config.get(
            "reproducibility"
        ),
        "reproducibility",
    )

    configured_epochs = int(
        training_config[
            "epochs"
        ]
    )

    total_epochs = (
        int(
            epochs_override
        )
        if epochs_override is not None
        else configured_epochs
    )

    if total_epochs <= 0:
        raise TrainingConfigurationError(
            "The final training epoch must be positive."
        )

    if resume_path is not None:
        resume_state = load_training_checkpoint(
            path=resume_path,
            runtime=runtime,
            strict_config=True,
            restore_rng=True,
        )

        start_epoch = resume_state.next_epoch

        early_stopping = build_early_stopping_from_config(
            config=runtime.config,
            best_metric=resume_state.best_metric,
            epochs_without_improvement=(
                resume_state
                .epochs_without_improvement
            ),
        )
    else:
        start_epoch = 1
        early_stopping = build_early_stopping_from_config(
            runtime.config
        )

    if start_epoch > total_epochs:
        raise TrainingConfigurationError(
            "The checkpoint already reached or exceeded the requested "
            f"final epoch. Resume epoch is {start_epoch}; "
            f"requested final epoch is {total_epochs}."
        )

    run_log_directory = resolve_run_log_directory(
        runtime.config
    )
    history_path = (
        run_log_directory
        / "training_history.jsonl"
    )
    resolved_config_path = (
        run_log_directory
        / "resolved_train_config.yaml"
    )

    run_log_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if resume_path is None:
        history_path.unlink(
            missing_ok=True
        )

    if bool(
        reproducibility_config.get(
            "save_resolved_config",
            True,
        )
    ):
        write_resolved_config(
            config=runtime.config,
            output_path=resolved_config_path,
        )

    wandb_run = initialize_wandb_run(
        config=runtime.config,
        run_log_directory=run_log_directory,
        mode_override=tracking_mode_override,
        resume_path=resume_path,
    )

    wandb_run_id = (
        str(
            getattr(
                wandb_run,
                "id",
                "",
            )
        )
        or None
        if wandb_run is not None
        else None
    )
    wandb_run_name = (
        str(
            getattr(
                wandb_run,
                "name",
                "",
            )
        )
        or None
        if wandb_run is not None
        else None
    )

    epochs_completed = 0
    final_epoch = start_epoch - 1
    stopped_early = False
    completed_successfully = False

    best_checkpoint_path, last_checkpoint_path = (
        resolve_checkpoint_paths(
            runtime.config
        )
    )

    try:
        for epoch in range(
            start_epoch,
            total_epochs + 1,
        ):
            learning_rate_before_scheduler = float(
                runtime.optimizer.param_groups[
                    0
                ][
                    "lr"
                ]
            )

            training_result = train_one_epoch(
                runtime=runtime,
                maximum_batches=(
                    maximum_training_batches
                ),
            )

            validation_result = validate_one_epoch(
                runtime=runtime,
                maximum_batches=(
                    maximum_validation_batches
                ),
            )

            runtime.scheduler.step()

            learning_rate_after_scheduler = float(
                runtime.optimizer.param_groups[
                    0
                ][
                    "lr"
                ]
            )

            checkpoint_update = update_epoch_checkpoints(
                runtime=runtime,
                epoch=epoch,
                training_result=training_result,
                validation_result=validation_result,
                early_stopping=early_stopping,
            )

            history_record = build_epoch_history_record(
                epoch=epoch,
                learning_rate_before_scheduler=(
                    learning_rate_before_scheduler
                ),
                learning_rate_after_scheduler=(
                    learning_rate_after_scheduler
                ),
                training_result=training_result,
                validation_result=validation_result,
                checkpoint_update=checkpoint_update,
            )

            append_json_record(
                path=history_path,
                record=history_record,
            )

            if wandb_run is not None:
                log_epoch_to_wandb(
                    run=wandb_run,
                    runtime=runtime,
                    epoch=epoch,
                    learning_rate_before_scheduler=(
                        learning_rate_before_scheduler
                    ),
                    learning_rate_after_scheduler=(
                        learning_rate_after_scheduler
                    ),
                    training_result=training_result,
                    validation_result=validation_result,
                    checkpoint_update=checkpoint_update,
                )

            epochs_completed += 1
            final_epoch = epoch

            print(
                f"Epoch {epoch}/{total_epochs}"
            )
            print(
                "  Learning rate:",
                learning_rate_before_scheduler,
            )
            print(
                "  Training loss:",
                training_result.total_loss,
            )
            print(
                "  Validation loss:",
                validation_result.total_loss,
            )
            print(
                "  Validation precision:",
                validation_result.metrics.precision,
            )
            print(
                "  Validation recall:",
                validation_result.metrics.recall,
            )
            print(
                "  Validation F1:",
                validation_result.metrics.f1,
            )
            print(
                "  Validation IoU:",
                validation_result.metrics.iou,
            )
            print(
                "  Validation accuracy:",
                validation_result.metrics.accuracy,
                "(secondary only)",
            )
            print(
                "  Best validation metric:",
                checkpoint_update.best_metric,
            )
            print(
                "  Improved:",
                checkpoint_update.improved,
            )
            print(
                "  Epochs without improvement:",
                (
                    checkpoint_update
                    .epochs_without_improvement
                ),
            )

            if checkpoint_update.should_stop:
                stopped_early = True

                print(
                    "  Early stopping triggered:",
                    True,
                )
                break

        if epochs_completed <= 0:
            raise TrainingConfigurationError(
                "The training loop completed no epochs."
            )

        if early_stopping.best_metric is None:
            raise TrainingConfigurationError(
                "Training finished without a best validation metric."
            )

        resolved_best_checkpoint = (
            best_checkpoint_path
            if best_checkpoint_path.is_file()
            else None
        )

        if wandb_run is not None:
            wandb_run.summary[
                "epochs_completed"
            ] = epochs_completed
            wandb_run.summary[
                "stopped_early"
            ] = stopped_early
            wandb_run.summary[
                "best_checkpoint"
            ] = (
                str(
                    resolved_best_checkpoint
                )
                if resolved_best_checkpoint is not None
                else None
            )
            wandb_run.summary[
                "last_checkpoint"
            ] = str(
                last_checkpoint_path
            )

        summary = TrainingLoopSummary(
            start_epoch=start_epoch,
            final_epoch=final_epoch,
            epochs_completed=epochs_completed,
            stopped_early=stopped_early,
            best_metric=float(
                early_stopping.best_metric
            ),
            last_checkpoint=last_checkpoint_path,
            best_checkpoint=resolved_best_checkpoint,
            history_path=history_path,
            resolved_config_path=resolved_config_path,
            wandb_enabled=(
                wandb_run is not None
            ),
            wandb_run_id=wandb_run_id,
            wandb_run_name=wandb_run_name,
        )

        completed_successfully = True

        return summary

    finally:
        if wandb_run is not None:
            wandb_run.finish(
                exit_code=(
                    0
                    if completed_successfully
                    else 1
                )
            )

def build_argument_parser() -> argparse.ArgumentParser:
    """Build the GeoWatch training command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run or train the configuration-driven GeoWatch "
            "Siamese U-Net."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/train_config.yaml"
        ),
    )

    execution_group = parser.add_mutually_exclusive_group(
        required=True
    )
    execution_group.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run one training batch and one validation batch."
        ),
    )
    execution_group.add_argument(
        "--train",
        action="store_true",
        help=(
            "Run complete or batch-limited multi-epoch training."
        ),
    )

    parser.add_argument(
        "--device",
        choices=(
            "auto",
            "cpu",
            "cuda",
        ),
        default=None,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--validation-batch-size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--disable-pretrained",
        action="store_true",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Override the final target epoch. On resume this is not "
            "the number of additional epochs."
        ),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "Resume complete state from a GeoWatch checkpoint."
        ),
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help=(
            "Limit training batches per epoch for smoke testing."
        ),
    )
    parser.add_argument(
        "--max-validation-batches",
        type=int,
        default=None,
        help=(
            "Limit validation batches per epoch for smoke testing."
        ),
    )
    parser.add_argument(
        "--wandb-mode",
        choices=(
            "online",
            "offline",
            "disabled",
        ),
        default=None,
        help=(
            "Override tracking.mode for this run."
        ),
    )

    parser.add_argument(
        "--log-level",
        choices=(
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
        ),
        default="INFO",
    )

    return parser


def print_dry_run_summary(
    runtime: TrainingRuntime,
    result: DryRunResult,
    config_path: Path,
) -> None:
    """Print the one-batch runtime audit."""
    total_parameters, trainable_parameters = (
        count_parameters(
            runtime.model
        )
    )

    metrics = result.validation_metrics

    print(
        "GeoWatch training-runtime dry run passed"
    )
    print(
        "  Configuration:",
        config_path,
    )
    print(
        "  Device:",
        runtime.device,
    )
    print(
        "  CUDA available:",
        torch.cuda.is_available(),
    )
    print(
        "  Mixed precision requested:",
        bool(
            runtime.config[
                "training"
            ][
                "mixed_precision"
            ]
        ),
    )
    print(
        "  Mixed precision enabled:",
        runtime.mixed_precision_enabled,
    )
    print(
        "  Training regions:",
        len(
            runtime.train_dataset.region_names
        ),
    )
    print(
        "  Validation regions:",
        len(
            runtime.validation_dataset.region_names
        ),
    )
    print(
        "  Training patches:",
        len(
            runtime.train_dataset
        ),
    )
    print(
        "  Validation patches:",
        len(
            runtime.validation_dataset
        ),
    )
    print(
        "  Training batch shape:",
        result.training_batch_shape,
    )
    print(
        "  Validation batch shape:",
        result.validation_batch_shape,
    )
    print(
        "  Total parameters:",
        total_parameters,
    )
    print(
        "  Trainable parameters:",
        trainable_parameters,
    )
    print(
        "  Training total loss:",
        result.training_total_loss,
    )
    print(
        "  Gradient norm before clipping:",
        result.gradient_norm,
    )
    print(
        "  Validation total loss:",
        result.validation_total_loss,
    )
    print(
        "  Validation precision:",
        metrics.precision,
    )
    print(
        "  Validation recall:",
        metrics.recall,
    )
    print(
        "  Validation F1:",
        metrics.f1_score,
    )
    print(
        "  Validation IoU:",
        metrics.iou,
    )
    print(
        "  Metric scope:",
        "positive change class",
    )
    print(
        "  Official test regions accessed:",
        False,
    )
    print(
        "  Official test labels accessed:",
        False,
    )


def main() -> int:
    """Execute a dry run or multi-epoch training."""
    parser = build_argument_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(
            logging,
            args.log_level,
        ),
        format="%(levelname)s: %(message)s",
    )

    try:
        if args.resume is not None and not args.train:
            raise TrainingConfigurationError(
                "--resume is valid only with --train."
            )

        if (
            args.wandb_mode is not None
            and not args.train
        ):
            raise TrainingConfigurationError(
                "--wandb-mode is valid only with --train."
            )

        config = load_training_config(
            args.config
        )

        runtime = build_training_runtime(
            config=config,
            device_override=args.device,
            batch_size_override=args.batch_size,
            validation_batch_size_override=(
                args.validation_batch_size
            ),
            num_workers_override=args.num_workers,
            disable_pretrained=(
                args.disable_pretrained
            ),
        )

        if args.dry_run:
            result = run_training_dry_run(
                runtime
            )

            print_dry_run_summary(
                runtime=runtime,
                result=result,
                config_path=args.config,
            )

            return 0

        summary = run_training_loop(
            runtime=runtime,
            epochs_override=args.epochs,
            resume_path=args.resume,
            maximum_training_batches=(
                args.max_train_batches
            ),
            maximum_validation_batches=(
                args.max_validation_batches
            ),
            tracking_mode_override=(
                args.wandb_mode
            ),
        )

        print(
            "GeoWatch multi-epoch training completed"
        )
        print(
            "  Start epoch:",
            summary.start_epoch,
        )
        print(
            "  Final epoch:",
            summary.final_epoch,
        )
        print(
            "  Epochs completed:",
            summary.epochs_completed,
        )
        print(
            "  Stopped early:",
            summary.stopped_early,
        )
        print(
            "  Best validation metric:",
            summary.best_metric,
        )
        print(
            "  Last checkpoint:",
            summary.last_checkpoint,
        )
        print(
            "  Best checkpoint:",
            summary.best_checkpoint,
        )
        print(
            "  Training history:",
            summary.history_path,
        )
        print(
            "  Resolved configuration:",
            summary.resolved_config_path,
        )
        print(
            "  Metric scope:",
            "positive change class",
        )
        print(
            "  Official test regions accessed:",
            False,
        )
        print(
            "  Official test labels accessed:",
            False,
        )
        print(
            "  W&B initialized:",
            summary.wandb_enabled,
        )
        print(
            "  W&B run ID:",
            summary.wandb_run_id,
        )
        print(
            "  W&B run name:",
            summary.wandb_run_name,
        )

        return 0

    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
        TrainingConfigurationError,
        OSError,
    ) as error:
        LOGGER.error(
            "%s",
            error,
        )
        return 1

    except Exception:
        LOGGER.exception(
            "Unexpected training failure."
        )
        return 1


if __name__ == "__main__":
    sys.exit(
        main()
    )
