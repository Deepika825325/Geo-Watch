"""Tests for GeoWatch full-epoch training and validation functions."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from src.training.losses import DiceFocalLoss
from src.training.train import (
    BinaryMetricAccumulator,
    TrainingRuntime,
    train_one_epoch,
    validate_one_epoch,
)


class SyntheticChangeDataset(
    Dataset[dict[str, Any]]
):
    """Small aligned bi-temporal dataset for CPU tests."""

    def __init__(
        self,
        sample_count: int = 5,
    ) -> None:
        self.sample_count = sample_count
        self.region_names = (
            "synthetic_training_region",
        )

    def __len__(
        self,
    ) -> int:
        return self.sample_count

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        generator = torch.Generator().manual_seed(
            100 + index
        )

        before = torch.rand(
            4,
            16,
            16,
            generator=generator,
        )
        after = before.clone()

        mask = torch.zeros(
            1,
            16,
            16,
            dtype=torch.float32,
        )

        row = (
            index
            % 4
        ) * 2

        mask[
            :,
            row:row + 4,
            4:8,
        ] = 1.0

        after[
            3,
            row:row + 4,
            4:8,
        ] += 0.25

        return {
            "before": before,
            "after": after,
            "mask": mask,
            "region": "synthetic_training_region",
            "patch_id": f"patch_{index}",
            "row": row,
            "column": 0,
        }


class TinySiameseModel(nn.Module):
    """Minimal two-input model matching the GeoWatch model contract."""

    def __init__(
        self,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(
                4,
                8,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                8,
                1,
                kernel_size=1,
            ),
        )

    def forward(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
    ) -> torch.Tensor:
        difference = torch.abs(
            after - before
        )

        return self.network(
            difference
        )


def build_test_runtime(
    sample_count: int = 5,
    batch_size: int = 2,
    accumulation_steps: int = 2,
) -> TrainingRuntime:
    """Build a CPU-only runtime for epoch-engine tests."""
    train_dataset = SyntheticChangeDataset(
        sample_count=sample_count
    )
    validation_dataset = SyntheticChangeDataset(
        sample_count=sample_count
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = TinySiameseModel()

    criterion = DiceFocalLoss(
        dice_weight=0.5,
        focal_weight=0.5,
        focal_alpha=0.75,
        focal_gamma=2.0,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=1.0e-3,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=3,
    )

    config = {
        "training": {
            "gradient_accumulation_steps": accumulation_steps,
            "maximum_gradient_norm": 1.0,
        },
        "metrics": {
            "threshold": 0.5,
        },
    }

    return TrainingRuntime(
        config=config,
        device=torch.device(
            "cpu"
        ),
        mixed_precision_enabled=False,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        train_loader=train_loader,
        validation_loader=validation_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=torch.amp.GradScaler(
            "cpu",
            enabled=False,
        ),
    )


def test_metric_accumulator_uses_global_confusion_counts() -> None:
    """Positive-class metrics must match known global counts."""
    accumulator = BinaryMetricAccumulator(
        threshold=0.5
    )

    probabilities = torch.tensor(
        [
            0.9,
            0.8,
            0.7,
            0.1,
            0.2,
            0.6,
        ],
        dtype=torch.float32,
    ).reshape(
        1,
        1,
        2,
        3,
    )

    logits = torch.logit(
        probabilities
    )

    targets = torch.tensor(
        [
            1.0,
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ],
        dtype=torch.float32,
    ).reshape(
        1,
        1,
        2,
        3,
    )

    accumulator.update(
        logits,
        targets,
    )

    metrics = accumulator.compute()

    assert metrics.true_positive == 2
    assert metrics.false_positive == 2
    assert metrics.false_negative == 1
    assert metrics.true_negative == 1

    assert metrics.precision == pytest.approx(
        0.5
    )
    assert metrics.recall == pytest.approx(
        2.0 / 3.0
    )
    assert metrics.f1 == pytest.approx(
        4.0 / 7.0
    )
    assert metrics.iou == pytest.approx(
        0.4
    )
    assert metrics.accuracy == pytest.approx(
        0.5
    )


def test_training_epoch_updates_model_and_handles_partial_group() -> None:
    """Five samples with accumulation two must complete two optimizer steps."""
    torch.manual_seed(
        42
    )

    runtime = build_test_runtime(
        sample_count=5,
        batch_size=2,
        accumulation_steps=2,
    )

    parameters_before = {
        name: parameter.detach().clone()
        for name, parameter
        in runtime.model.named_parameters()
    }

    result = train_one_epoch(
        runtime
    )

    assert result.batches == 3
    assert result.samples == 5
    assert result.optimizer_steps == 2

    assert result.total_loss > 0
    assert result.dice_loss > 0
    assert result.focal_loss >= 0

    assert torch.isfinite(
        torch.tensor(
            result.mean_gradient_norm
        )
    )
    assert torch.isfinite(
        torch.tensor(
            result.maximum_gradient_norm
        )
    )

    changed_parameters = [
        not torch.equal(
            parameters_before[name],
            parameter.detach(),
        )
        for name, parameter
        in runtime.model.named_parameters()
    ]

    assert any(
        changed_parameters
    )


def test_validation_epoch_does_not_update_parameters() -> None:
    """Validation must not mutate model parameters."""
    torch.manual_seed(
        42
    )

    runtime = build_test_runtime()

    parameters_before = deepcopy(
        runtime.model.state_dict()
    )

    result = validate_one_epoch(
        runtime
    )

    parameters_after = runtime.model.state_dict()

    for name in parameters_before:
        assert torch.equal(
            parameters_before[name],
            parameters_after[name],
        )

    assert result.batches == 3
    assert result.samples == 5
    assert result.optimizer_steps == 0
    assert result.mean_gradient_norm == 0.0
    assert result.maximum_gradient_norm == 0.0

    assert 0.0 <= result.metrics.precision <= 1.0
    assert 0.0 <= result.metrics.recall <= 1.0
    assert 0.0 <= result.metrics.f1 <= 1.0
    assert 0.0 <= result.metrics.iou <= 1.0
    assert 0.0 <= result.metrics.accuracy <= 1.0


def test_epoch_batch_limits_are_respected() -> None:
    """Smoke-test limits must stop both loops at the requested count."""
    runtime = build_test_runtime(
        sample_count=8,
        batch_size=2,
        accumulation_steps=2,
    )

    training_result = train_one_epoch(
        runtime,
        maximum_batches=2,
    )

    validation_result = validate_one_epoch(
        runtime,
        maximum_batches=1,
    )

    assert training_result.batches == 2
    assert training_result.samples == 4
    assert training_result.optimizer_steps == 1

    assert validation_result.batches == 1
    assert validation_result.samples == 2
