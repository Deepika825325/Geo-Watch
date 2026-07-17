"""Tests for GeoWatch W&B training integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

import src.training.train as training_module
from src.training.losses import DiceFocalLoss
from src.training.train import (
    ChangeClassMetrics,
    CheckpointUpdate,
    EpochResult,
    TrainingRuntime,
    build_epoch_wandb_payload,
    collect_validation_prediction_samples,
    run_training_loop,
)


class TinyTrackingDataset(
    Dataset[dict[str, Any]]
):
    """Small RGB-compatible four-band temporal dataset."""

    region_names = (
        "synthetic_region",
    )

    def __len__(
        self,
    ) -> int:
        return 2

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        before = torch.zeros(
            4,
            8,
            8,
        )
        after = before.clone()
        mask = torch.zeros(
            1,
            8,
            8,
        )

        before[
            0
        ] = 0.1
        before[
            1
        ] = 0.2
        before[
            2
        ] = 0.3
        before[
            3
        ] = 0.4

        after.copy_(
            before
        )

        mask[
            :,
            2:5,
            2:5,
        ] = 1.0
        after[
            3,
            2:5,
            2:5,
        ] = 0.8

        return {
            "before": before,
            "after": after,
            "mask": mask,
            "region": "synthetic_region",
            "patch_id": f"patch_{index}",
            "row": 0,
            "column": 0,
        }


class TinyTrackingModel(nn.Module):
    """Small two-input change model."""

    def __init__(
        self,
    ) -> None:
        super().__init__()

        self.layer = nn.Conv2d(
            4,
            1,
            kernel_size=1,
        )

    def forward(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
    ) -> torch.Tensor:
        return self.layer(
            torch.abs(
                after - before
            )
        )


class FakeWandbRun:
    """Capture W&B calls without network or filesystem activity."""

    def __init__(
        self,
    ) -> None:
        self.id = "fake-run-id"
        self.name = "fake-run-name"
        self.logged: list[
            tuple[int | None, dict[str, Any]]
        ] = []
        self.defined_metrics: list[
            tuple[tuple[Any, ...], dict[str, Any]]
        ] = []
        self.summary: dict[str, Any] = {}
        self.finished = False
        self.exit_code: int | None = None

    def define_metric(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.defined_metrics.append(
            (
                args,
                kwargs,
            )
        )

    def log(
        self,
        payload: dict[str, Any],
        step: int | None = None,
    ) -> None:
        self.logged.append(
            (
                step,
                payload,
            )
        )

    def finish(
        self,
        exit_code: int = 0,
    ) -> None:
        self.finished = True
        self.exit_code = exit_code


def build_runtime(
    root: Path,
) -> TrainingRuntime:
    """Build a CPU runtime with offline tracking enabled."""
    dataset = TinyTrackingDataset()

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    model = TinyTrackingModel()

    optimizer = AdamW(
        model.parameters(),
        lr=1.0e-3,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=2,
        eta_min=1.0e-5,
    )

    config = {
        "project": {
            "name": "geowatch",
            "experiment_name": "wandb_test",
            "seed": 42,
        },
        "paths": {
            "checkpoint_directory": str(
                root
                / "checkpoints"
            ),
            "log_directory": str(
                root
                / "logs"
            ),
        },
        "dataset": {
            "bands": [
                "B02",
                "B03",
                "B04",
                "B08",
            ],
        },
        "training": {
            "epochs": 1,
            "gradient_accumulation_steps": 1,
            "maximum_gradient_norm": 1.0,
            "early_stopping": {
                "enabled": True,
                "monitor": "validation_f1",
                "mode": "max",
                "patience": 2,
                "minimum_delta": 0.0001,
            },
        },
        "checkpointing": {
            "monitor": "validation_f1",
            "mode": "max",
            "save_best_only": True,
            "save_last": True,
            "filename": "best_model.pt",
        },
        "metrics": {
            "threshold": 0.5,
        },
        "tracking": {
            "enabled": True,
            "project": "geowatch-change-detection",
            "entity": None,
            "group": "week4",
            "job_type": "training",
            "mode": "offline",
            "log_validation_predictions_every_epochs": 1,
            "maximum_prediction_samples": 1,
        },
        "reproducibility": {
            "save_resolved_config": True,
        },
    }

    return TrainingRuntime(
        config=config,
        device=torch.device(
            "cpu"
        ),
        mixed_precision_enabled=False,
        train_dataset=dataset,
        validation_dataset=dataset,
        train_loader=loader,
        validation_loader=loader,
        model=model,
        criterion=DiceFocalLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=torch.amp.GradScaler(
            "cpu",
            enabled=False,
        ),
    )


def build_epoch_result() -> EpochResult:
    """Create one deterministic metric result."""
    return EpochResult(
        total_loss=0.5,
        dice_loss=0.9,
        focal_loss=0.1,
        metrics=ChangeClassMetrics(
            true_positive=10,
            false_positive=5,
            false_negative=5,
            true_negative=100,
            precision=2.0 / 3.0,
            recall=2.0 / 3.0,
            f1=2.0 / 3.0,
            iou=0.5,
            accuracy=110.0 / 120.0,
        ),
        batches=2,
        samples=4,
        optimizer_steps=1,
        mean_gradient_norm=0.4,
        maximum_gradient_norm=0.8,
    )


def test_wandb_payload_contains_primary_change_metrics() -> None:
    """Epoch logging must include F1, IoU and loss components."""
    result = build_epoch_result()

    checkpoint_update = CheckpointUpdate(
        last_checkpoint=Path(
            "last_model.pt"
        ),
        best_checkpoint=Path(
            "best_model.pt"
        ),
        monitor_name="validation_f1",
        monitor_value=result.metrics.f1,
        improved=True,
        should_stop=False,
        best_metric=result.metrics.f1,
        epochs_without_improvement=0,
    )

    payload = build_epoch_wandb_payload(
        epoch=1,
        learning_rate_before_scheduler=1.0e-4,
        learning_rate_after_scheduler=9.0e-5,
        training_result=result,
        validation_result=result,
        checkpoint_update=checkpoint_update,
    )

    assert payload[
        "epoch"
    ] == 1
    assert payload[
        "validation/f1"
    ] == pytest.approx(
        2.0 / 3.0
    )
    assert payload[
        "validation/iou"
    ] == pytest.approx(
        0.5
    )
    assert payload[
        "training/dice_loss"
    ] == pytest.approx(
        0.9
    )
    assert payload[
        "training/focal_loss"
    ] == pytest.approx(
        0.1
    )
    assert payload[
        "protocol/official_test_regions_accessed"
    ] == 0


def test_validation_prediction_panel_is_created(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prediction logging must produce one five-panel validation image."""
    runtime = build_runtime(
        tmp_path
    )

    captured_images: list[
        tuple[Any, str | None]
    ] = []

    def fake_image(
        data: Any,
        caption: str | None = None,
    ) -> dict[str, Any]:
        captured_images.append(
            (
                data,
                caption,
            )
        )

        return {
            "data": data,
            "caption": caption,
        }

    monkeypatch.setattr(
        training_module.wandb,
        "Image",
        fake_image,
    )

    images = collect_validation_prediction_samples(
        runtime=runtime,
        maximum_samples=1,
    )

    assert len(
        images
    ) == 1
    assert len(
        captured_images
    ) == 1

    panel = captured_images[
        0
    ][
        0
    ]

    assert panel.shape == (
        8,
        40,
        3,
    )
    assert "before, after, target" in captured_images[
        0
    ][
        1
    ]


def test_training_loop_logs_and_finishes_wandb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real epoch must log metrics, predictions and run summary."""
    torch.manual_seed(
        42
    )

    runtime = build_runtime(
        tmp_path
    )
    fake_run = FakeWandbRun()

    monkeypatch.setattr(
        training_module.wandb,
        "init",
        lambda **kwargs: fake_run,
    )
    monkeypatch.setattr(
        training_module.wandb,
        "Image",
        lambda data, caption=None: {
            "data": data,
            "caption": caption,
        },
    )

    summary = run_training_loop(
        runtime=runtime,
        epochs_override=1,
        maximum_training_batches=1,
        maximum_validation_batches=1,
        tracking_mode_override="offline",
    )

    assert summary.wandb_enabled is True
    assert summary.wandb_run_id == "fake-run-id"
    assert summary.wandb_run_name == "fake-run-name"

    assert fake_run.finished is True
    assert fake_run.exit_code == 0
    assert len(
        fake_run.logged
    ) == 1

    step, payload = fake_run.logged[
        0
    ]

    assert step == 1
    assert "validation/f1" in payload
    assert "validation/iou" in payload
    assert "validation/prediction_samples" in payload

    assert fake_run.summary[
        "official_test_regions_accessed"
    ] is False
    assert fake_run.summary[
        "epochs_completed"
    ] == 1
