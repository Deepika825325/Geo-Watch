"""Tests for GeoWatch multi-epoch orchestration."""

from __future__ import annotations

import json
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
    EpochResult,
    TrainingRuntime,
    run_training_loop,
)


class TinyChangeDataset(
    Dataset[dict[str, Any]]
):
    """Small deterministic temporal dataset."""

    region_names = (
        "synthetic_region",
    )

    def __len__(
        self,
    ) -> int:
        return 4

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

        mask[
            :,
            2:4,
            2:4,
        ] = 1.0
        after[
            3,
            2:4,
            2:4,
        ] = 0.5

        return {
            "before": before,
            "after": after,
            "mask": mask,
            "region": "synthetic_region",
            "patch_id": f"patch_{index}",
            "row": 0,
            "column": 0,
        }


class TinyModel(nn.Module):
    """Small model matching the two-image contract."""

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


def build_runtime(
    root: Path,
    patience: int = 5,
) -> TrainingRuntime:
    """Build a deterministic CPU runtime."""
    dataset = TinyChangeDataset()

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    model = TinyModel()

    optimizer = AdamW(
        model.parameters(),
        lr=1.0e-3,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=5,
        eta_min=1.0e-5,
    )

    config = {
        "project": {
            "name": "geowatch",
            "experiment_name": "unit_test",
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
        "training": {
            "epochs": 5,
            "gradient_accumulation_steps": 1,
            "maximum_gradient_norm": 1.0,
            "early_stopping": {
                "enabled": True,
                "monitor": "validation_f1",
                "mode": "max",
                "patience": patience,
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


def fixed_epoch_result(
    f1: float,
) -> EpochResult:
    """Return a synthetic result for early-stopping tests."""
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
            f1=f1,
            iou=0.5,
            accuracy=110.0 / 120.0,
        ),
        batches=1,
        samples=2,
        optimizer_steps=1,
        mean_gradient_norm=0.5,
        maximum_gradient_norm=0.5,
    )


def test_multi_epoch_run_saves_history_and_checkpoints(
    tmp_path: Path,
) -> None:
    """Two epochs must update the scheduler and persist all state."""
    torch.manual_seed(
        42
    )

    runtime = build_runtime(
        tmp_path
    )

    summary = run_training_loop(
        runtime=runtime,
        epochs_override=2,
        maximum_training_batches=1,
        maximum_validation_batches=1,
    )

    assert summary.start_epoch == 1
    assert summary.final_epoch == 2
    assert summary.epochs_completed == 2
    assert summary.stopped_early is False

    assert summary.last_checkpoint.is_file()
    assert summary.best_checkpoint is not None
    assert summary.best_checkpoint.is_file()
    assert summary.history_path.is_file()
    assert summary.resolved_config_path.is_file()

    records = [
        json.loads(
            line
        )
        for line in summary.history_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    assert len(
        records
    ) == 2
    assert records[
        0
    ][
        "epoch"
    ] == 1
    assert records[
        1
    ][
        "epoch"
    ] == 2

    assert runtime.scheduler.last_epoch == 2


def test_training_resume_continues_at_next_epoch(
    tmp_path: Path,
) -> None:
    """A second runtime must continue from the saved epoch."""
    first_runtime = build_runtime(
        tmp_path
    )

    first_summary = run_training_loop(
        runtime=first_runtime,
        epochs_override=1,
        maximum_training_batches=1,
        maximum_validation_batches=1,
    )

    resumed_runtime = build_runtime(
        tmp_path
    )

    resumed_summary = run_training_loop(
        runtime=resumed_runtime,
        epochs_override=2,
        resume_path=first_summary.last_checkpoint,
        maximum_training_batches=1,
        maximum_validation_batches=1,
    )

    assert resumed_summary.start_epoch == 2
    assert resumed_summary.final_epoch == 2
    assert resumed_summary.epochs_completed == 1
    assert resumed_runtime.scheduler.last_epoch == 2


def test_training_loop_honours_early_stopping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The orchestration must stop after configured non-improvements."""
    runtime = build_runtime(
        tmp_path,
        patience=2,
    )

    validation_results = iter(
        (
            fixed_epoch_result(
                0.20
            ),
            fixed_epoch_result(
                0.20001
            ),
            fixed_epoch_result(
                0.19
            ),
        )
    )

    def mocked_train_one_epoch(
        runtime: TrainingRuntime,
        maximum_batches: int | None = None,
    ) -> EpochResult:
        """Emulate the optimizer-step contract of a real training epoch."""
        del maximum_batches

        runtime.optimizer.zero_grad(
            set_to_none=True
        )
        runtime.optimizer.step()

        return fixed_epoch_result(
            0.1
        )

    monkeypatch.setattr(
        training_module,
        "train_one_epoch",
        mocked_train_one_epoch,
    )
    monkeypatch.setattr(
        training_module,
        "validate_one_epoch",
        lambda runtime, maximum_batches=None: next(
            validation_results
        ),
    )

    summary = run_training_loop(
        runtime=runtime,
        epochs_override=5,
    )

    assert summary.stopped_early is True
    assert summary.start_epoch == 1
    assert summary.final_epoch == 3
    assert summary.epochs_completed == 3
    assert summary.best_metric == pytest.approx(
        0.20
    )
