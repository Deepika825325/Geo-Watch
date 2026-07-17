"""Tests for GeoWatch checkpointing, resumption and early stopping."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from src.training.losses import DiceFocalLoss
from src.training.train import (
    ChangeClassMetrics,
    EarlyStopping,
    EpochResult,
    TrainingConfigurationError,
    TrainingRuntime,
    build_early_stopping_from_config,
    capture_random_states,
    load_training_checkpoint,
    restore_random_states,
    update_epoch_checkpoints,
)


def build_config(
    checkpoint_directory: Path,
) -> dict[str, Any]:
    """Create a minimal checkpoint configuration."""
    return {
        "paths": {
            "checkpoint_directory": str(
                checkpoint_directory
            ),
        },
        "checkpointing": {
            "monitor": "validation_f1",
            "mode": "max",
            "save_best_only": True,
            "save_last": True,
            "filename": "best_model.pt",
        },
        "training": {
            "early_stopping": {
                "enabled": True,
                "monitor": "validation_f1",
                "mode": "max",
                "patience": 2,
                "minimum_delta": 0.01,
            },
        },
    }


def build_runtime(
    checkpoint_directory: Path,
) -> TrainingRuntime:
    """Build a small CPU runtime for checkpoint tests."""
    dataset = TensorDataset(
        torch.zeros(
            1,
            1,
            2,
            2,
        )
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
    )

    model = nn.Conv2d(
        1,
        1,
        kernel_size=1,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=1.0e-3,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=5,
    )

    return TrainingRuntime(
        config=build_config(
            checkpoint_directory
        ),
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


def build_epoch_result(
    f1: float,
    total_loss: float = 0.5,
) -> EpochResult:
    """Create a deterministic epoch result."""
    return EpochResult(
        total_loss=total_loss,
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
        batches=2,
        samples=4,
        optimizer_steps=1,
        mean_gradient_norm=0.5,
        maximum_gradient_norm=0.8,
    )


def test_early_stopping_respects_delta_and_patience() -> None:
    """Stopping occurs after the configured non-improving epochs."""
    early_stopping = EarlyStopping(
        monitor="validation_f1",
        mode="max",
        patience=2,
        minimum_delta=0.01,
    )

    first = early_stopping.update(
        0.20
    )
    second = early_stopping.update(
        0.205
    )
    third = early_stopping.update(
        0.19
    )

    assert first.improved is True
    assert first.should_stop is False

    assert second.improved is False
    assert second.should_stop is False
    assert second.epochs_without_improvement == 1

    assert third.improved is False
    assert third.should_stop is True
    assert third.epochs_without_improvement == 2
    assert third.best_metric == pytest.approx(
        0.20
    )


def test_best_and_last_checkpoints_are_saved(
    tmp_path: Path,
) -> None:
    """First improvement must produce both best and last files."""
    runtime = build_runtime(
        tmp_path
    )
    early_stopping = build_early_stopping_from_config(
        runtime.config
    )

    result = build_epoch_result(
        f1=0.25
    )

    update = update_epoch_checkpoints(
        runtime=runtime,
        epoch=1,
        training_result=result,
        validation_result=result,
        early_stopping=early_stopping,
    )

    assert update.improved is True
    assert update.should_stop is False
    assert update.best_checkpoint is not None
    assert update.best_checkpoint.is_file()
    assert update.last_checkpoint.is_file()

    best_payload = torch.load(
        update.best_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    last_payload = torch.load(
        update.last_checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    assert best_payload["epoch"] == 1
    assert last_payload["epoch"] == 1
    assert best_payload["monitor_name"] == "validation_f1"
    assert best_payload["monitor_value"] == pytest.approx(
        0.25
    )


def test_checkpoint_restores_complete_runtime_state(
    tmp_path: Path,
) -> None:
    """Model, optimizer, scheduler and training position must resume."""
    torch.manual_seed(
        42
    )

    runtime = build_runtime(
        tmp_path
    )

    input_tensor = torch.ones(
        1,
        1,
        2,
        2,
    )

    output = runtime.model(
        input_tensor
    )
    loss = output.mean()
    loss.backward()

    runtime.optimizer.step()
    runtime.optimizer.zero_grad(
        set_to_none=True
    )
    runtime.scheduler.step()

    expected_parameters = {
        name: parameter.detach().clone()
        for name, parameter
        in runtime.model.named_parameters()
    }
    expected_learning_rate = runtime.optimizer.param_groups[
        0
    ][
        "lr"
    ]
    expected_scheduler_epoch = runtime.scheduler.last_epoch

    early_stopping = build_early_stopping_from_config(
        runtime.config
    )
    result = build_epoch_result(
        f1=0.30
    )

    update = update_epoch_checkpoints(
        runtime=runtime,
        epoch=3,
        training_result=result,
        validation_result=result,
        early_stopping=early_stopping,
    )

    with torch.no_grad():
        for parameter in runtime.model.parameters():
            parameter.add_(
                100.0
            )

    runtime.optimizer.param_groups[
        0
    ][
        "lr"
    ] = 0.5

    resume = load_training_checkpoint(
        path=update.last_checkpoint,
        runtime=runtime,
    )

    assert resume.next_epoch == 4
    assert resume.best_metric == pytest.approx(
        0.30
    )
    assert resume.epochs_without_improvement == 0
    assert resume.monitor_name == "validation_f1"

    for name, parameter in runtime.model.named_parameters():
        assert torch.equal(
            parameter.detach(),
            expected_parameters[
                name
            ],
        )

    assert runtime.optimizer.param_groups[
        0
    ][
        "lr"
    ] == pytest.approx(
        expected_learning_rate
    )
    assert runtime.scheduler.last_epoch == (
        expected_scheduler_epoch
    )


def test_configuration_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    """A checkpoint from a different experiment must not resume silently."""
    runtime = build_runtime(
        tmp_path
    )
    early_stopping = build_early_stopping_from_config(
        runtime.config
    )
    result = build_epoch_result(
        f1=0.20
    )

    update = update_epoch_checkpoints(
        runtime=runtime,
        epoch=1,
        training_result=result,
        validation_result=result,
        early_stopping=early_stopping,
    )

    runtime.config[
        "checkpointing"
    ][
        "monitor"
    ] = "validation_iou"

    with pytest.raises(
        TrainingConfigurationError,
        match="does not match",
    ):
        load_training_checkpoint(
            path=update.last_checkpoint,
            runtime=runtime,
            strict_config=True,
        )

def test_restore_random_states_normalizes_cpu_rng_dtype() -> None:
    states = capture_random_states()

    states[
        "torch_cpu"
    ] = states[
        "torch_cpu"
    ].to(
        dtype=torch.int16
    )

    restore_random_states(
        states
    )

    restored_state = torch.get_rng_state()

    assert restored_state.dtype == torch.uint8
    assert restored_state.device.type == "cpu"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_restore_random_states_normalizes_cuda_mapped_tensors() -> None:
    states = capture_random_states()

    states[
        "torch_cpu"
    ] = states[
        "torch_cpu"
    ].to(
        device="cuda"
    )

    cuda_states = states[
        "torch_cuda"
    ]

    assert cuda_states is not None

    states[
        "torch_cuda"
    ] = [
        state.to(
            device="cuda"
        )
        for state in cuda_states
    ]

    restore_random_states(
        states
    )

    restored_state = torch.get_rng_state()

    assert restored_state.dtype == torch.uint8
    assert restored_state.device.type == "cpu"
