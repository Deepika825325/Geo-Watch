"""Tests for the controlled GeoWatch BCE ablation configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.training.ablation_bce import (
    build_bce_ablation_config,
    normalize_for_fairness_check,
)


def build_source_config() -> dict[str, object]:
    """Create a minimal valid Dice+Focal experiment configuration."""

    return {
        "project": {
            "experiment_name": "week5_full_dice_focal",
            "seed": 42,
        },
        "paths": {
            "oscd_raw_root": "data/benchmark/oscd/raw",
            "checkpoint_directory": "experiments/run_full/checkpoints",
            "log_directory": "experiments/run_full/logs",
            "prediction_directory": "experiments/run_full/predictions",
        },
        "protocol": {
            "official_test_regions_sealed": True,
        },
        "dataset": {
            "train_regions": [
                "abudhabi",
                "aguasclaras",
            ],
            "validation_regions": [
                "hongkong",
                "mumbai",
                "paris",
            ],
            "loader": {
                "batch_size": 16,
            },
        },
        "training": {
            "epochs": 50,
        },
        "optimizer": {
            "learning_rate": 1.0e-4,
        },
        "metrics": {
            "threshold": 0.5,
        },
        "loss": {
            "name": "dice_focal",
        },
    }


def test_bce_ablation_changes_only_controlled_fields() -> None:
    """The generated config must preserve every non-ablation field."""

    source = build_source_config()

    generated = build_bce_ablation_config(
        source_config=source,
        experiment_root=Path(
            "experiments/run_ablation_bce"
        ),
    )

    assert generated[
        "loss"
    ] == {
        "name": "bce",
        "reduction": "mean",
    }

    assert generated[
        "project"
    ][
        "experiment_name"
    ] == "week5_ablation_bce"

    assert (
        normalize_for_fairness_check(
            source
        )
        == normalize_for_fairness_check(
            generated
        )
    )


def test_bce_ablation_rejects_nondefault_training_threshold() -> None:
    """The controlled training comparison must retain threshold 0.5."""

    source = build_source_config()
    source[
        "metrics"
    ][
        "threshold"
    ] = 0.76

    with pytest.raises(
        ValueError,
        match="retain threshold 0.5",
    ):
        build_bce_ablation_config(
            source_config=source,
            experiment_root=Path(
                "experiments/run_ablation_bce"
            ),
        )
