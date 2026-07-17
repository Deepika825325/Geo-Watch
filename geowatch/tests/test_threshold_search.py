"""Tests for GeoWatch validation threshold selection."""

from __future__ import annotations

import numpy as np
import pytest

from src.training.threshold_search import (
    build_threshold_grid,
    calculate_threshold_metrics,
    select_best_threshold,
)


def test_threshold_grid_is_inclusive_and_deterministic() -> None:
    """The requested inclusive threshold grid is constructed exactly."""

    thresholds = build_threshold_grid(
        minimum=0.1,
        maximum=0.5,
        step=0.1,
    )

    assert thresholds == (
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
    )


def test_threshold_metrics_count_positive_change_pixels() -> None:
    """Positive-change confusion counts produce the expected metrics."""

    probabilities = np.asarray(
        [
            0.1,
            0.4,
            0.6,
            0.9,
        ],
        dtype=np.float32,
    )
    targets = np.asarray(
        [
            0,
            1,
            1,
            0,
        ],
        dtype=np.uint8,
    )

    result = calculate_threshold_metrics(
        probabilities=probabilities,
        targets=targets,
        threshold=0.3,
    )

    assert result.true_positive == 2
    assert result.false_positive == 1
    assert result.false_negative == 0
    assert result.true_negative == 1
    assert result.precision == pytest.approx(
        2.0 / 3.0
    )
    assert result.recall == pytest.approx(
        1.0
    )
    assert result.f1 == pytest.approx(
        0.8
    )
    assert result.iou == pytest.approx(
        2.0 / 3.0
    )


def test_best_threshold_maximizes_f1() -> None:
    """F1 selection returns the strongest candidate threshold."""

    probabilities = np.asarray(
        [
            0.1,
            0.4,
            0.6,
            0.9,
        ],
        dtype=np.float32,
    )
    targets = np.asarray(
        [
            0,
            1,
            1,
            0,
        ],
        dtype=np.uint8,
    )

    results = tuple(
        calculate_threshold_metrics(
            probabilities=probabilities,
            targets=targets,
            threshold=threshold,
        )
        for threshold in (
            0.3,
            0.5,
            0.7,
        )
    )

    selected = select_best_threshold(
        results=results,
        objective="f1",
    )

    assert selected.threshold == pytest.approx(
        0.3
    )
    assert selected.f1 == pytest.approx(
        0.8
    )
