from __future__ import annotations

import pytest

from src.evaluation.failure_analysis import (
    analyze_region,
    build_generalization_comparison,
    determine_error_direction,
    determine_rank_tier,
)


def test_error_direction(
) -> None:
    assert determine_error_direction(
        10,
        5,
    ) == "false_positive_dominant"

    assert determine_error_direction(
        5,
        10,
    ) == "false_negative_dominant"

    assert determine_error_direction(
        5,
        5,
    ) == "balanced"


def test_rank_tiers(
) -> None:
    assert determine_rank_tier(
        1,
        10,
    ) == "top"

    assert determine_rank_tier(
        4,
        10,
    ) == "middle"

    assert determine_rank_tier(
        8,
        10,
    ) == "bottom"


def test_region_analysis(
) -> None:
    region = {
        "region": "sample",
        "height": 10,
        "width": 10,
        "patch_count": 1,
        "precision": 0.4,
        "recall": 0.8,
        "f1": 0.5333333333,
        "iou": 0.3636363636,
        "accuracy": 0.83,
        "change_prevalence": 0.1,
        "predicted_change_fraction": 0.2,
        "true_positive": 8,
        "false_positive": 12,
        "false_negative": 2,
        "true_negative": 78,
        "ignored_pixels": 0,
        "evaluated_pixels": 100,
    }

    analysis = analyze_region(
        region=region,
        rank=2,
        total_regions=10,
    )

    assert analysis[
        "rank_tier"
    ] == "top"

    assert analysis[
        "prediction_bias"
    ] == "overprediction"

    assert analysis[
        "dominant_error"
    ] == "false_positive_dominant"

    assert analysis[
        "prediction_to_truth_ratio"
    ] == pytest.approx(
        2.0
    )

    assert analysis[
        "total_error_rate"
    ] == pytest.approx(
        0.14
    )


def test_generalization_comparison(
) -> None:
    test = {
        "precision": 0.4,
        "recall": 0.6,
        "f1": 0.48,
        "iou": 0.32,
        "accuracy": 0.93,
    }

    validation = {
        "precision": 0.42,
        "recall": 0.42,
        "f1": 0.42,
        "iou": 0.27,
        "accuracy": 0.91,
    }

    result = build_generalization_comparison(
        test_micro=test,
        validation_metrics=validation,
    )

    assert result[
        "delta_test_minus_validation"
    ][
        "f1"
    ] == pytest.approx(
        0.06
    )

    assert result[
        "delta_test_minus_validation"
    ][
        "precision"
    ] == pytest.approx(
        -0.02
    )
