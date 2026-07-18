from __future__ import annotations

import numpy as np
import pytest
import torch

from src.evaluation.failure_gallery import (
    build_error_masks,
    normalize_rgb_pair,
    select_top_records,
)


def test_error_masks_exclude_invalid_pixels(
) -> None:
    probabilities = torch.tensor(
        [
            [
                [0.90, 0.90],
                [0.10, 0.99],
            ]
        ],
        dtype=torch.float32,
    )

    targets = torch.tensor(
        [
            [
                [1.0, 0.0],
                [1.0, 1.0],
            ]
        ],
        dtype=torch.float32,
    )

    valid = torch.tensor(
        [
            [
                [1.0, 1.0],
                [1.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )

    masks = build_error_masks(
        probabilities=probabilities,
        targets=targets,
        valid_mask=valid,
        threshold=0.76,
    )

    assert int(
        masks[
            "true_positive"
        ].sum()
    ) == 1

    assert int(
        masks[
            "false_positive"
        ].sum()
    ) == 1

    assert int(
        masks[
            "false_negative"
        ].sum()
    ) == 1

    assert not masks[
        "prediction"
    ][
        1,
        1,
    ]


def test_rgb_normalization_shape_and_range(
) -> None:
    before = torch.arange(
        4
        * 4
        * 4,
        dtype=torch.float32,
    ).reshape(
        4,
        4,
        4,
    )

    after = before + 10.0

    valid = torch.ones(
        (
            1,
            4,
            4,
        ),
        dtype=torch.float32,
    )

    before_rgb, after_rgb = normalize_rgb_pair(
        before=before,
        after=after,
        valid_mask=valid,
    )

    assert before_rgb.shape == (
        4,
        4,
        3,
    )

    assert after_rgb.shape == (
        4,
        4,
        3,
    )

    assert np.isfinite(
        before_rgb
    ).all()

    assert np.isfinite(
        after_rgb
    ).all()

    assert float(
        before_rgb.min()
    ) >= 0.0

    assert float(
        after_rgb.max()
    ) <= 1.0


def test_top_record_selection_is_per_region(
) -> None:
    records = [
        {
            "region": "a",
            "patch_id": "a1",
            "false_positive_rate_all_pixels": 0.20,
            "false_negative_rate_all_pixels": 0.01,
            "total_error_rate": 0.21,
        },
        {
            "region": "a",
            "patch_id": "a2",
            "false_positive_rate_all_pixels": 0.40,
            "false_negative_rate_all_pixels": 0.02,
            "total_error_rate": 0.42,
        },
        {
            "region": "b",
            "patch_id": "b1",
            "false_positive_rate_all_pixels": 0.10,
            "false_negative_rate_all_pixels": 0.50,
            "total_error_rate": 0.60,
        },
        {
            "region": "b",
            "patch_id": "b2",
            "false_positive_rate_all_pixels": 0.05,
            "false_negative_rate_all_pixels": 0.20,
            "total_error_rate": 0.25,
        },
    ]

    false_positive = select_top_records(
        records=records,
        regions=(
            "a",
        ),
        error_type="false_positive",
        count_per_region=1,
    )

    false_negative = select_top_records(
        records=records,
        regions=(
            "b",
        ),
        error_type="false_negative",
        count_per_region=1,
    )

    assert false_positive[
        0
    ][
        "patch_id"
    ] == "a2"

    assert false_negative[
        0
    ][
        "patch_id"
    ] == "b1"

    assert false_positive[
        0
    ][
        "selection_rank"
    ] == 1

    with pytest.raises(
        ValueError,
        match="Unsupported error type",
    ):
        select_top_records(
            records=records,
            regions=(
                "a",
            ),
            error_type="other",
            count_per_region=1,
        )
