from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.evaluation.evaluate_test import (
    OSCDTestDataset,
    calculate_valid_shape,
    evaluation_starts,
    load_frozen_protocol,
)


CONFIG_PATH = Path(
    "experiments/run_full/train_config.yaml"
)

SUMMARY_PATH = Path(
    "experiments/run_full/threshold_search/"
    "threshold_search_summary.json"
)

CHECKPOINT_PATH = Path(
    "experiments/run_full/checkpoints/"
    "best_model_epoch24.pt"
)

OFFICIAL_ROOT = Path(
    "data/benchmark/oscd/week6_official"
)


def build_test_dataset(
) -> OSCDTestDataset:
    frozen = load_frozen_protocol(
        config_path=CONFIG_PATH,
        threshold_summary_path=SUMMARY_PATH,
        checkpoint_path=CHECKPOINT_PATH,
    )

    return OSCDTestDataset(
        official_root=OFFICIAL_ROOT,
        region_names=frozen.test_regions,
        band_names=frozen.bands,
        patch_size=frozen.patch_size,
        stride=frozen.stride,
        reflectance_scale=frozen.reflectance_scale,
        clip_minimum=frozen.clip_minimum,
        clip_maximum=frozen.clip_maximum,
    )


def test_evaluation_starts_use_exact_grid(
) -> None:
    assert evaluation_starts(
        241,
        256,
        256,
    ) == (
        0,
    )

    assert evaluation_starts(
        256,
        256,
        256,
    ) == (
        0,
    )

    assert evaluation_starts(
        257,
        256,
        256,
    ) == (
        0,
        256,
    )

    assert evaluation_starts(
        774,
        256,
        256,
    ) == (
        0,
        256,
        512,
        768,
    )

    with pytest.raises(
        ValueError,
        match="stride",
    ):
        evaluation_starts(
            512,
            256,
            128,
        )


def test_test_patch_index_covers_every_pixel_once(
) -> None:
    dataset = build_test_dataset()

    total_valid_pixels = 0

    for region_index, region in enumerate(
        dataset.regions
    ):
        coverage = np.zeros(
            (
                region.height,
                region.width,
            ),
            dtype=np.uint8,
        )

        region_patches = (
            patch
            for patch in dataset.patches
            if patch.region_index == region_index
        )

        for patch in region_patches:
            valid_height, valid_width = calculate_valid_shape(
                region_height=region.height,
                region_width=region.width,
                patch=patch,
            )

            coverage[
                patch.row:
                patch.row + valid_height,
                patch.column:
                patch.column + valid_width,
            ] += 1

        assert int(
            coverage.min()
        ) == 1

        assert int(
            coverage.max()
        ) == 1

        assert int(
            coverage.sum()
        ) == (
            region.height
            * region.width
        )

        total_valid_pixels += int(
            coverage.sum()
        )

    assert total_valid_pixels == 3_077_936
    assert len(
        dataset
    ) == 69


def test_norcia_padding_and_valid_mask(
) -> None:
    dataset = build_test_dataset()

    sample_index = next(
        index
        for index, patch in enumerate(
            dataset.patches
        )
        if (
            dataset.regions[
                patch.region_index
            ].name
            == "norcia"
            and patch.row == 0
            and patch.column == 0
        )
    )

    sample = dataset[
        sample_index
    ]

    assert sample[
        "valid_height"
    ] == 241

    assert sample[
        "valid_width"
    ] == 256

    assert int(
        sample[
            "valid_mask"
        ].sum().item()
    ) == (
        241
        * 256
    )

    assert torch.count_nonzero(
        sample[
            "valid_mask"
        ][
            :,
            241:,
            :,
        ]
    ).item() == 0

    assert torch.count_nonzero(
        sample[
            "before"
        ][
            :,
            241:,
            :,
        ]
    ).item() == 0

    assert torch.count_nonzero(
        sample[
            "after"
        ][
            :,
            241:,
            :,
        ]
    ).item() == 0

    assert torch.count_nonzero(
        sample[
            "mask"
        ][
            :,
            241:,
            :,
        ]
    ).item() == 0


def test_frozen_threshold_cannot_be_modified(
    tmp_path: Path,
) -> None:
    payload = json.loads(
        SUMMARY_PATH.read_text(
            encoding="utf-8"
        )
    )

    payload[
        "best_threshold_metrics"
    ][
        "threshold"
    ] = 0.75

    modified_summary = (
        tmp_path
        / "modified_summary.json"
    )

    modified_summary.write_text(
        json.dumps(
            payload
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="selected threshold",
    ):
        load_frozen_protocol(
            config_path=CONFIG_PATH,
            threshold_summary_path=modified_summary,
            checkpoint_path=CHECKPOINT_PATH,
        )


from src.evaluation.evaluate_test import (
    BinaryCounts,
    calculate_macro_metrics,
    calculate_patch_counts,
    counts_to_metrics,
    merge_counts,
)


def test_patch_confusion_counts_respect_valid_mask(
) -> None:
    probabilities = torch.tensor(
        [
            [
                [0.90, 0.80],
                [0.20, 0.99],
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

    valid_mask = torch.tensor(
        [
            [
                [1.0, 1.0],
                [1.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )

    counts = calculate_patch_counts(
        probabilities=probabilities,
        targets=targets,
        valid_mask=valid_mask,
        threshold=0.76,
    )

    assert counts == BinaryCounts(
        true_positive=1,
        false_positive=1,
        false_negative=1,
        true_negative=0,
        ignored_pixels=1,
        evaluated_pixels=3,
    )

    metrics = counts_to_metrics(
        counts
    )

    assert metrics[
        "precision"
    ] == pytest.approx(
        0.5
    )

    assert metrics[
        "recall"
    ] == pytest.approx(
        0.5
    )

    assert metrics[
        "f1"
    ] == pytest.approx(
        0.5
    )

    assert metrics[
        "iou"
    ] == pytest.approx(
        1.0
        / 3.0
    )

    assert metrics[
        "accuracy"
    ] == pytest.approx(
        1.0
        / 3.0
    )


def test_count_merging_is_additive(
) -> None:
    first = BinaryCounts(
        true_positive=2,
        false_positive=3,
        false_negative=4,
        true_negative=5,
        ignored_pixels=6,
        evaluated_pixels=14,
    )

    second = BinaryCounts(
        true_positive=7,
        false_positive=8,
        false_negative=9,
        true_negative=10,
        ignored_pixels=11,
        evaluated_pixels=34,
    )

    assert merge_counts(
        first,
        second,
    ) == BinaryCounts(
        true_positive=9,
        false_positive=11,
        false_negative=13,
        true_negative=15,
        ignored_pixels=17,
        evaluated_pixels=48,
    )


def test_macro_metrics_report_mean_and_median(
) -> None:
    per_region = [
        {
            "region": "a",
            "precision": 0.1,
            "recall": 0.2,
            "f1": 0.3,
            "iou": 0.4,
            "accuracy": 0.5,
            "change_prevalence": 0.6,
            "predicted_change_fraction": 0.7,
        },
        {
            "region": "b",
            "precision": 0.3,
            "recall": 0.4,
            "f1": 0.5,
            "iou": 0.6,
            "accuracy": 0.7,
            "change_prevalence": 0.8,
            "predicted_change_fraction": 0.9,
        },
        {
            "region": "c",
            "precision": 0.8,
            "recall": 0.9,
            "f1": 1.0,
            "iou": 0.2,
            "accuracy": 0.4,
            "change_prevalence": 0.1,
            "predicted_change_fraction": 0.2,
        },
    ]

    macro = calculate_macro_metrics(
        per_region
    )

    assert macro[
        "mean"
    ][
        "f1"
    ] == pytest.approx(
        0.6
    )

    assert macro[
        "median"
    ][
        "f1"
    ] == pytest.approx(
        0.5
    )

    assert macro[
        "mean"
    ][
        "precision"
    ] == pytest.approx(
        0.4
    )

    assert macro[
        "median"
    ][
        "precision"
    ] == pytest.approx(
        0.3
    )
