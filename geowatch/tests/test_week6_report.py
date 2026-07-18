from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation.week6_report import (
    HYDERABAD_REQUIRED_FILES,
    HyderabadRecord,
    build_region_table,
    inspect_hyderabad,
    load_json,
)


def test_load_json_requires_object(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "value.json"
    )

    path.write_text(
        json.dumps(
            [
                1,
                2,
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        TypeError,
        match="JSON object",
    ):
        load_json(
            path
        )


def test_hyderabad_status_is_pending_without_files(
    tmp_path: Path,
) -> None:
    record = inspect_hyderabad(
        tmp_path
    )

    assert record.status == "pending_input"

    assert len(
        record.missing_files
    ) == len(
        HYDERABAD_REQUIRED_FILES
    )


def test_hyderabad_status_is_input_ready(
    tmp_path: Path,
) -> None:
    root = (
        tmp_path
        / "data"
        / "qualitative"
        / "hyderabad"
    )

    for relative in HYDERABAD_REQUIRED_FILES:
        path = (
            root
            / relative
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_bytes(
            b"valid"
        )

    record = inspect_hyderabad(
        tmp_path
    )

    assert record.status == "input_ready"
    assert not record.missing_files


def test_region_table_ranks_by_f1(
) -> None:
    regions = [
        {
            "region": "weak",
            "precision": 0.1,
            "recall": 0.2,
            "f1": 0.15,
            "iou": 0.08,
            "change_prevalence": 0.01,
            "predicted_change_fraction": 0.02,
        },
        {
            "region": "strong",
            "precision": 0.8,
            "recall": 0.7,
            "f1": 0.75,
            "iou": 0.6,
            "change_prevalence": 0.1,
            "predicted_change_fraction": 0.11,
        },
    ]

    table = build_region_table(
        regions
    )

    assert table.index(
        "strong"
    ) < table.index(
        "weak"
    )
