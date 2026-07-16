"""Leakage-safe geographic dataset splitting for GeoWatch.

This module assigns accepted GeoWatch patches to contiguous geographic
train, validation, and test blocks. It does not randomly shuffle individual
patches because neighbouring satellite patches are spatially correlated.

The default configuration divides patch columns from west to east:

* western columns: training;
* following columns: validation;
* eastern columns: testing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


LOGGER = logging.getLogger("geowatch.split")


class ConfigurationError(ValueError):
    """Raised when split configuration is invalid."""


class SplitError(RuntimeError):
    """Raised when geographic splitting cannot be completed safely."""


@dataclass(frozen=True)
class SplitSettings:
    """Validated geographic split settings."""

    patch_index_path: Path
    tiling_report_path: Path
    split_directory: Path
    split_report_path: Path
    split_axis: str
    split_order: tuple[str, ...]
    split_ratios: Mapping[str, float]


def require_mapping(
    value: object,
    context: str,
) -> Mapping[str, Any]:
    """Validate and return a mapping value."""
    if not isinstance(value, Mapping):
        raise ConfigurationError(
            f"Value '{context}' must be a mapping."
        )

    return value


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load a GeoWatch YAML configuration."""
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file does not exist: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8-sig",
    ) as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ConfigurationError(
            "Configuration root must be a YAML mapping."
        )

    return config


def load_json_mapping(json_path: Path) -> Mapping[str, Any]:
    """Load a JSON document whose root must be an object."""
    if not json_path.is_file():
        raise FileNotFoundError(
            f"JSON file does not exist: {json_path}"
        )

    try:
        with json_path.open(
            "r",
            encoding="utf-8",
        ) as json_file:
            payload = json.load(json_file)
    except json.JSONDecodeError as error:
        raise SplitError(
            f"Invalid JSON document: {json_path}"
        ) from error

    if not isinstance(payload, Mapping):
        raise SplitError(
            f"JSON root must be an object: {json_path}"
        )

    return payload


def calculate_sha256(file_path: Path) -> str:
    """Calculate the SHA-256 checksum of a file."""
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Cannot hash missing file: {file_path}"
        )

    digest = hashlib.sha256()

    with file_path.open("rb") as input_file:
        while chunk := input_file.read(1_048_576):
            digest.update(chunk)

    return digest.hexdigest()


def write_json_atomic(
    payload: Mapping[str, Any],
    output_path: Path,
) -> None:
    """Write JSON atomically."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        f"{output_path.suffix}.tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as output_file:
        json.dump(
            payload,
            output_file,
            indent=2,
            ensure_ascii=False,
        )
        output_file.write("\n")

    temporary_path.replace(output_path)


def parse_boolean(value: str) -> bool:
    """Parse a CSV boolean value."""
    normalized = value.strip().lower()

    if normalized in {"true", "1", "yes"}:
        return True

    if normalized in {"false", "0", "no"}:
        return False

    raise SplitError(
        f"Invalid boolean value in patch index: {value!r}"
    )


def load_settings(
    config_path: Path,
    patch_index_override: Path | None,
    output_directory_override: Path | None,
    report_override: Path | None,
    axis_override: str | None,
) -> SplitSettings:
    """Load and validate geographic split settings."""
    config = load_yaml_config(config_path)

    processing = require_mapping(
        config.get("processing"),
        "processing",
    )

    patch_index_path = (
        patch_index_override
        if patch_index_override is not None
        else Path(
            str(
                processing.get(
                    "patch_index",
                    "data/processed/patches/patch_index.csv",
                )
            )
        )
    )

    tiling_report_path = Path(
        str(
            processing.get(
                "tiling_report",
                "data/processed/patches/tiling_report.json",
            )
        )
    )

    split_directory = (
        output_directory_override
        if output_directory_override is not None
        else Path(
            str(
                processing.get(
                    "split_dir",
                    "data/processed/splits",
                )
            )
        )
    )

    split_report_path = (
        report_override
        if report_override is not None
        else Path(
            str(
                processing.get(
                    "split_report",
                    "data/processed/splits/split_report.json",
                )
            )
        )
    )

    split_axis = (
        axis_override
        if axis_override is not None
        else str(
            processing.get(
                "geographic_split_axis",
                "column",
            )
        )
    ).strip().lower()

    if split_axis not in {"column", "row"}:
        raise ConfigurationError(
            "geographic_split_axis must be 'column' or 'row'."
        )

    configured_order = processing.get(
        "geographic_split_order",
        ["train", "validation", "test"],
    )

    if (
        not isinstance(configured_order, Sequence)
        or isinstance(configured_order, (str, bytes))
    ):
        raise ConfigurationError(
            "geographic_split_order must be a list."
        )

    split_order = tuple(
        str(value).strip().lower()
        for value in configured_order
    )

    expected_splits = {
        "train",
        "validation",
        "test",
    }

    if (
        len(split_order) != 3
        or set(split_order) != expected_splits
    ):
        raise ConfigurationError(
            "geographic_split_order must contain train, "
            "validation and test exactly once."
        )

    configured_ratios = require_mapping(
        processing.get(
            "split_ratios",
            {
                "train": 0.60,
                "validation": 0.20,
                "test": 0.20,
            },
        ),
        "processing.split_ratios",
    )

    split_ratios: dict[str, float] = {}

    for split_name in expected_splits:
        try:
            ratio = float(
                configured_ratios.get(split_name)
            )
        except (TypeError, ValueError) as error:
            raise ConfigurationError(
                f"Invalid ratio for split '{split_name}'."
            ) from error

        if ratio <= 0:
            raise ConfigurationError(
                f"Split ratio for '{split_name}' "
                "must be greater than zero."
            )

        split_ratios[split_name] = ratio

    ratio_sum = sum(split_ratios.values())

    if not math.isclose(
        ratio_sum,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ConfigurationError(
            f"Split ratios must sum to 1.0; received {ratio_sum}."
        )

    return SplitSettings(
        patch_index_path=patch_index_path,
        tiling_report_path=tiling_report_path,
        split_directory=split_directory,
        split_report_path=split_report_path,
        split_axis=split_axis,
        split_order=split_order,
        split_ratios=split_ratios,
    )


def read_patch_index(
    patch_index_path: Path,
) -> tuple[list[str], list[dict[str, str]]]:
    """Read and validate the patch-index CSV."""
    if not patch_index_path.is_file():
        raise FileNotFoundError(
            f"Patch index does not exist: {patch_index_path}"
        )

    with patch_index_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as index_file:
        reader = csv.DictReader(index_file)

        if reader.fieldnames is None:
            raise SplitError(
                "Patch index contains no header."
            )

        fieldnames = list(reader.fieldnames)
        rows = [
            dict(row)
            for row in reader
        ]

    required_fields = {
        "patch_id",
        "row_index",
        "column_index",
        "accepted",
        "before_path",
        "after_path",
        "mask_path",
        "left",
        "bottom",
        "right",
        "top",
        "split",
    }

    missing_fields = required_fields.difference(
        fieldnames
    )

    if missing_fields:
        raise SplitError(
            "Patch index is missing required columns: "
            f"{sorted(missing_fields)}"
        )

    if not rows:
        raise SplitError(
            "Patch index contains no records."
        )

    patch_ids = [
        row["patch_id"].strip()
        for row in rows
    ]

    if any(not patch_id for patch_id in patch_ids):
        raise SplitError(
            "Patch index contains an empty patch ID."
        )

    if len(patch_ids) != len(set(patch_ids)):
        raise SplitError(
            "Patch index contains duplicate patch IDs."
        )

    return fieldnames, rows


def allocate_group_counts(
    number_of_groups: int,
    split_order: Sequence[str],
    split_ratios: Mapping[str, float],
) -> dict[str, int]:
    """Allocate complete geographic units to each split.

    Every split receives at least one complete geographic unit.
    """
    if number_of_groups < len(split_order):
        raise SplitError(
            "At least three geographic rows or columns are required "
            "to create train, validation and test splits."
        )

    targets = {
        split_name: (
            number_of_groups
            * float(split_ratios[split_name])
        )
        for split_name in split_order
    }

    counts = {
        split_name: max(
            1,
            int(math.floor(targets[split_name])),
        )
        for split_name in split_order
    }

    while sum(counts.values()) > number_of_groups:
        candidates = [
            split_name
            for split_name in split_order
            if counts[split_name] > 1
        ]

        if not candidates:
            raise SplitError(
                "Could not allocate geographic groups safely."
            )

        split_to_reduce = max(
            candidates,
            key=lambda split_name: (
                counts[split_name]
                - targets[split_name]
            ),
        )

        counts[split_to_reduce] -= 1

    while sum(counts.values()) < number_of_groups:
        split_to_expand = max(
            split_order,
            key=lambda split_name: (
                targets[split_name]
                - counts[split_name]
            ),
        )

        counts[split_to_expand] += 1

    return counts


def build_geographic_assignment(
    accepted_rows: Sequence[Mapping[str, str]],
    settings: SplitSettings,
) -> tuple[dict[int, str], dict[str, list[int]]]:
    """Assign complete ordered rows or columns to dataset splits."""
    axis_field = (
        "column_index"
        if settings.split_axis == "column"
        else "row_index"
    )

    try:
        geographic_units = sorted(
            {
                int(row[axis_field])
                for row in accepted_rows
            }
        )
    except (TypeError, ValueError, KeyError) as error:
        raise SplitError(
            f"Invalid {axis_field} value in patch index."
        ) from error

    counts = allocate_group_counts(
        number_of_groups=len(geographic_units),
        split_order=settings.split_order,
        split_ratios=settings.split_ratios,
    )

    assignment: dict[int, str] = {}
    split_units: dict[str, list[int]] = {
        split_name: []
        for split_name in settings.split_order
    }

    cursor = 0

    for split_name in settings.split_order:
        unit_count = counts[split_name]
        selected_units = geographic_units[
            cursor : cursor + unit_count
        ]

        if not selected_units:
            raise SplitError(
                f"No geographic units were assigned to {split_name}."
            )

        split_units[split_name] = selected_units

        for unit in selected_units:
            if unit in assignment:
                raise SplitError(
                    f"Geographic unit {unit} was assigned twice."
                )

            assignment[unit] = split_name

        cursor += unit_count

    if cursor != len(geographic_units):
        raise SplitError(
            "Not all geographic units were assigned."
        )

    return assignment, split_units


def validate_patch_files(
    rows: Sequence[Mapping[str, str]],
) -> None:
    """Ensure all accepted patch artifacts exist."""
    missing_files: list[str] = []

    for row in rows:
        for field in (
            "before_path",
            "after_path",
            "mask_path",
        ):
            file_path = Path(
                row.get(field, "").strip()
            )

            if not file_path.is_file():
                missing_files.append(
                    f"{row.get('patch_id')}:{field}:{file_path}"
                )

    if missing_files:
        raise FileNotFoundError(
            "Accepted patch files are missing: "
            f"{missing_files[:20]}"
        )


def calculate_extent(
    rows: Sequence[Mapping[str, str]],
) -> dict[str, float]:
    """Calculate the projected geographic extent of a split."""
    if not rows:
        raise SplitError(
            "Cannot calculate an extent for an empty split."
        )

    try:
        return {
            "left": min(float(row["left"]) for row in rows),
            "bottom": min(float(row["bottom"]) for row in rows),
            "right": max(float(row["right"]) for row in rows),
            "top": max(float(row["top"]) for row in rows),
        }
    except (TypeError, ValueError, KeyError) as error:
        raise SplitError(
            "Invalid projected bounds in patch index."
        ) from error


def write_csv_atomic(
    rows: Sequence[Mapping[str, str]],
    fieldnames: Sequence[str],
    output_path: Path,
) -> None:
    """Write CSV rows atomically."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        f"{output_path.suffix}.tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field: row.get(field, "")
                    for field in fieldnames
                }
            )

    temporary_path.replace(output_path)


def validate_disjoint_splits(
    split_rows: Mapping[str, Sequence[Mapping[str, str]]],
) -> None:
    """Verify that no patch ID appears in more than one split."""
    patch_sets = {
        split_name: {
            str(row["patch_id"])
            for row in rows
        }
        for split_name, rows in split_rows.items()
    }

    split_names = list(patch_sets)

    for first_index, first_name in enumerate(split_names):
        for second_name in split_names[first_index + 1 :]:
            overlap = (
                patch_sets[first_name]
                & patch_sets[second_name]
            )

            if overlap:
                raise SplitError(
                    f"Patch leakage between {first_name} and "
                    f"{second_name}: {sorted(overlap)}"
                )


def run_geographic_split(
    settings: SplitSettings,
) -> dict[str, Any]:
    """Create deterministic contiguous geographic dataset splits."""
    fieldnames, all_rows = read_patch_index(
        settings.patch_index_path
    )

    tiling_report = load_json_mapping(
        settings.tiling_report_path
    )

    if tiling_report.get("status") != "success":
        raise SplitError(
            "Tiling report does not have success status."
        )

    patch_size = int(
        tiling_report.get("patch_size_pixels", 0)
    )
    patch_stride = int(
        tiling_report.get("patch_stride_pixels", 0)
    )

    if patch_size <= 0 or patch_stride <= 0:
        raise SplitError(
            "Tiling report has invalid patch dimensions."
        )

    if patch_stride != patch_size:
        raise SplitError(
            "Strict geographic splitting currently requires "
            "non-overlapping patches where stride equals patch size."
        )

    accepted_rows = [
        row
        for row in all_rows
        if parse_boolean(row["accepted"])
    ]

    if not accepted_rows:
        raise SplitError(
            "No accepted patches are available for splitting."
        )

    validate_patch_files(accepted_rows)

    geographic_assignment, split_units = (
        build_geographic_assignment(
            accepted_rows=accepted_rows,
            settings=settings,
        )
    )

    axis_field = (
        "column_index"
        if settings.split_axis == "column"
        else "row_index"
    )

    split_rows: dict[str, list[dict[str, str]]] = {
        split_name: []
        for split_name in settings.split_order
    }

    accepted_patch_ids: set[str] = set()

    for row in all_rows:
        if not parse_boolean(row["accepted"]):
            row["split"] = ""
            continue

        geographic_unit = int(row[axis_field])
        split_name = geographic_assignment.get(
            geographic_unit
        )

        if split_name is None:
            raise SplitError(
                f"No split assignment for geographic unit "
                f"{geographic_unit}."
            )

        row["split"] = split_name
        split_rows[split_name].append(row)
        accepted_patch_ids.add(row["patch_id"])

    assigned_patch_ids = {
        row["patch_id"]
        for rows in split_rows.values()
        for row in rows
    }

    if assigned_patch_ids != accepted_patch_ids:
        raise SplitError(
            "Accepted patches were not assigned exactly once."
        )

    validate_disjoint_splits(split_rows)

    settings.split_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_outputs: dict[str, Any] = {}

    for split_name in settings.split_order:
        rows = sorted(
            split_rows[split_name],
            key=lambda row: (
                int(row["row_index"]),
                int(row["column_index"]),
            ),
        )

        output_path = (
            settings.split_directory
            / f"{split_name}.csv"
        )

        write_csv_atomic(
            rows=rows,
            fieldnames=fieldnames,
            output_path=output_path,
        )

        split_outputs[split_name] = {
            "patch_count": len(rows),
            "geographic_units": split_units[split_name],
            "extent": calculate_extent(rows),
            "output_path": str(output_path),
            "sha256": calculate_sha256(output_path),
            "patch_ids": [
                row["patch_id"]
                for row in rows
            ],
        }

        print(
            f"  {split_name}: {len(rows)} patches, "
            f"{settings.split_axis}s="
            f"{split_units[split_name]}"
        )

    write_csv_atomic(
        rows=all_rows,
        fieldnames=fieldnames,
        output_path=settings.patch_index_path,
    )

    total_accepted = len(accepted_rows)
    actual_ratios = {
        split_name: (
            len(split_rows[split_name])
            / total_accepted
        )
        for split_name in settings.split_order
    }

    report = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "success",
        "strategy": "contiguous_geographic_blocks",
        "split_axis": settings.split_axis,
        "split_order": list(settings.split_order),
        "configured_ratios": dict(
            settings.split_ratios
        ),
        "actual_ratios": actual_ratios,
        "source_patch_index": str(
            settings.patch_index_path
        ),
        "source_tiling_report": str(
            settings.tiling_report_path
        ),
        "source_tiling_report_sha256": (
            calculate_sha256(
                settings.tiling_report_path
            )
        ),
        "patch_size_pixels": patch_size,
        "patch_stride_pixels": patch_stride,
        "overlapping_patches": False,
        "accepted_patch_count": total_accepted,
        "assigned_patch_count": len(
            assigned_patch_ids
        ),
        "unassigned_patch_count": (
            total_accepted
            - len(assigned_patch_ids)
        ),
        "patch_index_sha256_after_assignment": (
            calculate_sha256(
                settings.patch_index_path
            )
        ),
        "splits": split_outputs,
    }

    write_json_atomic(
        payload=report,
        output_path=settings.split_report_path,
    )

    return report


def print_summary(
    report: Mapping[str, Any],
    settings: SplitSettings,
) -> None:
    """Print geographic split completion details."""
    splits = require_mapping(
        report.get("splits"),
        "split_report.splits",
    )

    print("GeoWatch geographic dataset split completed")
    print(f"  Status: {report.get('status')}")
    print(
        f"  Strategy: {report.get('strategy')}"
    )
    print(
        f"  Axis: {report.get('split_axis')}"
    )
    print(
        f"  Accepted patches: "
        f"{report.get('accepted_patch_count')}"
    )

    for split_name in settings.split_order:
        split = require_mapping(
            splits.get(split_name),
            f"split_report.splits.{split_name}",
        )

        print(
            f"  {split_name.title()}: "
            f"{split.get('patch_count')} patches "
            f"({float(report['actual_ratios'][split_name]):.2%})"
        )

    print(
        f"  Unassigned patches: "
        f"{report.get('unassigned_patch_count')}"
    )
    print(
        f"  Updated patch index: "
        f"{settings.patch_index_path}"
    )
    print(
        f"  Split report: "
        f"{settings.split_report_path}"
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the geographic split command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Assign GeoWatch patches to contiguous geographic "
            "train, validation and test blocks."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data_config.yaml"),
        help=(
            "GeoWatch configuration path. "
            "Default: configs/data_config.yaml"
        ),
    )

    parser.add_argument(
        "--patch-index",
        type=Path,
        default=None,
        help="Optional patch-index CSV override.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional split-output directory override.",
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional split-report JSON override.",
    )

    parser.add_argument(
        "--axis",
        choices=("column", "row"),
        default=None,
        help=(
            "Geographic split axis override. "
            "Default comes from configuration."
        ),
    )

    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging level. Default: INFO",
    )

    return parser


def main() -> int:
    """Run leakage-safe geographic splitting."""
    parser = build_argument_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    try:
        settings = load_settings(
            config_path=args.config,
            patch_index_override=args.patch_index,
            output_directory_override=args.output_dir,
            report_override=args.report,
            axis_override=args.axis,
        )

        report = run_geographic_split(
            settings=settings,
        )

        print_summary(
            report=report,
            settings=settings,
        )

        return 0

    except (
        FileNotFoundError,
        PermissionError,
        ConfigurationError,
        SplitError,
        ValueError,
        yaml.YAMLError,
    ) as error:
        LOGGER.error("%s", error)
        return 1

    except Exception:
        LOGGER.exception(
            "Unexpected geographic split failure."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
