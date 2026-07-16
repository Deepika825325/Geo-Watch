"""Exploratory data analysis for the GeoWatch OSCD benchmark.

This module performs quantitative analysis only on the labelled OSCD
benchmark. GeoWatch's custom Hyderabad AOI is intentionally excluded from
quantitative evaluation because those patches do not have verified
ground-truth change masks.

Outputs include:

* per-region label statistics;
* aggregate train, test and overall class distributions;
* a region-level change-fraction chart;
* representative before/after/ground-truth visualizations.

The OSCD test labels are used here only for descriptive dataset analysis.
They must not be used later for baseline threshold selection or model
hyperparameter tuning.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.errors import RasterioIOError

from src.evaluation.metrics import load_mask


LOGGER = logging.getLogger("geowatch.eda")


class EDAError(RuntimeError):
    """Raised when OSCD exploratory analysis cannot be completed safely."""


@dataclass(frozen=True)
class OSCDPaths:
    """Resolved OSCD image and label directories."""

    images_root: Path
    train_labels_root: Path
    test_labels_root: Path


@dataclass(frozen=True)
class RegionStatistics:
    """Label and image metadata for one OSCD region."""

    region: str
    split: str
    date_1: str
    date_2: str
    width: int
    height: int
    total_pixels: int
    changed_pixels: int
    unchanged_pixels: int
    change_fraction: float
    unchanged_fraction: float
    unchanged_to_change_ratio: float | None
    date_1_band_count: int
    date_2_band_count: int
    image_region_path: str
    label_path: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable region record."""
        return asdict(self)


def find_unique_directory(
    root: Path,
    required_text: str,
) -> Path:
    """Find exactly one child directory containing required text."""
    if not root.is_dir():
        raise FileNotFoundError(
            f"OSCD raw directory does not exist: {root}"
        )

    matches = sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
        and required_text.lower() in path.name.lower()
    )

    if not matches:
        raise EDAError(
            f"Could not find an OSCD directory containing "
            f"{required_text!r} under {root}."
        )

    if len(matches) > 1:
        raise EDAError(
            f"Multiple OSCD directories matched {required_text!r}: "
            f"{[str(path) for path in matches]}"
        )

    return matches[0]


def resolve_oscd_paths(raw_root: Path) -> OSCDPaths:
    """Resolve OSCD image, training-label and test-label roots."""
    return OSCDPaths(
        images_root=find_unique_directory(
            raw_root,
            "Images",
        ),
        train_labels_root=find_unique_directory(
            raw_root,
            "Train Labels",
        ),
        test_labels_root=find_unique_directory(
            raw_root,
            "Test Labels",
        ),
    )


def list_region_directories(root: Path) -> list[Path]:
    """Return sorted region directories from an OSCD root."""
    if not root.is_dir():
        raise FileNotFoundError(
            f"Region root does not exist: {root}"
        )

    regions = sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
    )

    if not regions:
        raise EDAError(
            f"No region directories were found under {root}."
        )

    return regions


def find_tiff_files(directory: Path) -> list[Path]:
    """Find TIFF files in one OSCD date directory."""
    if not directory.is_dir():
        raise FileNotFoundError(
            f"OSCD image directory does not exist: {directory}"
        )

    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".tif", ".tiff"}
    )


def normalize_band_token(value: str) -> str:
    """Normalize a band filename for robust matching."""
    return "".join(
        character
        for character in value.upper()
        if character.isalnum()
    )


def find_band_file(
    directory: Path,
    band_name: str,
) -> Path:
    """Locate an OSCD band file using canonical and short aliases."""
    canonical = normalize_band_token(band_name)

    if not canonical.startswith("B"):
        raise EDAError(
            f"Invalid Sentinel-2 band name: {band_name}"
        )

    numeric_part = canonical[1:]

    aliases = {
        canonical,
        (
            f"B{int(numeric_part)}"
            if numeric_part.isdigit()
            else canonical
        ),
    }

    candidates: list[Path] = []

    for path in find_tiff_files(directory):
        normalized_stem = normalize_band_token(
            path.stem
        )

        if (
            normalized_stem in aliases
            or any(
                normalized_stem.endswith(alias)
                for alias in aliases
            )
        ):
            candidates.append(path)

    if not candidates:
        raise EDAError(
            f"Band {band_name} was not found in {directory}."
        )

    if len(candidates) > 1:
        exact_candidates = [
            path
            for path in candidates
            if normalize_band_token(path.stem)
            == canonical
        ]

        if len(exact_candidates) == 1:
            return exact_candidates[0]

        raise EDAError(
            f"Multiple files matched band {band_name} in "
            f"{directory}: {[path.name for path in candidates]}"
        )

    return candidates[0]


def read_dates(path: Path) -> tuple[str, str]:
    """Read the two acquisition dates recorded by OSCD."""
    if not path.is_file():
        raise FileNotFoundError(
            f"OSCD dates file does not exist: {path}"
        )

    values = [
        line.strip()
        for line in path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
        if line.strip()
    ]

    if len(values) < 2:
        raise EDAError(
            f"Expected two dates in {path}, found {values}."
        )

    return values[0], values[1]


def read_single_band(path: Path) -> np.ndarray:
    """Read one OSCD raster band as float32."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Raster band does not exist: {path}"
        )

    try:
        with rasterio.open(path) as dataset:
            if dataset.count != 1:
                raise EDAError(
                    f"Expected one raster band in {path}; "
                    f"found {dataset.count}."
                )

            array = dataset.read(
                1,
                masked=True,
            ).filled(0)

    except RasterioIOError as error:
        raise EDAError(
            f"Rasterio could not read {path}."
        ) from error

    if array.ndim != 2:
        raise EDAError(
            f"Expected a two-dimensional band in {path}; "
            f"received shape {array.shape}."
        )

    return array.astype(
        np.float32,
        copy=False,
    )


def stretch_channel(
    channel: np.ndarray,
    lower_percentile: float,
    upper_percentile: float,
) -> np.ndarray:
    """Apply robust percentile stretching to one display channel."""
    finite = np.isfinite(channel)
    positive = finite & (channel > 0)

    values = (
        channel[positive]
        if np.any(positive)
        else channel[finite]
    )

    if values.size == 0:
        raise EDAError(
            "Cannot stretch a channel containing no finite values."
        )

    lower = float(
        np.percentile(
            values,
            lower_percentile,
        )
    )
    upper = float(
        np.percentile(
            values,
            upper_percentile,
        )
    )

    if upper <= lower:
        return np.zeros_like(
            channel,
            dtype=np.float32,
        )

    stretched = (
        channel.astype(np.float32)
        - lower
    ) / (upper - lower)

    return np.clip(
        stretched,
        0.0,
        1.0,
    )


def load_rgb(
    date_directory: Path,
    lower_percentile: float,
    upper_percentile: float,
) -> np.ndarray:
    """Load and stretch OSCD B04/B03/B02 as an RGB image."""
    channel_paths = [
        find_band_file(
            date_directory,
            band_name,
        )
        for band_name in (
            "B04",
            "B03",
            "B02",
        )
    ]

    channels = [
        read_single_band(path)
        for path in channel_paths
    ]

    shapes = {
        channel.shape
        for channel in channels
    }

    if len(shapes) != 1:
        raise EDAError(
            f"RGB bands do not share a common shape in "
            f"{date_directory}: {sorted(shapes)}"
        )

    stretched = [
        stretch_channel(
            channel=channel,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
        )
        for channel in channels
    ]

    return np.stack(
        stretched,
        axis=-1,
    )


def calculate_region_statistics(
    region: str,
    split: str,
    images_root: Path,
    labels_root: Path,
) -> RegionStatistics:
    """Calculate label statistics and validate one OSCD region."""
    image_region = images_root / region
    label_path = (
        labels_root
        / region
        / "cm"
        / "cm.png"
    )

    date_1_directory = (
        image_region
        / "imgs_1_rect"
    )
    date_2_directory = (
        image_region
        / "imgs_2_rect"
    )

    date_1_bands = find_tiff_files(
        date_1_directory
    )
    date_2_bands = find_tiff_files(
        date_2_directory
    )

    if len(date_1_bands) != 13:
        raise EDAError(
            f"{region} date 1 contains {len(date_1_bands)} "
            "bands; expected 13."
        )

    if len(date_2_bands) != 13:
        raise EDAError(
            f"{region} date 2 contains {len(date_2_bands)} "
            "bands; expected 13."
        )

    date_1, date_2 = read_dates(
        image_region / "dates.txt"
    )

    mask = load_mask(label_path)

    if not np.issubdtype(
        mask.dtype,
        np.integer,
    ):
        raise EDAError(
            f"OSCD label {label_path} must use an integer data "
            f"type; received {mask.dtype}."
        )

    minimum_value = int(
        np.min(mask)
    )
    maximum_value = int(
        np.max(mask)
    )

    if minimum_value < 0:
        raise EDAError(
            f"OSCD label {label_path} contains negative values."
        )

    # Some OSCD masks contain intermediate grayscale boundary values.
    # The official-compatible binary rule is:
    #     0   -> unchanged
    #     > 0 -> changed
    total_pixels = int(mask.size)
    changed_pixels = int(
        np.count_nonzero(mask > 0)
    )
    unchanged_pixels = int(
        np.count_nonzero(mask == 0)
    )

    if (
        changed_pixels
        + unchanged_pixels
        != total_pixels
    ):
        raise EDAError(
            f"Label counts do not sum to total pixels for {region}."
        )

    change_fraction = (
        changed_pixels / total_pixels
    )
    unchanged_fraction = (
        unchanged_pixels / total_pixels
    )
    imbalance_ratio = (
        unchanged_pixels / changed_pixels
        if changed_pixels > 0
        else None
    )

    return RegionStatistics(
        region=region,
        split=split,
        date_1=date_1,
        date_2=date_2,
        width=int(mask.shape[1]),
        height=int(mask.shape[0]),
        total_pixels=total_pixels,
        changed_pixels=changed_pixels,
        unchanged_pixels=unchanged_pixels,
        change_fraction=change_fraction,
        unchanged_fraction=unchanged_fraction,
        unchanged_to_change_ratio=(
            float(imbalance_ratio)
            if imbalance_ratio is not None
            else None
        ),
        date_1_band_count=len(date_1_bands),
        date_2_band_count=len(date_2_bands),
        image_region_path=str(image_region),
        label_path=str(label_path),
    )


def summarize_records(
    records: Sequence[RegionStatistics],
) -> dict[str, Any]:
    """Aggregate pixel statistics across a collection of regions."""
    if not records:
        raise EDAError(
            "Cannot summarize an empty region collection."
        )

    total_pixels = sum(
        record.total_pixels
        for record in records
    )
    changed_pixels = sum(
        record.changed_pixels
        for record in records
    )
    unchanged_pixels = sum(
        record.unchanged_pixels
        for record in records
    )

    region_change_fractions = [
        record.change_fraction
        for record in records
    ]

    return {
        "region_count": len(records),
        "total_pixels": total_pixels,
        "changed_pixels": changed_pixels,
        "unchanged_pixels": unchanged_pixels,
        "pixel_weighted_change_fraction": (
            changed_pixels / total_pixels
        ),
        "pixel_weighted_unchanged_fraction": (
            unchanged_pixels / total_pixels
        ),
        "unchanged_to_change_ratio": (
            unchanged_pixels / changed_pixels
            if changed_pixels > 0
            else None
        ),
        "mean_region_change_fraction": float(
            statistics.fmean(
                region_change_fractions
            )
        ),
        "median_region_change_fraction": float(
            statistics.median(
                region_change_fractions
            )
        ),
        "minimum_region_change_fraction": min(
            region_change_fractions
        ),
        "maximum_region_change_fraction": max(
            region_change_fractions
        ),
    }


def write_csv_atomic(
    records: Sequence[RegionStatistics],
    output_path: Path,
) -> None:
    """Write per-region statistics atomically."""
    if not records:
        raise EDAError(
            "Cannot write an empty region-statistics table."
        )

    rows = [
        record.to_dict()
        for record in records
    ]
    fieldnames = list(
        rows[0].keys()
    )

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
        writer.writerows(rows)

    temporary_path.replace(
        output_path
    )


def write_json_atomic(
    payload: Mapping[str, Any],
    output_path: Path,
) -> None:
    """Write the EDA summary atomically."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        f"{output_path.suffix}.tmp"
    )

    temporary_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(
        output_path
    )


def select_representative_records(
    records: Sequence[RegionStatistics],
    maximum_examples: int,
) -> list[RegionStatistics]:
    """Select non-cherry-picked examples near each split's median."""
    if maximum_examples <= 0:
        raise EDAError(
            "maximum_examples must be greater than zero."
        )

    selected: list[RegionStatistics] = []

    for split in ("train", "test"):
        split_records = [
            record
            for record in records
            if record.split == split
        ]

        if not split_records:
            continue

        median_fraction = statistics.median(
            record.change_fraction
            for record in split_records
        )

        ranked = sorted(
            split_records,
            key=lambda record: (
                abs(
                    record.change_fraction
                    - median_fraction
                ),
                record.region,
            ),
        )

        split_limit = max(
            1,
            maximum_examples // 2,
        )

        selected.extend(
            ranked[:split_limit]
        )

    selected = selected[
        :maximum_examples
    ]

    if not selected:
        raise EDAError(
            "No representative examples could be selected."
        )

    return selected


def create_distribution_chart(
    records: Sequence[RegionStatistics],
    output_path: Path,
) -> None:
    """Create a region-level change-fraction bar chart."""
    ordered = sorted(
        records,
        key=lambda record: (
            record.change_fraction,
            record.region,
        ),
        reverse=True,
    )

    labels = [
        f"{record.region}\n({record.split})"
        for record in ordered
    ]
    percentages = [
        100.0 * record.change_fraction
        for record in ordered
    ]

    figure, axis = plt.subplots(
        figsize=(
            max(
                12.0,
                len(ordered) * 0.55,
            ),
            6.5,
        )
    )

    axis.bar(
        range(len(ordered)),
        percentages,
    )
    axis.set_title(
        "OSCD change-pixel fraction by region"
    )
    axis.set_ylabel(
        "Changed pixels (%)"
    )
    axis.set_xlabel(
        "Region and official OSCD split"
    )
    axis.set_xticks(
        range(len(ordered))
    )
    axis.set_xticklabels(
        labels,
        rotation=60,
        ha="right",
    )
    axis.grid(
        axis="y",
        alpha=0.3,
    )

    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def create_representative_figure(
    records: Sequence[RegionStatistics],
    images_root: Path,
    output_path: Path,
    lower_percentile: float,
    upper_percentile: float,
) -> None:
    """Create before, after and label panels for selected regions."""
    if not records:
        raise EDAError(
            "No records were supplied for visualization."
        )

    figure, axes = plt.subplots(
        nrows=len(records),
        ncols=3,
        figsize=(
            12.0,
            4.0 * len(records),
        ),
        squeeze=False,
    )

    for row_index, record in enumerate(
        records
    ):
        image_region = (
            images_root
            / record.region
        )

        before_rgb = load_rgb(
            image_region / "imgs_1_rect",
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
        )
        after_rgb = load_rgb(
            image_region / "imgs_2_rect",
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
        )
        mask = load_mask(
            Path(record.label_path)
        )

        if (
            before_rgb.shape[:2]
            != mask.shape
            or after_rgb.shape[:2]
            != mask.shape
        ):
            raise EDAError(
                f"RGB and label shapes do not match for "
                f"{record.region}: before={before_rgb.shape[:2]}, "
                f"after={after_rgb.shape[:2]}, label={mask.shape}."
            )

        axes[row_index, 0].imshow(
            before_rgb
        )
        axes[row_index, 0].set_title(
            f"{record.region} — before\n{record.date_1}"
        )

        axes[row_index, 1].imshow(
            after_rgb
        )
        axes[row_index, 1].set_title(
            f"{record.region} — after\n{record.date_2}"
        )

        binary_mask = (
            mask > 0
        ).astype(
            np.uint8
        )

        axes[row_index, 2].imshow(
            binary_mask,
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[row_index, 2].set_title(
            f"Ground truth — {record.split}\n"
            f"change={record.change_fraction:.2%}"
        )

        for column_index in range(3):
            axes[row_index, column_index].axis(
                "off"
            )

    figure.suptitle(
        "OSCD representative bi-temporal examples",
        fontsize=14,
    )
    figure.tight_layout(
        rect=(0.0, 0.0, 1.0, 0.98)
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def analyze_oscd(
    raw_root: Path,
    output_directory: Path,
    maximum_examples: int,
    lower_percentile: float,
    upper_percentile: float,
) -> dict[str, Any]:
    """Run complete labelled OSCD exploratory analysis."""
    if not (
        0.0
        <= lower_percentile
        < upper_percentile
        <= 100.0
    ):
        raise EDAError(
            "Display percentiles must satisfy "
            "0 <= lower < upper <= 100."
        )

    paths = resolve_oscd_paths(
        raw_root
    )

    train_regions = list_region_directories(
        paths.train_labels_root
    )
    test_regions = list_region_directories(
        paths.test_labels_root
    )
    image_regions = list_region_directories(
        paths.images_root
    )

    train_names = {
        path.name
        for path in train_regions
    }
    test_names = {
        path.name
        for path in test_regions
    }
    image_names = {
        path.name
        for path in image_regions
    }

    if train_names & test_names:
        raise EDAError(
            "A region appears in both official OSCD splits: "
            f"{sorted(train_names & test_names)}"
        )

    if (
        train_names
        | test_names
    ) != image_names:
        raise EDAError(
            "OSCD image regions do not match the union of "
            "training and testing label regions."
        )

    if len(train_names) != 14:
        raise EDAError(
            f"Expected 14 OSCD training regions; "
            f"found {len(train_names)}."
        )

    if len(test_names) != 10:
        raise EDAError(
            f"Expected 10 OSCD testing regions; "
            f"found {len(test_names)}."
        )

    records: list[RegionStatistics] = []

    for split, region_names, labels_root in (
        (
            "train",
            sorted(train_names),
            paths.train_labels_root,
        ),
        (
            "test",
            sorted(test_names),
            paths.test_labels_root,
        ),
    ):
        for region_name in region_names:
            record = calculate_region_statistics(
                region=region_name,
                split=split,
                images_root=paths.images_root,
                labels_root=labels_root,
            )
            records.append(record)

            print(
                f"  [{split}] {region_name}: "
                f"change={record.change_fraction:.2%}, "
                f"shape={record.height}x{record.width}"
            )

    train_records = [
        record
        for record in records
        if record.split == "train"
    ]
    test_records = [
        record
        for record in records
        if record.split == "test"
    ]

    region_csv_path = (
        output_directory
        / "oscd_region_statistics.csv"
    )
    summary_json_path = (
        output_directory
        / "oscd_dataset_statistics.json"
    )
    distribution_path = (
        output_directory
        / "oscd_change_fraction_by_region.png"
    )
    examples_path = (
        output_directory
        / "oscd_representative_examples.png"
    )

    write_csv_atomic(
        records=records,
        output_path=region_csv_path,
    )

    create_distribution_chart(
        records=records,
        output_path=distribution_path,
    )

    representative_records = (
        select_representative_records(
            records=records,
            maximum_examples=maximum_examples,
        )
    )

    create_representative_figure(
        records=representative_records,
        images_root=paths.images_root,
        output_path=examples_path,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )

    summary = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "success",
        "dataset": "OSCD",
        "quantitative_scope": (
            "OSCD labelled regions only"
        ),
        "custom_hyderabad_aoi_used_for_metrics": False,
        "official_split_preserved": True,
        "test_label_usage": (
            "Descriptive EDA and final evaluation only; "
            "not for threshold or hyperparameter selection."
        ),
        "label_encoding": {
            "raw_unchanged_value": 0,
            "binary_unchanged_value": 0,
            "binary_changed_value": 1,
            "binarization_rule": "raw mask value > 0",
        },
        "region_counts": {
            "train": len(train_records),
            "test": len(test_records),
            "overall": len(records),
        },
        "train": summarize_records(
            train_records
        ),
        "test": summarize_records(
            test_records
        ),
        "overall": summarize_records(
            records
        ),
        "representative_regions": [
            {
                "region": record.region,
                "split": record.split,
                "change_fraction": (
                    record.change_fraction
                ),
            }
            for record in representative_records
        ],
        "visualization": {
            "rgb_bands": [
                "B04",
                "B03",
                "B02",
            ],
            "lower_percentile": lower_percentile,
            "upper_percentile": upper_percentile,
        },
        "outputs": {
            "region_statistics_csv": str(
                region_csv_path
            ),
            "change_distribution_figure": str(
                distribution_path
            ),
            "representative_examples_figure": str(
                examples_path
            ),
        },
    }

    write_json_atomic(
        payload=summary,
        output_path=summary_json_path,
    )

    return {
        **summary,
        "summary_json": str(
            summary_json_path
        ),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the OSCD EDA command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze OSCD label imbalance and generate "
            "representative bi-temporal visualizations."
        )
    )

    parser.add_argument(
        "--oscd-root",
        type=Path,
        default=Path(
            "data/benchmark/oscd/raw"
        ),
        help=(
            "Extracted OSCD raw-data directory. "
            "Default: data/benchmark/oscd/raw"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "reports/week2/eda"
        ),
        help=(
            "EDA output directory. "
            "Default: reports/week2/eda"
        ),
    )
    parser.add_argument(
        "--maximum-examples",
        type=int,
        default=4,
        help=(
            "Maximum representative regions in the example figure. "
            "Default: 4"
        ),
    )
    parser.add_argument(
        "--lower-percentile",
        type=float,
        default=2.0,
        help=(
            "Lower RGB display percentile. Default: 2"
        ),
    )
    parser.add_argument(
        "--upper-percentile",
        type=float,
        default=98.0,
        help=(
            "Upper RGB display percentile. Default: 98"
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=(
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
        ),
        default="INFO",
    )

    return parser


def print_summary(
    report: Mapping[str, Any],
) -> None:
    """Print a concise EDA completion summary."""
    train = report["train"]
    test = report["test"]
    overall = report["overall"]

    print("OSCD exploratory analysis completed")
    print(
        "  Status:",
        report["status"],
    )
    print(
        "  Regions:",
        report["region_counts"]["overall"],
    )
    print(
        "  Train regions:",
        report["region_counts"]["train"],
    )
    print(
        "  Test regions:",
        report["region_counts"]["test"],
    )
    print(
        "  Train change fraction:",
        f"{train['pixel_weighted_change_fraction']:.4%}",
    )
    print(
        "  Test change fraction:",
        f"{test['pixel_weighted_change_fraction']:.4%}",
    )
    print(
        "  Overall change fraction:",
        f"{overall['pixel_weighted_change_fraction']:.4%}",
    )
    print(
        "  Overall unchanged:change ratio:",
        f"{overall['unchanged_to_change_ratio']:.2f}:1",
    )
    print(
        "  Summary JSON:",
        report["summary_json"],
    )
    print(
        "  Region CSV:",
        report["outputs"]["region_statistics_csv"],
    )
    print(
        "  Distribution figure:",
        report["outputs"][
            "change_distribution_figure"
        ],
    )
    print(
        "  Example figure:",
        report["outputs"][
            "representative_examples_figure"
        ],
    )


def main() -> int:
    """Run OSCD exploratory analysis."""
    parser = build_argument_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(
            logging,
            args.log_level,
        ),
        format="%(levelname)s: %(message)s",
    )

    try:
        report = analyze_oscd(
            raw_root=args.oscd_root,
            output_directory=args.output_dir,
            maximum_examples=args.maximum_examples,
            lower_percentile=args.lower_percentile,
            upper_percentile=args.upper_percentile,
        )

        print_summary(report)

        return 0

    except (
        EDAError,
        FileNotFoundError,
        PermissionError,
        RasterioIOError,
        ValueError,
        TypeError,
        OSError,
        csv.Error,
        json.JSONDecodeError,
    ) as error:
        LOGGER.error("%s", error)
        return 1

    except Exception:
        LOGGER.exception(
            "Unexpected OSCD EDA failure."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
