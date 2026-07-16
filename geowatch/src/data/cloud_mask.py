"""Sentinel-2 SCL-based cloud and quality masking for GeoWatch.

This module creates period-specific valid-pixel masks and a common
bi-temporal valid mask from aligned Sentinel-2 Scene Classification Layer
rasters.

A value of 1 in an output mask means that the pixel is valid. A value of 0
means that it must be excluded from training, validation, or inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import rasterio
import yaml
from affine import Affine
from rasterio.errors import RasterioIOError


LOGGER = logging.getLogger("geowatch.cloud_mask")


class ConfigurationError(ValueError):
    """Raised when cloud-mask configuration is invalid."""


class CloudMaskError(RuntimeError):
    """Raised when cloud-mask generation or validation fails."""


@dataclass(frozen=True)
class RasterGrid:
    """Spatial grid shared by aligned GeoWatch rasters."""

    width: int
    height: int
    crs: str
    epsg: int
    transform: Affine

    def matches(self, other: "RasterGrid") -> bool:
        """Return whether another raster uses the exact same pixel grid."""
        return (
            self.width == other.width
            and self.height == other.height
            and self.epsg == other.epsg
            and self.transform == other.transform
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert grid metadata into a JSON-serializable dictionary."""
        return {
            "width": self.width,
            "height": self.height,
            "crs": self.crs,
            "epsg": self.epsg,
            "transform": [
                float(self.transform.a),
                float(self.transform.b),
                float(self.transform.c),
                float(self.transform.d),
                float(self.transform.e),
                float(self.transform.f),
            ],
        }


@dataclass(frozen=True)
class MaskSettings:
    """Validated cloud-mask pipeline settings."""

    aligned_directory: Path
    output_directory: Path
    report_path: Path
    alignment_report_path: Path
    manifest_path: Path
    hard_invalid_classes: tuple[int, ...]
    cloud_classes: tuple[int, ...]
    cloud_buffer_pixels: int
    minimum_pair_valid_fraction: float
    target_resolution_meters: float
    target_epsg: int
    dataset_version: str
    aoi_name: str


def require_mapping(
    value: object,
    context: str,
) -> Mapping[str, Any]:
    """Validate that a value is a mapping."""
    if not isinstance(value, Mapping):
        raise ConfigurationError(
            f"Configuration value '{context}' must be a mapping."
        )

    return value


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load and validate a GeoWatch YAML configuration."""
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
    """Load a JSON file whose root must be an object."""
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
        raise CloudMaskError(
            f"Invalid JSON document: {json_path}"
        ) from error

    if not isinstance(payload, Mapping):
        raise CloudMaskError(
            f"JSON root must be an object: {json_path}"
        )

    return payload


def calculate_sha256(file_path: Path) -> str:
    """Calculate a file's SHA-256 checksum."""
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
    """Write JSON atomically to avoid partial reports."""
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


def normalize_scl_classes(
    values: Sequence[Any],
    context: str,
) -> tuple[int, ...]:
    """Validate a configured collection of Sentinel-2 SCL classes."""
    normalized: list[int] = []

    for value in values:
        try:
            class_id = int(value)
        except (TypeError, ValueError) as error:
            raise ConfigurationError(
                f"{context} must contain integer SCL class IDs."
            ) from error

        if not 0 <= class_id <= 11:
            raise ConfigurationError(
                f"{context} contains invalid SCL class {class_id}. "
                "Sentinel-2 SCL classes must be between 0 and 11."
            )

        normalized.append(class_id)

    if not normalized:
        raise ConfigurationError(
            f"{context} cannot be empty."
        )

    return tuple(sorted(set(normalized)))


def load_settings(
    config_path: Path,
    aligned_directory_override: Path | None,
    output_directory_override: Path | None,
    report_override: Path | None,
    hard_invalid_override: Sequence[int] | None,
    cloud_classes_override: Sequence[int] | None,
    buffer_override: int | None,
    minimum_valid_override: float | None,
) -> MaskSettings:
    """Load and validate cloud-mask settings."""
    config = load_yaml_config(config_path)

    project = require_mapping(
        config.get("project"),
        "project",
    )
    paths = require_mapping(
        config.get("paths"),
        "paths",
    )
    acquisition = require_mapping(
        config.get("acquisition"),
        "acquisition",
    )
    processing = require_mapping(
        config.get("processing"),
        "processing",
    )

    aligned_directory = (
        aligned_directory_override
        if aligned_directory_override is not None
        else Path(
            str(
                processing.get(
                    "aligned_dir",
                    "data/processed/aligned",
                )
            )
        )
    )

    output_directory = (
        output_directory_override
        if output_directory_override is not None
        else Path(
            str(
                processing.get(
                    "cloud_mask_dir",
                    "data/processed/cloud_masks",
                )
            )
        )
    )

    report_path = (
        report_override
        if report_override is not None
        else Path(
            str(
                processing.get(
                    "cloud_mask_report",
                    (
                        "data/processed/cloud_masks/"
                        "cloud_mask_report.json"
                    ),
                )
            )
        )
    )

    alignment_report_path = Path(
        str(
            processing.get(
                "alignment_report",
                (
                    "data/processed/aligned/"
                    "alignment_report.json"
                ),
            )
        )
    )

    manifest_path = Path(
        str(
            paths.get(
                "manifest_path",
                "data/manifest.csv",
            )
        )
    )

    configured_hard_invalid = processing.get(
        "scl_hard_invalid_classes",
        [0, 1, 7],
    )
    configured_cloud_classes = processing.get(
        "scl_cloud_classes",
        [3, 8, 9, 10, 11],
    )

    hard_invalid_values = (
        hard_invalid_override
        if hard_invalid_override is not None
        else configured_hard_invalid
    )
    cloud_class_values = (
        cloud_classes_override
        if cloud_classes_override is not None
        else configured_cloud_classes
    )

    if (
        not isinstance(hard_invalid_values, Sequence)
        or isinstance(hard_invalid_values, (str, bytes))
    ):
        raise ConfigurationError(
            "scl_hard_invalid_classes must be a list."
        )

    if (
        not isinstance(cloud_class_values, Sequence)
        or isinstance(cloud_class_values, (str, bytes))
    ):
        raise ConfigurationError(
            "scl_cloud_classes must be a list."
        )

    hard_invalid_classes = normalize_scl_classes(
        hard_invalid_values,
        "scl_hard_invalid_classes",
    )
    cloud_classes = normalize_scl_classes(
        cloud_class_values,
        "scl_cloud_classes",
    )

    overlap = set(hard_invalid_classes).intersection(
        cloud_classes
    )

    if overlap:
        raise ConfigurationError(
            "Hard-invalid and cloud class lists must not overlap. "
            f"Duplicated classes: {sorted(overlap)}"
        )

    cloud_buffer_pixels = (
        buffer_override
        if buffer_override is not None
        else int(
            processing.get(
                "cloud_buffer_pixels",
                3,
            )
        )
    )

    if cloud_buffer_pixels < 0:
        raise ConfigurationError(
            "cloud_buffer_pixels cannot be negative."
        )

    minimum_pair_valid_fraction = (
        minimum_valid_override
        if minimum_valid_override is not None
        else float(
            processing.get(
                "minimum_pair_valid_fraction",
                processing.get(
                    "minimum_valid_pixel_fraction",
                    0.90,
                ),
            )
        )
    )

    if not 0.0 < minimum_pair_valid_fraction <= 1.0:
        raise ConfigurationError(
            "minimum_pair_valid_fraction must be greater than "
            "zero and no greater than one."
        )

    target_resolution = float(
        processing.get(
            "target_resolution_meters",
            10.0,
        )
    )
    target_epsg = int(
        processing.get("target_epsg")
    )

    dataset_version = str(
        project.get("dataset_version", "")
    ).strip()
    aoi_name = str(
        acquisition.get("aoi_name", "")
    ).strip()

    if not dataset_version:
        raise ConfigurationError(
            "project.dataset_version cannot be empty."
        )

    if not aoi_name:
        raise ConfigurationError(
            "acquisition.aoi_name cannot be empty."
        )

    return MaskSettings(
        aligned_directory=aligned_directory,
        output_directory=output_directory,
        report_path=report_path,
        alignment_report_path=alignment_report_path,
        manifest_path=manifest_path,
        hard_invalid_classes=hard_invalid_classes,
        cloud_classes=cloud_classes,
        cloud_buffer_pixels=cloud_buffer_pixels,
        minimum_pair_valid_fraction=minimum_pair_valid_fraction,
        target_resolution_meters=target_resolution,
        target_epsg=target_epsg,
        dataset_version=dataset_version,
        aoi_name=aoi_name,
    )


def get_raster_grid(dataset: rasterio.io.DatasetReader) -> RasterGrid:
    """Extract comparable pixel-grid metadata from a raster."""
    if dataset.crs is None:
        raise CloudMaskError(
            f"Raster has no CRS: {dataset.name}"
        )

    epsg = dataset.crs.to_epsg()

    if epsg is None:
        raise CloudMaskError(
            f"Raster CRS has no EPSG identifier: {dataset.name}"
        )

    return RasterGrid(
        width=dataset.width,
        height=dataset.height,
        crs=dataset.crs.to_string(),
        epsg=epsg,
        transform=dataset.transform,
    )


def read_scl_raster(
    raster_path: Path,
    expected_epsg: int,
) -> tuple[np.ndarray, dict[str, Any], RasterGrid]:
    """Read and validate an aligned Sentinel-2 SCL raster."""
    if not raster_path.is_file():
        raise FileNotFoundError(
            f"Aligned SCL raster does not exist: {raster_path}"
        )

    try:
        with rasterio.open(raster_path) as dataset:
            if dataset.count != 1:
                raise CloudMaskError(
                    f"SCL raster must contain one band: {raster_path}"
                )

            grid = get_raster_grid(dataset)

            if grid.epsg != expected_epsg:
                raise CloudMaskError(
                    f"{raster_path} uses EPSG:{grid.epsg}; "
                    f"expected EPSG:{expected_epsg}."
                )

            scl = dataset.read(1)

            if scl.ndim != 2:
                raise CloudMaskError(
                    f"SCL raster is not two-dimensional: {raster_path}"
                )

            unique_values = np.unique(scl)
            unexpected_values = [
                int(value)
                for value in unique_values
                if value < 0 or value > 11
            ]

            if unexpected_values:
                raise CloudMaskError(
                    f"{raster_path} contains invalid SCL values: "
                    f"{unexpected_values}"
                )

            profile = dataset.profile.copy()

    except RasterioIOError as error:
        raise CloudMaskError(
            f"Rasterio could not open SCL raster: {raster_path}"
        ) from error

    return scl, profile, grid


def dilate_binary_mask(
    mask: np.ndarray,
    radius_pixels: int,
) -> np.ndarray:
    """Dilate a boolean mask using a circular pixel neighbourhood.

    Args:
        mask: Two-dimensional boolean input mask.
        radius_pixels: Dilation radius in pixels.

    Returns:
        Dilated boolean mask.
    """
    if mask.ndim != 2:
        raise ValueError(
            "Binary mask dilation requires a two-dimensional array."
        )

    if radius_pixels < 0:
        raise ValueError(
            "Dilation radius cannot be negative."
        )

    if radius_pixels == 0:
        return mask.copy()

    height, width = mask.shape
    padded = np.pad(
        mask,
        pad_width=radius_pixels,
        mode="constant",
        constant_values=False,
    )
    dilated = np.zeros_like(
        mask,
        dtype=bool,
    )

    for y_offset in range(
        -radius_pixels,
        radius_pixels + 1,
    ):
        for x_offset in range(
            -radius_pixels,
            radius_pixels + 1,
        ):
            if (
                x_offset * x_offset
                + y_offset * y_offset
                > radius_pixels * radius_pixels
            ):
                continue

            row_start = radius_pixels + y_offset
            column_start = radius_pixels + x_offset

            dilated |= padded[
                row_start : row_start + height,
                column_start : column_start + width,
            ]

    return dilated


def create_valid_mask(
    scl: np.ndarray,
    hard_invalid_classes: Sequence[int],
    cloud_classes: Sequence[int],
    cloud_buffer_pixels: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Create a valid-pixel mask from Sentinel-2 SCL classes."""
    hard_invalid = np.isin(
        scl,
        np.asarray(
            hard_invalid_classes,
            dtype=scl.dtype,
        ),
    )
    atmospheric_invalid = np.isin(
        scl,
        np.asarray(
            cloud_classes,
            dtype=scl.dtype,
        ),
    )

    buffered_atmospheric_invalid = dilate_binary_mask(
        atmospheric_invalid,
        radius_pixels=cloud_buffer_pixels,
    )

    final_invalid = (
        hard_invalid
        | buffered_atmospheric_invalid
    )
    valid_mask = (~final_invalid).astype(
        np.uint8
    )

    total_pixels = int(scl.size)

    statistics = {
        "total_pixels": total_pixels,
        "hard_invalid_pixels": int(
            np.count_nonzero(hard_invalid)
        ),
        "raw_atmospheric_invalid_pixels": int(
            np.count_nonzero(atmospheric_invalid)
        ),
        "buffered_atmospheric_invalid_pixels": int(
            np.count_nonzero(
                buffered_atmospheric_invalid
            )
        ),
        "final_invalid_pixels": int(
            np.count_nonzero(final_invalid)
        ),
        "valid_pixels": int(
            np.count_nonzero(valid_mask)
        ),
    }

    statistics["hard_invalid_fraction"] = (
        statistics["hard_invalid_pixels"] / total_pixels
    )
    statistics["raw_atmospheric_invalid_fraction"] = (
        statistics["raw_atmospheric_invalid_pixels"]
        / total_pixels
    )
    statistics["buffered_atmospheric_invalid_fraction"] = (
        statistics[
            "buffered_atmospheric_invalid_pixels"
        ]
        / total_pixels
    )
    statistics["final_invalid_fraction"] = (
        statistics["final_invalid_pixels"] / total_pixels
    )
    statistics["valid_fraction"] = (
        statistics["valid_pixels"] / total_pixels
    )

    return valid_mask, statistics


def calculate_scl_histogram(
    scl: np.ndarray,
) -> dict[str, int]:
    """Count pixels belonging to every Sentinel-2 SCL class."""
    unique_values, counts = np.unique(
        scl,
        return_counts=True,
    )

    histogram = {
        str(class_id): 0
        for class_id in range(12)
    }

    for class_id, count in zip(
        unique_values,
        counts,
    ):
        histogram[str(int(class_id))] = int(count)

    return histogram


def validate_existing_mask(
    output_path: Path,
    expected_mask: np.ndarray,
    expected_grid: RasterGrid,
) -> None:
    """Validate an existing mask against the expected output."""
    try:
        with rasterio.open(output_path) as dataset:
            grid = get_raster_grid(dataset)

            if not grid.matches(expected_grid):
                raise CloudMaskError(
                    f"Existing mask has the wrong grid: {output_path}"
                )

            if dataset.count != 1 or dataset.dtypes[0] != "uint8":
                raise CloudMaskError(
                    f"Existing mask has an invalid raster type: "
                    f"{output_path}"
                )

            actual_mask = dataset.read(1)

    except RasterioIOError as error:
        raise CloudMaskError(
            f"Could not validate existing mask: {output_path}"
        ) from error

    if not np.array_equal(
        actual_mask,
        expected_mask,
    ):
        raise CloudMaskError(
            f"Existing mask content differs from the expected mask: "
            f"{output_path}. Run with --overwrite."
        )


def write_mask_raster(
    mask: np.ndarray,
    output_path: Path,
    source_profile: Mapping[str, Any],
    grid: RasterGrid,
    tags: Mapping[str, str],
    overwrite: bool,
) -> dict[str, Any]:
    """Write a uint8 valid-pixel mask as a compressed GeoTIFF."""
    if output_path.exists() and not overwrite:
        validate_existing_mask(
            output_path=output_path,
            expected_mask=mask,
            expected_grid=grid,
        )

        return {
            "output_path": str(output_path),
            "status": "already_present",
            "sha256": calculate_sha256(output_path),
            "size_bytes": output_path.stat().st_size,
        }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_name(
        f"{output_path.stem}.tmp{output_path.suffix}"
    )
    temporary_path.unlink(missing_ok=True)

    profile = dict(source_profile)
    profile.update(
        {
            "driver": "GTiff",
            "width": grid.width,
            "height": grid.height,
            "count": 1,
            "dtype": "uint8",
            "crs": grid.crs,
            "transform": grid.transform,
            "nodata": None,
            "compress": "deflate",
            "predictor": 1,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "BIGTIFF": "IF_SAFER",
        }
    )

    try:
        with rasterio.open(
            temporary_path,
            "w",
            **profile,
        ) as output:
            output.write(mask, 1)
            output.update_tags(**dict(tags))

    except RasterioIOError as error:
        temporary_path.unlink(missing_ok=True)
        raise CloudMaskError(
            f"Could not write mask raster: {output_path}"
        ) from error
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    temporary_path.replace(output_path)

    return {
        "output_path": str(output_path),
        "status": "created",
        "sha256": calculate_sha256(output_path),
        "size_bytes": output_path.stat().st_size,
    }


def update_manifest(
    report: Mapping[str, Any],
    settings: MaskSettings,
) -> None:
    """Add cloud-mask lineage records to data/manifest.csv."""
    fieldnames = [
        "dataset_version",
        "record_id",
        "record_type",
        "source",
        "product_id",
        "acquisition_date",
        "aoi_name",
        "crs",
        "resolution_meters",
        "relative_path",
        "sha256",
        "parent_record_ids",
        "processing_status",
        "created_at_utc",
    ]

    existing_records: dict[str, dict[str, str]] = {}

    if settings.manifest_path.exists():
        with settings.manifest_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as manifest_file:
            for row in csv.DictReader(manifest_file):
                record_id = row.get(
                    "record_id",
                    "",
                ).strip()

                if record_id:
                    existing_records[record_id] = {
                        field: row.get(field, "")
                        for field in fieldnames
                    }

    periods = require_mapping(
        report.get("periods"),
        "cloud_mask_report.periods",
    )

    created_at = datetime.now(
        timezone.utc
    ).isoformat()

    parent_ids: dict[str, str] = {}

    for period in ("before", "after"):
        period_report = require_mapping(
            periods.get(period),
            f"cloud_mask_report.periods.{period}",
        )

        item_id = str(
            period_report.get("item_id", "")
        )
        acquisition_datetime = str(
            period_report.get(
                "acquisition_datetime",
                "",
            )
        )
        output = require_mapping(
            period_report.get("output"),
            f"cloud_mask_report.{period}.output",
        )

        record_id = (
            f"quality-mask-{period}-{item_id}"
        )
        parent_id = (
            f"aligned-{period}-{item_id}-SCL_20m"
        )
        parent_ids[period] = parent_id

        existing_records[record_id] = {
            "dataset_version": settings.dataset_version,
            "record_id": record_id,
            "record_type": "valid_pixel_mask",
            "source": "derived_from_sentinel2_scl",
            "product_id": item_id,
            "acquisition_date": acquisition_datetime[:10],
            "aoi_name": settings.aoi_name,
            "crs": f"EPSG:{settings.target_epsg}",
            "resolution_meters": str(
                settings.target_resolution_meters
            ),
            "relative_path": Path(
                str(output.get("output_path", ""))
            ).as_posix(),
            "sha256": str(output.get("sha256", "")),
            "parent_record_ids": parent_id,
            "processing_status": "quality_mask_created",
            "created_at_utc": created_at,
        }

    pair_output = require_mapping(
        report.get("pair_output"),
        "cloud_mask_report.pair_output",
    )
    pair_record_id = "quality-mask-before-after-pair"

    existing_records[pair_record_id] = {
        "dataset_version": settings.dataset_version,
        "record_id": pair_record_id,
        "record_type": "paired_valid_pixel_mask",
        "source": "intersection_of_period_valid_masks",
        "product_id": "before_after_pair",
        "acquisition_date": "",
        "aoi_name": settings.aoi_name,
        "crs": f"EPSG:{settings.target_epsg}",
        "resolution_meters": str(
            settings.target_resolution_meters
        ),
        "relative_path": Path(
            str(pair_output.get("output_path", ""))
        ).as_posix(),
        "sha256": str(
            pair_output.get("sha256", "")
        ),
        "parent_record_ids": (
            f"{parent_ids['before']};"
            f"{parent_ids['after']}"
        ),
        "processing_status": "paired_quality_mask_created",
        "created_at_utc": created_at,
    }

    settings.manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = settings.manifest_path.with_suffix(
        f"{settings.manifest_path.suffix}.tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for record_id in sorted(existing_records):
            writer.writerow(
                existing_records[record_id]
            )

    temporary_path.replace(
        settings.manifest_path
    )


def run_cloud_masking(
    config_path: Path,
    settings: MaskSettings,
    overwrite: bool,
) -> dict[str, Any]:
    """Generate period-specific and paired quality masks."""
    alignment_report = load_json_mapping(
        settings.alignment_report_path
    )
    alignment_periods = require_mapping(
        alignment_report.get("periods"),
        "alignment_report.periods",
    )

    period_results: dict[str, Any] = {}
    valid_masks: dict[str, np.ndarray] = {}
    reference_profile: dict[str, Any] | None = None
    reference_grid: RasterGrid | None = None

    for period in ("before", "after"):
        period_metadata = require_mapping(
            alignment_periods.get(period),
            f"alignment_report.periods.{period}",
        )

        scl_path = (
            settings.aligned_directory
            / period
            / "SCL_20m.tif"
        )

        scl, profile, grid = read_scl_raster(
            raster_path=scl_path,
            expected_epsg=settings.target_epsg,
        )

        if reference_grid is None:
            reference_grid = grid
            reference_profile = profile
        elif not grid.matches(reference_grid):
            raise CloudMaskError(
                f"{period} SCL does not match the reference grid."
            )

        valid_mask, statistics = create_valid_mask(
            scl=scl,
            hard_invalid_classes=(
                settings.hard_invalid_classes
            ),
            cloud_classes=settings.cloud_classes,
            cloud_buffer_pixels=(
                settings.cloud_buffer_pixels
            ),
        )

        valid_masks[period] = valid_mask

        output_path = (
            settings.output_directory
            / f"{period}_valid_mask.tif"
        )

        output_result = write_mask_raster(
            mask=valid_mask,
            output_path=output_path,
            source_profile=profile,
            grid=grid,
            tags={
                "mask_semantics": (
                    "1=valid, 0=invalid"
                ),
                "source_scl": str(scl_path),
                "hard_invalid_classes": ",".join(
                    str(value)
                    for value
                    in settings.hard_invalid_classes
                ),
                "cloud_classes": ",".join(
                    str(value)
                    for value
                    in settings.cloud_classes
                ),
                "cloud_buffer_pixels": str(
                    settings.cloud_buffer_pixels
                ),
            },
            overwrite=overwrite,
        )

        period_results[period] = {
            "item_id": period_metadata.get("item_id"),
            "acquisition_datetime": (
                period_metadata.get(
                    "acquisition_datetime"
                )
            ),
            "source_scl": str(scl_path),
            "source_scl_sha256": calculate_sha256(
                scl_path
            ),
            "scl_histogram": calculate_scl_histogram(
                scl
            ),
            "statistics": statistics,
            "output": output_result,
        }

        print(
            f"  [{period}] valid fraction: "
            f"{statistics['valid_fraction']:.2%}"
        )

    if reference_grid is None or reference_profile is None:
        raise CloudMaskError(
            "No SCL rasters were processed."
        )

    pair_valid_mask = (
        valid_masks["before"].astype(bool)
        & valid_masks["after"].astype(bool)
    ).astype(np.uint8)

    pair_valid_pixels = int(
        np.count_nonzero(pair_valid_mask)
    )
    total_pixels = int(pair_valid_mask.size)
    pair_valid_fraction = (
        pair_valid_pixels / total_pixels
        if total_pixels > 0
        else 0.0
    )

    pair_output_path = (
        settings.output_directory
        / "pair_valid_mask.tif"
    )

    pair_output = write_mask_raster(
        mask=pair_valid_mask,
        output_path=pair_output_path,
        source_profile=reference_profile,
        grid=reference_grid,
        tags={
            "mask_semantics": "1=valid in both dates, 0=invalid",
            "operation": (
                "before_valid_mask AND after_valid_mask"
            ),
        },
        overwrite=overwrite,
    )

    quality_gate_passed = (
        pair_valid_fraction
        >= settings.minimum_pair_valid_fraction
    )

    report = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": (
            "success"
            if quality_gate_passed
            else "failed"
        ),
        "source_alignment_report": str(
            settings.alignment_report_path
        ),
        "source_alignment_report_sha256": (
            calculate_sha256(
                settings.alignment_report_path
            )
        ),
        "grid": reference_grid.to_dict(),
        "hard_invalid_classes": list(
            settings.hard_invalid_classes
        ),
        "cloud_classes": list(
            settings.cloud_classes
        ),
        "cloud_buffer_pixels": (
            settings.cloud_buffer_pixels
        ),
        "cloud_buffer_meters": (
            settings.cloud_buffer_pixels
            * settings.target_resolution_meters
        ),
        "minimum_pair_valid_fraction": (
            settings.minimum_pair_valid_fraction
        ),
        "pair_valid_pixels": pair_valid_pixels,
        "total_pixels": total_pixels,
        "pair_valid_fraction": pair_valid_fraction,
        "quality_gate_passed": quality_gate_passed,
        "periods": period_results,
        "pair_output": pair_output,
    }

    write_json_atomic(
        payload=report,
        output_path=settings.report_path,
    )

    update_manifest(
        report=report,
        settings=settings,
    )

    return report


def print_summary(
    report: Mapping[str, Any],
    settings: MaskSettings,
) -> None:
    """Print a concise cloud-mask completion summary."""
    periods = require_mapping(
        report.get("periods"),
        "cloud_mask_report.periods",
    )

    print("Sentinel-2 SCL cloud masking completed")
    print(f"  Status: {report.get('status')}")

    for period in ("before", "after"):
        period_report = require_mapping(
            periods.get(period),
            f"cloud_mask_report.periods.{period}",
        )
        statistics = require_mapping(
            period_report.get("statistics"),
            f"cloud_mask_report.{period}.statistics",
        )

        print(
            f"  {period.title()} valid fraction: "
            f"{float(statistics.get('valid_fraction', 0.0)):.2%}"
        )

    print(
        "  Pair valid fraction: "
        f"{float(report.get('pair_valid_fraction', 0.0)):.2%}"
    )
    print(
        "  Minimum required: "
        f"{settings.minimum_pair_valid_fraction:.2%}"
    )
    print(
        f"  Quality gate passed: "
        f"{report.get('quality_gate_passed')}"
    )
    print(f"  Report: {settings.report_path}")
    print(f"  Manifest: {settings.manifest_path}")


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the cloud-mask command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate Sentinel-2 SCL-based valid-pixel masks "
            "for GeoWatch."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data_config.yaml"),
        help=(
            "GeoWatch data configuration path. "
            "Default: configs/data_config.yaml"
        ),
    )

    parser.add_argument(
        "--aligned-dir",
        type=Path,
        default=None,
        help="Optional aligned-raster directory override.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional cloud-mask output directory override.",
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional cloud-mask report path override.",
    )

    parser.add_argument(
        "--hard-invalid-classes",
        type=int,
        nargs="+",
        default=None,
        help=(
            "SCL classes always considered invalid. "
            "Defaults to configuration values."
        ),
    )

    parser.add_argument(
        "--cloud-classes",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Atmospheric SCL classes to mask and buffer. "
            "Defaults to configuration values."
        ),
    )

    parser.add_argument(
        "--buffer-pixels",
        type=int,
        default=None,
        help=(
            "Cloud-mask dilation radius in output pixels. "
            "Defaults to configuration value."
        ),
    )

    parser.add_argument(
        "--minimum-pair-valid-fraction",
        type=float,
        default=None,
        help=(
            "Minimum fraction valid in both dates. "
            "Defaults to configuration value."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing cloud-mask outputs.",
    )

    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging level. Default: INFO",
    )

    return parser


def main() -> int:
    """Run Sentinel-2 SCL cloud masking."""
    parser = build_argument_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    try:
        settings = load_settings(
            config_path=args.config,
            aligned_directory_override=args.aligned_dir,
            output_directory_override=args.output_dir,
            report_override=args.report,
            hard_invalid_override=args.hard_invalid_classes,
            cloud_classes_override=args.cloud_classes,
            buffer_override=args.buffer_pixels,
            minimum_valid_override=(
                args.minimum_pair_valid_fraction
            ),
        )

        report = run_cloud_masking(
            config_path=args.config,
            settings=settings,
            overwrite=args.overwrite,
        )

        print_summary(
            report=report,
            settings=settings,
        )

        if report["status"] != "success":
            LOGGER.error(
                "Pair-valid fraction is below the configured "
                "quality threshold."
            )
            return 1

        return 0

    except (
        FileNotFoundError,
        PermissionError,
        ConfigurationError,
        CloudMaskError,
        RasterioIOError,
        ValueError,
        yaml.YAMLError,
    ) as error:
        LOGGER.error("%s", error)
        return 1

    except Exception:
        LOGGER.exception(
            "Unexpected cloud-mask processing failure."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
