"""Production-grade paired patch generation for GeoWatch.

This module converts aligned, cloud-screened Sentinel-2 rasters into
georeferenced before/after patch pairs suitable for Siamese change-detection
models.

Each accepted sample contains:

* one six-band before GeoTIFF;
* one six-band after GeoTIFF;
* one uint8 pair-valid mask;
* one patch-index record with geospatial coordinates and lineage metadata.

Only complete windows are generated. Incomplete image-border regions are
reported but not padded, preventing artificial zero-padding boundaries from
entering the training dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import rasterio
import yaml
from affine import Affine
from rasterio.errors import RasterioIOError
from rasterio.windows import Window, bounds as window_bounds
from rasterio.windows import transform as window_transform


LOGGER = logging.getLogger("geowatch.tiling")


class ConfigurationError(ValueError):
    """Raised when patch-generation configuration is invalid."""


class TilingError(RuntimeError):
    """Raised when paired patch generation cannot be completed safely."""


@dataclass(frozen=True)
class RasterGrid:
    """Comparable raster-grid metadata."""

    width: int
    height: int
    epsg: int
    crs: str
    transform: Affine

    @classmethod
    def from_dataset(
        cls,
        dataset: rasterio.io.DatasetReader,
    ) -> "RasterGrid":
        """Create grid metadata from an open Rasterio dataset."""
        if dataset.crs is None:
            raise TilingError(
                f"Raster has no CRS: {dataset.name}"
            )

        epsg = dataset.crs.to_epsg()

        if epsg is None:
            raise TilingError(
                f"Raster CRS has no EPSG identifier: {dataset.name}"
            )

        return cls(
            width=dataset.width,
            height=dataset.height,
            epsg=epsg,
            crs=dataset.crs.to_string(),
            transform=dataset.transform,
        )

    def matches(self, other: "RasterGrid") -> bool:
        """Return whether two rasters use the exact same pixel grid."""
        return (
            self.width == other.width
            and self.height == other.height
            and self.epsg == other.epsg
            and self.transform == other.transform
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert grid metadata to JSON-safe values."""
        return {
            "width": self.width,
            "height": self.height,
            "epsg": self.epsg,
            "crs": self.crs,
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
class TilingSettings:
    """Validated GeoWatch patch-generation settings."""

    aligned_directory: Path
    pair_mask_path: Path
    alignment_report_path: Path
    cloud_mask_report_path: Path
    output_directory: Path
    patch_index_path: Path
    report_path: Path
    manifest_path: Path
    input_assets: tuple[str, ...]
    patch_size: int
    stride: int
    minimum_valid_fraction: float
    target_epsg: int
    resolution_meters: float
    dataset_version: str
    aoi_name: str


@dataclass(frozen=True)
class PatchWindow:
    """One deterministic spatial patch window."""

    patch_id: str
    row_index: int
    column_index: int
    row_offset: int
    column_offset: int
    size: int

    def to_rasterio_window(self) -> Window:
        """Convert to a Rasterio window."""
        return Window(
            col_off=self.column_offset,
            row_off=self.row_offset,
            width=self.size,
            height=self.size,
        )


def require_mapping(
    value: object,
    context: str,
) -> Mapping[str, Any]:
    """Validate that a value is a mapping."""
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
        raise TilingError(
            f"Invalid JSON document: {json_path}"
        ) from error

    if not isinstance(payload, Mapping):
        raise TilingError(
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


def load_settings(
    config_path: Path,
    output_directory_override: Path | None,
    patch_index_override: Path | None,
    report_override: Path | None,
    patch_size_override: int | None,
    stride_override: int | None,
    minimum_valid_override: float | None,
) -> TilingSettings:
    """Load and validate patch-generation configuration."""
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

    aligned_directory = Path(
        str(
            processing.get(
                "aligned_dir",
                "data/processed/aligned",
            )
        )
    )

    cloud_mask_directory = Path(
        str(
            processing.get(
                "cloud_mask_dir",
                "data/processed/cloud_masks",
            )
        )
    )

    pair_mask_path = (
        cloud_mask_directory
        / "pair_valid_mask.tif"
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

    cloud_mask_report_path = Path(
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

    output_directory = (
        output_directory_override
        if output_directory_override is not None
        else Path(
            str(
                processing.get(
                    "patches_dir",
                    "data/processed/patches",
                )
            )
        )
    )

    patch_index_path = (
        patch_index_override
        if patch_index_override is not None
        else Path(
            str(
                processing.get(
                    "patch_index",
                    (
                        "data/processed/patches/"
                        "patch_index.csv"
                    ),
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
                    "tiling_report",
                    (
                        "data/processed/patches/"
                        "tiling_report.json"
                    ),
                )
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

    configured_assets = processing.get(
        "model_input_assets"
    )

    if (
        not isinstance(configured_assets, Sequence)
        or isinstance(configured_assets, (str, bytes))
        or not configured_assets
    ):
        raise ConfigurationError(
            "processing.model_input_assets must be a non-empty list."
        )

    input_assets = tuple(
        str(asset_key)
        for asset_key in configured_assets
    )

    if len(input_assets) != len(set(input_assets)):
        raise ConfigurationError(
            "processing.model_input_assets contains duplicates."
        )

    if any(asset.startswith("SCL") for asset in input_assets):
        raise ConfigurationError(
            "SCL must remain a quality-control layer and cannot "
            "be included in model_input_assets."
        )

    patch_size = (
        patch_size_override
        if patch_size_override is not None
        else int(
            processing.get(
                "patch_size_pixels",
                256,
            )
        )
    )

    stride = (
        stride_override
        if stride_override is not None
        else int(
            processing.get(
                "patch_stride_pixels",
                patch_size,
            )
        )
    )

    minimum_valid_fraction = (
        minimum_valid_override
        if minimum_valid_override is not None
        else float(
            processing.get(
                "minimum_patch_valid_fraction",
                processing.get(
                    "minimum_valid_pixel_fraction",
                    0.90,
                ),
            )
        )
    )

    if patch_size <= 0:
        raise ConfigurationError(
            "Patch size must be greater than zero."
        )

    if patch_size % 16 != 0:
        raise ConfigurationError(
            "Patch size must be divisible by 16 for tiled GeoTIFF output."
        )

    if stride <= 0:
        raise ConfigurationError(
            "Patch stride must be greater than zero."
        )

    if stride > patch_size:
        raise ConfigurationError(
            "Patch stride cannot exceed patch size because this "
            "would leave unrepresented gaps."
        )

    if not 0.0 < minimum_valid_fraction <= 1.0:
        raise ConfigurationError(
            "Minimum valid fraction must be greater than zero "
            "and no greater than one."
        )

    target_epsg = int(
        processing.get("target_epsg")
    )
    resolution_meters = float(
        processing.get(
            "target_resolution_meters",
            10.0,
        )
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

    return TilingSettings(
        aligned_directory=aligned_directory,
        pair_mask_path=pair_mask_path,
        alignment_report_path=alignment_report_path,
        cloud_mask_report_path=cloud_mask_report_path,
        output_directory=output_directory,
        patch_index_path=patch_index_path,
        report_path=report_path,
        manifest_path=manifest_path,
        input_assets=input_assets,
        patch_size=patch_size,
        stride=stride,
        minimum_valid_fraction=minimum_valid_fraction,
        target_epsg=target_epsg,
        resolution_meters=resolution_meters,
        dataset_version=dataset_version,
        aoi_name=aoi_name,
    )


def generate_patch_windows(
    raster_width: int,
    raster_height: int,
    patch_size: int,
    stride: int,
) -> list[PatchWindow]:
    """Generate deterministic complete patch windows.

    Incomplete right and bottom border windows are excluded instead of
    padded. Excluded border dimensions are recorded in the tiling report.
    """
    if raster_width < patch_size or raster_height < patch_size:
        raise TilingError(
            "Raster dimensions are smaller than the requested patch size."
        )

    column_offsets = list(
        range(
            0,
            raster_width - patch_size + 1,
            stride,
        )
    )
    row_offsets = list(
        range(
            0,
            raster_height - patch_size + 1,
            stride,
        )
    )

    windows: list[PatchWindow] = []

    for row_index, row_offset in enumerate(row_offsets):
        for column_index, column_offset in enumerate(
            column_offsets
        ):
            windows.append(
                PatchWindow(
                    patch_id=(
                        f"GW_R{row_index:03d}_"
                        f"C{column_index:03d}"
                    ),
                    row_index=row_index,
                    column_index=column_index,
                    row_offset=row_offset,
                    column_offset=column_offset,
                    size=patch_size,
                )
            )

    return windows


def validate_dataset_grid(
    dataset: rasterio.io.DatasetReader,
    expected_grid: RasterGrid,
    expected_epsg: int,
) -> None:
    """Validate one aligned input raster against the reference grid."""
    actual_grid = RasterGrid.from_dataset(dataset)

    if actual_grid.epsg != expected_epsg:
        raise TilingError(
            f"{dataset.name} uses EPSG:{actual_grid.epsg}; "
            f"expected EPSG:{expected_epsg}."
        )

    if not actual_grid.matches(expected_grid):
        raise TilingError(
            f"Input raster does not match the reference grid: "
            f"{dataset.name}"
        )

    if dataset.count != 1:
        raise TilingError(
            f"Input raster must contain one band: {dataset.name}"
        )


def validate_existing_stack(
    output_path: Path,
    expected_grid: RasterGrid,
    expected_transform: Affine,
    expected_size: int,
    expected_band_names: Sequence[str],
) -> None:
    """Validate an existing multi-band patch."""
    try:
        with rasterio.open(output_path) as dataset:
            grid = RasterGrid.from_dataset(dataset)

            if grid.epsg != expected_grid.epsg:
                raise TilingError(
                    f"Existing patch has incorrect CRS: {output_path}"
                )

            if (
                dataset.width != expected_size
                or dataset.height != expected_size
                or dataset.transform != expected_transform
                or dataset.count != len(expected_band_names)
            ):
                raise TilingError(
                    f"Existing patch has an incorrect grid or band count: "
                    f"{output_path}. Run with --overwrite."
                )

            descriptions = tuple(
                description or ""
                for description in dataset.descriptions
            )

            if descriptions != tuple(expected_band_names):
                raise TilingError(
                    f"Existing patch has incorrect band descriptions: "
                    f"{output_path}. Run with --overwrite."
                )

    except RasterioIOError as error:
        raise TilingError(
            f"Could not validate existing patch: {output_path}"
        ) from error


def write_multiband_patch(
    arrays: Sequence[np.ndarray],
    band_names: Sequence[str],
    output_path: Path,
    source_profile: Mapping[str, Any],
    patch_transform: Affine,
    reference_grid: RasterGrid,
    patch_id: str,
    period: str,
    overwrite: bool,
) -> dict[str, Any]:
    """Write one compressed georeferenced multi-band patch."""
    if len(arrays) != len(band_names):
        raise TilingError(
            "Array count and band-name count do not match."
        )

    if not arrays:
        raise TilingError(
            "Cannot create a patch with no image bands."
        )

    first_shape = arrays[0].shape
    first_dtype = arrays[0].dtype

    for array in arrays:
        if array.shape != first_shape:
            raise TilingError(
                f"Input arrays for {patch_id}/{period} "
                "do not share one shape."
            )

        if array.dtype != first_dtype:
            raise TilingError(
                f"Input arrays for {patch_id}/{period} "
                "do not share one data type."
            )

    patch_size = first_shape[0]

    if first_shape != (patch_size, patch_size):
        raise TilingError(
            f"Patch {patch_id}/{period} is not square."
        )

    if output_path.exists() and not overwrite:
        validate_existing_stack(
            output_path=output_path,
            expected_grid=reference_grid,
            expected_transform=patch_transform,
            expected_size=patch_size,
            expected_band_names=band_names,
        )

        return {
            "output_path": str(output_path),
            "status": "already_present",
            "size_bytes": output_path.stat().st_size,
            "sha256": calculate_sha256(output_path),
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
            "width": patch_size,
            "height": patch_size,
            "count": len(arrays),
            "dtype": str(first_dtype),
            "crs": reference_grid.crs,
            "transform": patch_transform,
            "nodata": 0,
            "compress": "deflate",
            "predictor": 2,
            "tiled": True,
            "blockxsize": patch_size,
            "blockysize": patch_size,
            "BIGTIFF": "IF_SAFER",
        }
    )

    try:
        with rasterio.open(
            temporary_path,
            "w",
            **profile,
        ) as output:
            for band_index, (
                array,
                band_name,
            ) in enumerate(
                zip(arrays, band_names),
                start=1,
            ):
                output.write(array, band_index)
                output.set_band_description(
                    band_index,
                    band_name,
                )

            output.update_tags(
                patch_id=patch_id,
                period=period,
                channel_order=",".join(band_names),
                mask_applied="false",
                mask_usage=(
                    "Use the corresponding pair-valid mask "
                    "during training and evaluation."
                ),
            )

    except RasterioIOError as error:
        temporary_path.unlink(missing_ok=True)
        raise TilingError(
            f"Could not write patch: {output_path}"
        ) from error
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    temporary_path.replace(output_path)

    return {
        "output_path": str(output_path),
        "status": "created",
        "size_bytes": output_path.stat().st_size,
        "sha256": calculate_sha256(output_path),
    }


def validate_existing_mask(
    output_path: Path,
    expected_mask: np.ndarray,
    expected_transform: Affine,
    expected_grid: RasterGrid,
) -> None:
    """Validate an existing patch-validity mask."""
    try:
        with rasterio.open(output_path) as dataset:
            grid = RasterGrid.from_dataset(dataset)

            if (
                grid.epsg != expected_grid.epsg
                or dataset.width != expected_mask.shape[1]
                or dataset.height != expected_mask.shape[0]
                or dataset.transform != expected_transform
                or dataset.count != 1
                or dataset.dtypes[0] != "uint8"
            ):
                raise TilingError(
                    f"Existing mask has invalid metadata: {output_path}"
                )

            actual_mask = dataset.read(1)

    except RasterioIOError as error:
        raise TilingError(
            f"Could not validate existing mask: {output_path}"
        ) from error

    if not np.array_equal(actual_mask, expected_mask):
        raise TilingError(
            f"Existing mask content differs: {output_path}. "
            "Run with --overwrite."
        )


def write_mask_patch(
    mask: np.ndarray,
    output_path: Path,
    source_profile: Mapping[str, Any],
    patch_transform: Affine,
    reference_grid: RasterGrid,
    patch_id: str,
    overwrite: bool,
) -> dict[str, Any]:
    """Write one uint8 paired-validity mask patch."""
    if output_path.exists() and not overwrite:
        validate_existing_mask(
            output_path=output_path,
            expected_mask=mask,
            expected_transform=patch_transform,
            expected_grid=reference_grid,
        )

        return {
            "output_path": str(output_path),
            "status": "already_present",
            "size_bytes": output_path.stat().st_size,
            "sha256": calculate_sha256(output_path),
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
            "width": mask.shape[1],
            "height": mask.shape[0],
            "count": 1,
            "dtype": "uint8",
            "crs": reference_grid.crs,
            "transform": patch_transform,
            "nodata": None,
            "compress": "deflate",
            "predictor": 1,
            "tiled": True,
            "blockxsize": mask.shape[1],
            "blockysize": mask.shape[0],
            "BIGTIFF": "IF_SAFER",
        }
    )

    try:
        with rasterio.open(
            temporary_path,
            "w",
            **profile,
        ) as output:
            output.write(mask.astype(np.uint8), 1)
            output.update_tags(
                patch_id=patch_id,
                mask_semantics=(
                    "1=valid in both dates, 0=invalid"
                ),
            )

    except RasterioIOError as error:
        temporary_path.unlink(missing_ok=True)
        raise TilingError(
            f"Could not write patch mask: {output_path}"
        ) from error
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    temporary_path.replace(output_path)

    return {
        "output_path": str(output_path),
        "status": "created",
        "size_bytes": output_path.stat().st_size,
        "sha256": calculate_sha256(output_path),
    }


def write_patch_index_atomic(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    """Write the patch index atomically."""
    fieldnames = [
        "patch_id",
        "row_index",
        "column_index",
        "row_offset",
        "column_offset",
        "patch_size_pixels",
        "left",
        "bottom",
        "right",
        "top",
        "center_x",
        "center_y",
        "crs",
        "valid_fraction",
        "invalid_fraction",
        "accepted",
        "rejection_reason",
        "before_path",
        "after_path",
        "mask_path",
        "before_sha256",
        "after_sha256",
        "mask_sha256",
        "split",
    ]

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
    ) as index_file:
        writer = csv.DictWriter(
            index_file,
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


def update_manifest(
    accepted_rows: Sequence[Mapping[str, Any]],
    settings: TilingSettings,
    alignment_report: Mapping[str, Any],
) -> None:
    """Add accepted patch artifacts to data/manifest.csv."""
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

    records: dict[str, dict[str, str]] = {}

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
                    records[record_id] = {
                        field: row.get(field, "")
                        for field in fieldnames
                    }

    periods = require_mapping(
        alignment_report.get("periods"),
        "alignment_report.periods",
    )

    period_metadata: dict[str, Mapping[str, Any]] = {
        period: require_mapping(
            periods.get(period),
            f"alignment_report.periods.{period}",
        )
        for period in ("before", "after")
    }

    created_at = datetime.now(
        timezone.utc
    ).isoformat()

    for row in accepted_rows:
        patch_id = str(row["patch_id"])

        for period in ("before", "after"):
            metadata = period_metadata[period]
            item_id = str(metadata.get("item_id", ""))
            acquisition_datetime = str(
                metadata.get(
                    "acquisition_datetime",
                    "",
                )
            )

            path_key = f"{period}_path"
            checksum_key = f"{period}_sha256"

            parent_ids = ";".join(
                (
                    f"aligned-{period}-{item_id}-"
                    f"{asset_key}"
                )
                for asset_key in settings.input_assets
            )

            record_id = (
                f"patch-{patch_id}-{period}"
            )

            records[record_id] = {
                "dataset_version": settings.dataset_version,
                "record_id": record_id,
                "record_type": "multiband_satellite_patch",
                "source": "derived_from_aligned_satellite_bands",
                "product_id": item_id,
                "acquisition_date": acquisition_datetime[:10],
                "aoi_name": settings.aoi_name,
                "crs": f"EPSG:{settings.target_epsg}",
                "resolution_meters": str(
                    settings.resolution_meters
                ),
                "relative_path": Path(
                    str(row[path_key])
                ).as_posix(),
                "sha256": str(row[checksum_key]),
                "parent_record_ids": parent_ids,
                "processing_status": "patch_generated",
                "created_at_utc": created_at,
            }

        mask_record_id = (
            f"patch-{patch_id}-pair-mask"
        )

        records[mask_record_id] = {
            "dataset_version": settings.dataset_version,
            "record_id": mask_record_id,
            "record_type": "patch_valid_mask",
            "source": "derived_from_paired_valid_pixel_mask",
            "product_id": "before_after_pair",
            "acquisition_date": "",
            "aoi_name": settings.aoi_name,
            "crs": f"EPSG:{settings.target_epsg}",
            "resolution_meters": str(
                settings.resolution_meters
            ),
            "relative_path": Path(
                str(row["mask_path"])
            ).as_posix(),
            "sha256": str(row["mask_sha256"]),
            "parent_record_ids": (
                "quality-mask-before-after-pair"
            ),
            "processing_status": "patch_mask_generated",
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

        for record_id in sorted(records):
            writer.writerow(records[record_id])

    temporary_path.replace(
        settings.manifest_path
    )


def run_tiling(
    settings: TilingSettings,
    overwrite: bool,
) -> dict[str, Any]:
    """Generate accepted paired image and validity-mask patches."""
    alignment_report = load_json_mapping(
        settings.alignment_report_path
    )
    cloud_mask_report = load_json_mapping(
        settings.cloud_mask_report_path
    )

    if alignment_report.get("status") != "success":
        raise TilingError(
            "Alignment report does not have success status."
        )

    if cloud_mask_report.get("status") != "success":
        raise TilingError(
            "Cloud-mask report does not have success status."
        )

    settings.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    before_paths = {
        asset_key: (
            settings.aligned_directory
            / "before"
            / f"{asset_key}.tif"
        )
        for asset_key in settings.input_assets
    }
    after_paths = {
        asset_key: (
            settings.aligned_directory
            / "after"
            / f"{asset_key}.tif"
        )
        for asset_key in settings.input_assets
    }

    all_required_paths = [
        *before_paths.values(),
        *after_paths.values(),
        settings.pair_mask_path,
    ]

    missing_paths = [
        str(path)
        for path in all_required_paths
        if not path.is_file()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Required aligned inputs are missing: "
            f"{missing_paths}"
        )

    index_rows: list[dict[str, Any]] = []
    accepted_rows: list[dict[str, Any]] = []

    with ExitStack() as stack:
        before_datasets = {
            asset_key: stack.enter_context(
                rasterio.open(path)
            )
            for asset_key, path in before_paths.items()
        }
        after_datasets = {
            asset_key: stack.enter_context(
                rasterio.open(path)
            )
            for asset_key, path in after_paths.items()
        }
        mask_dataset = stack.enter_context(
            rasterio.open(settings.pair_mask_path)
        )

        reference_dataset = next(
            iter(before_datasets.values())
        )
        reference_grid = RasterGrid.from_dataset(
            reference_dataset
        )

        if reference_grid.epsg != settings.target_epsg:
            raise TilingError(
                f"Reference raster uses EPSG:{reference_grid.epsg}; "
                f"expected EPSG:{settings.target_epsg}."
            )

        for dataset in (
            *before_datasets.values(),
            *after_datasets.values(),
            mask_dataset,
        ):
            validate_dataset_grid(
                dataset=dataset,
                expected_grid=reference_grid,
                expected_epsg=settings.target_epsg,
            )

        if mask_dataset.dtypes[0] != "uint8":
            raise TilingError(
                "Pair-valid mask must use uint8 values."
            )

        windows = generate_patch_windows(
            raster_width=reference_grid.width,
            raster_height=reference_grid.height,
            patch_size=settings.patch_size,
            stride=settings.stride,
        )

        before_profile = reference_dataset.profile.copy()
        mask_profile = mask_dataset.profile.copy()

        for patch_window in windows:
            window = patch_window.to_rasterio_window()
            mask = mask_dataset.read(
                1,
                window=window,
            )

            if mask.shape != (
                settings.patch_size,
                settings.patch_size,
            ):
                raise TilingError(
                    f"Unexpected mask shape for "
                    f"{patch_window.patch_id}: {mask.shape}"
                )

            unique_mask_values = set(
                int(value)
                for value in np.unique(mask)
            )

            if not unique_mask_values.issubset({0, 1}):
                raise TilingError(
                    f"Mask for {patch_window.patch_id} "
                    f"contains invalid values: "
                    f"{sorted(unique_mask_values)}"
                )

            valid_fraction = float(
                np.count_nonzero(mask == 1)
                / mask.size
            )
            invalid_fraction = 1.0 - valid_fraction
            accepted = (
                valid_fraction
                >= settings.minimum_valid_fraction
            )

            patch_transform = window_transform(
                window,
                reference_grid.transform,
            )
            left, bottom, right, top = window_bounds(
                window,
                reference_grid.transform,
            )

            row: dict[str, Any] = {
                "patch_id": patch_window.patch_id,
                "row_index": patch_window.row_index,
                "column_index": patch_window.column_index,
                "row_offset": patch_window.row_offset,
                "column_offset": patch_window.column_offset,
                "patch_size_pixels": settings.patch_size,
                "left": float(left),
                "bottom": float(bottom),
                "right": float(right),
                "top": float(top),
                "center_x": float(
                    (left + right) / 2.0
                ),
                "center_y": float(
                    (bottom + top) / 2.0
                ),
                "crs": reference_grid.crs,
                "valid_fraction": round(
                    valid_fraction,
                    8,
                ),
                "invalid_fraction": round(
                    invalid_fraction,
                    8,
                ),
                "accepted": accepted,
                "rejection_reason": (
                    ""
                    if accepted
                    else "valid_fraction_below_threshold"
                ),
                "before_path": "",
                "after_path": "",
                "mask_path": "",
                "before_sha256": "",
                "after_sha256": "",
                "mask_sha256": "",
                "split": "",
            }

            if not accepted:
                index_rows.append(row)
                print(
                    f"  [rejected] {patch_window.patch_id}: "
                    f"valid={valid_fraction:.2%}"
                )
                continue

            before_arrays = [
                before_datasets[asset_key].read(
                    1,
                    window=window,
                )
                for asset_key in settings.input_assets
            ]
            after_arrays = [
                after_datasets[asset_key].read(
                    1,
                    window=window,
                )
                for asset_key in settings.input_assets
            ]

            before_output = (
                settings.output_directory
                / "before"
                / f"{patch_window.patch_id}.tif"
            )
            after_output = (
                settings.output_directory
                / "after"
                / f"{patch_window.patch_id}.tif"
            )
            mask_output = (
                settings.output_directory
                / "masks"
                / f"{patch_window.patch_id}.tif"
            )

            before_result = write_multiband_patch(
                arrays=before_arrays,
                band_names=settings.input_assets,
                output_path=before_output,
                source_profile=before_profile,
                patch_transform=patch_transform,
                reference_grid=reference_grid,
                patch_id=patch_window.patch_id,
                period="before",
                overwrite=overwrite,
            )
            after_result = write_multiband_patch(
                arrays=after_arrays,
                band_names=settings.input_assets,
                output_path=after_output,
                source_profile=before_profile,
                patch_transform=patch_transform,
                reference_grid=reference_grid,
                patch_id=patch_window.patch_id,
                period="after",
                overwrite=overwrite,
            )
            mask_result = write_mask_patch(
                mask=mask,
                output_path=mask_output,
                source_profile=mask_profile,
                patch_transform=patch_transform,
                reference_grid=reference_grid,
                patch_id=patch_window.patch_id,
                overwrite=overwrite,
            )

            row.update(
                {
                    "before_path": before_result[
                        "output_path"
                    ],
                    "after_path": after_result[
                        "output_path"
                    ],
                    "mask_path": mask_result[
                        "output_path"
                    ],
                    "before_sha256": before_result[
                        "sha256"
                    ],
                    "after_sha256": after_result[
                        "sha256"
                    ],
                    "mask_sha256": mask_result[
                        "sha256"
                    ],
                }
            )

            index_rows.append(row)
            accepted_rows.append(row)

            print(
                f"  [accepted] {patch_window.patch_id}: "
                f"valid={valid_fraction:.2%}"
            )

    write_patch_index_atomic(
        rows=index_rows,
        output_path=settings.patch_index_path,
    )

    update_manifest(
        accepted_rows=accepted_rows,
        settings=settings,
        alignment_report=alignment_report,
    )

    covered_width = (
        max(
            int(row["column_offset"])
            for row in index_rows
        )
        + settings.patch_size
    )
    covered_height = (
        max(
            int(row["row_offset"])
            for row in index_rows
        )
        + settings.patch_size
    )

    dropped_right_pixels = (
        reference_grid.width - covered_width
    )
    dropped_bottom_pixels = (
        reference_grid.height - covered_height
    )

    accepted_valid_fractions = [
        float(row["valid_fraction"])
        for row in accepted_rows
    ]

    report = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": (
            "success"
            if accepted_rows
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
        "source_cloud_mask_report": str(
            settings.cloud_mask_report_path
        ),
        "source_cloud_mask_report_sha256": (
            calculate_sha256(
                settings.cloud_mask_report_path
            )
        ),
        "reference_grid": reference_grid.to_dict(),
        "input_assets": list(settings.input_assets),
        "input_channel_count": len(
            settings.input_assets
        ),
        "patch_size_pixels": settings.patch_size,
        "patch_stride_pixels": settings.stride,
        "edge_policy": "drop_incomplete",
        "minimum_patch_valid_fraction": (
            settings.minimum_valid_fraction
        ),
        "candidate_patch_count": len(index_rows),
        "accepted_patch_count": len(accepted_rows),
        "rejected_patch_count": (
            len(index_rows) - len(accepted_rows)
        ),
        "minimum_accepted_valid_fraction": (
            min(accepted_valid_fractions)
            if accepted_valid_fractions
            else None
        ),
        "mean_accepted_valid_fraction": (
            float(
                np.mean(
                    accepted_valid_fractions
                )
            )
            if accepted_valid_fractions
            else None
        ),
        "maximum_accepted_valid_fraction": (
            max(accepted_valid_fractions)
            if accepted_valid_fractions
            else None
        ),
        "covered_width_pixels": covered_width,
        "covered_height_pixels": covered_height,
        "dropped_right_border_pixels": (
            dropped_right_pixels
        ),
        "dropped_bottom_border_pixels": (
            dropped_bottom_pixels
        ),
        "covered_area_fraction": (
            (covered_width * covered_height)
            / (
                reference_grid.width
                * reference_grid.height
            )
        ),
        "patch_index": str(
            settings.patch_index_path
        ),
        "output_directory": str(
            settings.output_directory
        ),
        "manifest": str(
            settings.manifest_path
        ),
    }

    write_json_atomic(
        payload=report,
        output_path=settings.report_path,
    )

    return report


def print_summary(
    report: Mapping[str, Any],
    settings: TilingSettings,
) -> None:
    """Print patch-generation completion details."""
    print("GeoWatch paired patch generation completed")
    print(f"  Status: {report.get('status')}")
    print(
        "  Candidate patches: "
        f"{report.get('candidate_patch_count')}"
    )
    print(
        "  Accepted patches: "
        f"{report.get('accepted_patch_count')}"
    )
    print(
        "  Rejected patches: "
        f"{report.get('rejected_patch_count')}"
    )
    print(
        "  Input channels: "
        f"{report.get('input_channel_count')}"
    )
    print(
        "  Mean accepted valid fraction: "
        f"{float(report.get('mean_accepted_valid_fraction', 0.0)):.2%}"
    )
    print(
        "  Covered AOI-grid fraction: "
        f"{float(report.get('covered_area_fraction', 0.0)):.2%}"
    )
    print(
        "  Dropped border: "
        f"right={report.get('dropped_right_border_pixels')} px, "
        f"bottom={report.get('dropped_bottom_border_pixels')} px"
    )
    print(f"  Patch index: {settings.patch_index_path}")
    print(f"  Tiling report: {settings.report_path}")
    print(f"  Dataset manifest: {settings.manifest_path}")


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the patch-generation command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate georeferenced paired Sentinel-2 patches "
            "for GeoWatch."
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
        "--output-dir",
        type=Path,
        default=None,
        help="Optional patch-output directory override.",
    )

    parser.add_argument(
        "--patch-index",
        type=Path,
        default=None,
        help="Optional patch-index CSV path override.",
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional tiling-report JSON path override.",
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        default=None,
        help="Optional square patch size override.",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Optional patch stride override.",
    )

    parser.add_argument(
        "--minimum-valid-fraction",
        type=float,
        default=None,
        help=(
            "Minimum valid-pixel fraction required to retain a patch."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing patch outputs.",
    )

    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging level. Default: INFO",
    )

    return parser


def main() -> int:
    """Run paired patch generation."""
    parser = build_argument_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    try:
        settings = load_settings(
            config_path=args.config,
            output_directory_override=args.output_dir,
            patch_index_override=args.patch_index,
            report_override=args.report,
            patch_size_override=args.patch_size,
            stride_override=args.stride,
            minimum_valid_override=(
                args.minimum_valid_fraction
            ),
        )

        report = run_tiling(
            settings=settings,
            overwrite=args.overwrite,
        )

        print_summary(
            report=report,
            settings=settings,
        )

        if report["status"] != "success":
            LOGGER.error(
                "No patch passed the configured quality threshold."
            )
            return 1

        return 0

    except (
        FileNotFoundError,
        PermissionError,
        ConfigurationError,
        TilingError,
        RasterioIOError,
        ValueError,
        yaml.YAMLError,
    ) as error:
        LOGGER.error("%s", error)
        return 1

    except Exception:
        LOGGER.exception(
            "Unexpected patch-generation failure."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
