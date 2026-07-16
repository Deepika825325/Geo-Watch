"""Geospatial validation, projection, cropping, and alignment for GeoWatch.

The module supports two commands:

1. ``aoi`` validates a WGS84 area of interest and determines its UTM CRS.
2. ``align`` crops all selected Sentinel-2 bands to the AOI and places them
   on one common 10-metre reference grid for bi-temporal change detection.
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

import numpy as np
import rasterio
import yaml
from affine import Affine
from pyproj import CRS, Geod, Transformer
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.transform import array_bounds
from rasterio.warp import reproject, transform_bounds


LOGGER = logging.getLogger("geowatch.geo_utils")


class ConfigurationError(ValueError):
    """Raised when GeoWatch configuration values are invalid."""


class AlignmentError(RuntimeError):
    """Raised when raster alignment cannot be completed safely."""


@dataclass(frozen=True)
class BoundingBox:
    """WGS84 bounding box ordered as west, south, east, north."""

    west: float
    south: float
    east: float
    north: float

    @classmethod
    def from_sequence(cls, values: Sequence[float]) -> "BoundingBox":
        """Create and validate a bounding box from four values."""
        if len(values) != 4:
            raise ValueError(
                "Bounding box requires WEST SOUTH EAST NORTH."
            )

        bbox = cls(*(float(value) for value in values))
        bbox.validate()
        return bbox

    def validate(self) -> None:
        """Validate coordinate ranges and orientation."""
        if not -180.0 <= self.west < self.east <= 180.0:
            raise ValueError(
                "Longitude values must satisfy "
                "-180 <= west < east <= 180."
            )

        if not -90.0 <= self.south < self.north <= 90.0:
            raise ValueError(
                "Latitude values must satisfy "
                "-90 <= south < north <= 90."
            )

        if self.south < -80.0 or self.north > 84.0:
            raise ValueError(
                "AOI falls outside the standard UTM latitude range."
            )

    @property
    def centroid(self) -> tuple[float, float]:
        """Return centroid longitude and latitude."""
        return (
            (self.west + self.east) / 2.0,
            (self.south + self.north) / 2.0,
        )

    def as_list(self) -> list[float]:
        """Return a STAC-compatible coordinate list."""
        return [self.west, self.south, self.east, self.north]


@dataclass(frozen=True)
class AoiSummary:
    """Validated AOI summary."""

    name: str
    bbox: BoundingBox
    centroid_longitude: float
    centroid_latitude: float
    utm_epsg: int
    utm_name: str
    width_meters: float
    height_meters: float
    geodesic_area_square_km: float


@dataclass(frozen=True)
class ReferenceGrid:
    """Common output grid used by every aligned raster."""

    crs_epsg: int
    resolution_meters: float
    width: int
    height: int
    transform: Affine
    bounds: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, Any]:
        """Convert the reference grid to JSON-safe metadata."""
        return {
            "crs": f"EPSG:{self.crs_epsg}",
            "epsg": self.crs_epsg,
            "resolution_meters": self.resolution_meters,
            "width": self.width,
            "height": self.height,
            "transform": [
                float(self.transform.a),
                float(self.transform.b),
                float(self.transform.c),
                float(self.transform.d),
                float(self.transform.e),
                float(self.transform.f),
            ],
            "bounds": [float(value) for value in self.bounds],
        }


def require_mapping(value: object, context: str) -> Mapping[str, Any]:
    """Validate and return a mapping value."""
    if not isinstance(value, Mapping):
        raise ConfigurationError(
            f"Configuration value '{context}' must be a mapping."
        )

    return value


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file does not exist: {config_path}"
        )

    with config_path.open("r", encoding="utf-8-sig") as config_file:
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
        with json_path.open("r", encoding="utf-8") as json_file:
            payload = json.load(json_file)
    except json.JSONDecodeError as error:
        raise AlignmentError(
            f"Invalid JSON document: {json_path}"
        ) from error

    if not isinstance(payload, Mapping):
        raise AlignmentError(
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
    """Write a JSON document atomically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

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


def longitude_to_utm_zone(longitude: float) -> int:
    """Convert longitude to UTM zone number."""
    if not -180.0 <= longitude <= 180.0:
        raise ValueError(
            f"Invalid longitude: {longitude}"
        )

    return 60 if longitude == 180.0 else int(
        (longitude + 180.0) // 6.0
    ) + 1


def determine_utm_epsg(longitude: float, latitude: float) -> int:
    """Determine a WGS84 UTM EPSG code."""
    if not -80.0 <= latitude <= 84.0:
        raise ValueError(
            f"Latitude is outside UTM coverage: {latitude}"
        )

    zone = longitude_to_utm_zone(longitude)
    return (32600 if latitude >= 0.0 else 32700) + zone


def ensure_single_utm_zone(bbox: BoundingBox) -> int:
    """Ensure both horizontal AOI edges belong to one UTM zone."""
    epsilon = 1e-10
    west_zone = longitude_to_utm_zone(bbox.west + epsilon)
    east_zone = longitude_to_utm_zone(bbox.east - epsilon)

    if west_zone != east_zone:
        raise ValueError(
            f"AOI spans UTM zones {west_zone} and {east_zone}."
        )

    return west_zone


def summarize_aoi(name: str, bbox: BoundingBox) -> AoiSummary:
    """Create a projected and geodesic AOI summary."""
    normalized_name = name.strip()

    if not normalized_name:
        raise ValueError("AOI name cannot be empty.")

    ensure_single_utm_zone(bbox)

    longitude, latitude = bbox.centroid
    epsg = determine_utm_epsg(longitude, latitude)

    transformer = Transformer.from_crs(
        "EPSG:4326",
        f"EPSG:{epsg}",
        always_xy=True,
    )

    longitudes = [
        bbox.west,
        bbox.east,
        bbox.east,
        bbox.west,
    ]
    latitudes = [
        bbox.south,
        bbox.south,
        bbox.north,
        bbox.north,
    ]

    eastings, northings = transformer.transform(
        longitudes,
        latitudes,
    )

    geod = Geod(ellps="WGS84")
    area_square_meters, _ = geod.polygon_area_perimeter(
        longitudes,
        latitudes,
    )

    return AoiSummary(
        name=normalized_name,
        bbox=bbox,
        centroid_longitude=longitude,
        centroid_latitude=latitude,
        utm_epsg=epsg,
        utm_name=CRS.from_epsg(epsg).name,
        width_meters=max(eastings) - min(eastings),
        height_meters=max(northings) - min(northings),
        geodesic_area_square_km=(
            abs(area_square_meters) / 1_000_000.0
        ),
    )


def update_config_with_aoi(
    config_path: Path,
    summary: AoiSummary,
) -> None:
    """Write validated AOI information into the YAML configuration."""
    config = load_yaml_config(config_path)

    acquisition = config.setdefault("acquisition", {})
    processing = config.setdefault("processing", {})

    if not isinstance(acquisition, dict):
        raise ConfigurationError(
            "acquisition must be a mapping."
        )

    if not isinstance(processing, dict):
        raise ConfigurationError(
            "processing must be a mapping."
        )

    acquisition["aoi_name"] = summary.name
    acquisition["bbox_wgs84"] = summary.bbox.as_list()
    processing["target_crs_strategy"] = "auto_utm"
    processing["target_epsg"] = summary.utm_epsg

    temporary_path = config_path.with_suffix(
        f"{config_path.suffix}.tmp"
    )

    with temporary_path.open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(
            config,
            config_file,
            sort_keys=False,
            allow_unicode=True,
        )

    temporary_path.replace(config_path)


def find_asset(
    download_report: Mapping[str, Any],
    period: str,
    asset_key: str,
) -> Mapping[str, Any]:
    """Find one downloaded asset record."""
    periods = require_mapping(
        download_report.get("periods"),
        "download_report.periods",
    )
    period_report = require_mapping(
        periods.get(period),
        f"download_report.periods.{period}",
    )

    assets = period_report.get("assets")

    if not isinstance(assets, Sequence):
        raise AlignmentError(
            f"Assets for period '{period}' are invalid."
        )

    for asset in assets:
        if (
            isinstance(asset, Mapping)
            and asset.get("asset_key") == asset_key
        ):
            return asset

    raise AlignmentError(
        f"Asset '{asset_key}' was not found for period '{period}'."
    )


def create_reference_grid(
    reference_path: Path,
    bbox_wgs84: BoundingBox,
    target_epsg: int,
    target_resolution: float,
) -> ReferenceGrid:
    """Build a pixel-aligned AOI grid from a native 10 m reference band.

    The AOI is expanded to complete reference pixels. This avoids fractional
    pixel offsets and guarantees that all periods use the exact same origin.
    """
    if target_resolution <= 0:
        raise ConfigurationError(
            "Target resolution must be greater than zero."
        )

    try:
        with rasterio.open(reference_path) as reference:
            reference_epsg = (
                reference.crs.to_epsg()
                if reference.crs is not None
                else None
            )

            if reference_epsg != target_epsg:
                raise AlignmentError(
                    f"Reference raster uses EPSG:{reference_epsg}; "
                    f"expected EPSG:{target_epsg}."
                )

            transform = reference.transform

            if not (
                math.isclose(transform.b, 0.0, abs_tol=1e-12)
                and math.isclose(transform.d, 0.0, abs_tol=1e-12)
                and transform.a > 0.0
                and transform.e < 0.0
            ):
                raise AlignmentError(
                    "Reference raster is not a north-up grid."
                )

            if not (
                math.isclose(
                    abs(transform.a),
                    target_resolution,
                    abs_tol=1e-6,
                )
                and math.isclose(
                    abs(transform.e),
                    target_resolution,
                    abs_tol=1e-6,
                )
            ):
                raise AlignmentError(
                    "Reference raster resolution does not match "
                    f"{target_resolution} metres."
                )

            left, bottom, right, top = transform_bounds(
                "EPSG:4326",
                reference.crs,
                bbox_wgs84.west,
                bbox_wgs84.south,
                bbox_wgs84.east,
                bbox_wgs84.north,
                densify_pts=21,
            )

            column_start = math.floor(
                (left - transform.c) / transform.a
            )
            column_stop = math.ceil(
                (right - transform.c) / transform.a
            )
            row_start = math.floor(
                (transform.f - top) / abs(transform.e)
            )
            row_stop = math.ceil(
                (transform.f - bottom) / abs(transform.e)
            )

            column_start = max(0, column_start)
            row_start = max(0, row_start)
            column_stop = min(reference.width, column_stop)
            row_stop = min(reference.height, row_stop)

            width = column_stop - column_start
            height = row_stop - row_start

            if width <= 0 or height <= 0:
                raise AlignmentError(
                    "The AOI does not overlap the reference raster."
                )

            output_transform = transform * Affine.translation(
                column_start,
                row_start,
            )

            output_bounds = array_bounds(
                height,
                width,
                output_transform,
            )

    except RasterioIOError as error:
        raise AlignmentError(
            f"Could not open reference raster: {reference_path}"
        ) from error

    return ReferenceGrid(
        crs_epsg=target_epsg,
        resolution_meters=target_resolution,
        width=width,
        height=height,
        transform=output_transform,
        bounds=tuple(float(value) for value in output_bounds),
    )


def parse_resampling_method(
    method_name: str,
) -> Resampling:
    """Convert a configured resampling name to Rasterio's enum."""
    supported_methods = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
    }

    normalized_name = method_name.strip().lower()

    if normalized_name not in supported_methods:
        raise ConfigurationError(
            f"Unsupported resampling method: {method_name}. "
            f"Supported values: {sorted(supported_methods)}"
        )

    return supported_methods[normalized_name]


def validate_existing_output(
    output_path: Path,
    grid: ReferenceGrid,
) -> None:
    """Validate that an existing output matches the reference grid."""
    try:
        with rasterio.open(output_path) as dataset:
            epsg = (
                dataset.crs.to_epsg()
                if dataset.crs is not None
                else None
            )

            if (
                epsg != grid.crs_epsg
                or dataset.width != grid.width
                or dataset.height != grid.height
                or dataset.transform != grid.transform
                or dataset.count != 1
            ):
                raise AlignmentError(
                    f"Existing output does not match the reference grid: "
                    f"{output_path}. Run with --overwrite."
                )

    except RasterioIOError as error:
        raise AlignmentError(
            f"Could not validate existing output: {output_path}"
        ) from error


def align_single_asset(
    source_path: Path,
    output_path: Path,
    asset_key: str,
    grid: ReferenceGrid,
    resampling: Resampling,
    compression: str,
    overwrite: bool,
) -> dict[str, Any]:
    """Crop and reproject one source raster to the common output grid."""
    if output_path.exists() and not overwrite:
        validate_existing_output(output_path, grid)

        return {
            "asset_key": asset_key,
            "source_path": str(source_path),
            "output_path": str(output_path),
            "status": "already_present",
            "sha256": calculate_sha256(output_path),
        }

    if not source_path.is_file():
        raise FileNotFoundError(
            f"Source raster does not exist: {source_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_name(
        f"{output_path.stem}.tmp{output_path.suffix}"
    )
    temporary_path.unlink(missing_ok=True)

    try:
        with rasterio.open(source_path) as source:
            if source.count != 1:
                raise AlignmentError(
                    f"{asset_key} must contain exactly one raster band."
                )

            source_epsg = (
                source.crs.to_epsg()
                if source.crs is not None
                else None
            )

            if source_epsg != grid.crs_epsg:
                raise AlignmentError(
                    f"{asset_key} uses EPSG:{source_epsg}; "
                    f"expected EPSG:{grid.crs_epsg}."
                )

            destination_nodata = 0
            source_nodata = (
                source.nodata
                if source.nodata is not None
                else 0
            )

            destination = np.full(
                (grid.height, grid.width),
                destination_nodata,
                dtype=source.dtypes[0],
            )

            reproject(
                source=rasterio.band(source, 1),
                destination=destination,
                src_transform=source.transform,
                src_crs=source.crs,
                src_nodata=source_nodata,
                dst_transform=grid.transform,
                dst_crs=f"EPSG:{grid.crs_epsg}",
                dst_nodata=destination_nodata,
                resampling=resampling,
                num_threads=2,
                init_dest_nodata=True,
            )

            profile = {
                "driver": "GTiff",
                "width": grid.width,
                "height": grid.height,
                "count": 1,
                "dtype": source.dtypes[0],
                "crs": f"EPSG:{grid.crs_epsg}",
                "transform": grid.transform,
                "nodata": destination_nodata,
                "compress": compression,
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
                "BIGTIFF": "IF_SAFER",
            }

            if asset_key.startswith("SCL"):
                profile["predictor"] = 1
            else:
                profile["predictor"] = 2

            with rasterio.open(
                temporary_path,
                "w",
                **profile,
            ) as output:
                output.write(destination, 1)
                output.update_tags(
                    source_path=str(source_path),
                    source_asset_key=asset_key,
                    processing="AOI crop and common-grid alignment",
                )

    except RasterioIOError as error:
        temporary_path.unlink(missing_ok=True)
        raise AlignmentError(
            f"Raster processing failed for {source_path}: {error}"
        ) from error
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    temporary_path.replace(output_path)

    valid_pixels = int(np.count_nonzero(destination))
    total_pixels = int(destination.size)

    return {
        "asset_key": asset_key,
        "source_path": str(source_path),
        "output_path": str(output_path),
        "status": "created",
        "dtype": str(destination.dtype),
        "nodata": 0,
        "resampling": resampling.name,
        "valid_pixel_fraction": (
            valid_pixels / total_pixels
            if total_pixels > 0
            else 0.0
        ),
        "minimum": int(destination.min()),
        "maximum": int(destination.max()),
        "size_bytes": output_path.stat().st_size,
        "sha256": calculate_sha256(output_path),
    }


def update_manifest_with_aligned_outputs(
    alignment_report: Mapping[str, Any],
    config_path: Path,
    manifest_path: Path,
) -> None:
    """Add aligned-raster lineage records to the dataset manifest."""
    config = load_yaml_config(config_path)
    project = require_mapping(config.get("project"), "project")
    acquisition = require_mapping(
        config.get("acquisition"),
        "acquisition",
    )

    dataset_version = str(
        project.get("dataset_version", "")
    ).strip()
    aoi_name = str(
        acquisition.get("aoi_name", "")
    ).strip()

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

    if manifest_path.exists():
        with manifest_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as manifest_file:
            for row in csv.DictReader(manifest_file):
                record_id = row.get("record_id", "").strip()

                if record_id:
                    records[record_id] = {
                        field: row.get(field, "")
                        for field in fieldnames
                    }

    periods = require_mapping(
        alignment_report.get("periods"),
        "alignment_report.periods",
    )

    created_at = datetime.now(timezone.utc).isoformat()

    for period in ("before", "after"):
        period_report = require_mapping(
            periods.get(period),
            f"alignment_report.periods.{period}",
        )

        item_id = str(period_report.get("item_id", ""))
        acquisition_datetime = str(
            period_report.get("acquisition_datetime", "")
        )
        acquisition_date = acquisition_datetime[:10]

        outputs = period_report.get("outputs")

        if not isinstance(outputs, Sequence):
            raise AlignmentError(
                f"Aligned outputs are invalid for period {period}."
            )

        for output in outputs:
            output_record = require_mapping(
                output,
                f"alignment_report.{period}.output",
            )

            asset_key = str(output_record.get("asset_key", ""))
            output_path = Path(
                str(output_record.get("output_path", ""))
            )

            record_id = (
                f"aligned-{period}-{item_id}-{asset_key}"
            )
            parent_id = (
                f"raw-{period}-{item_id}-{asset_key}"
            )

            records[record_id] = {
                "dataset_version": dataset_version,
                "record_id": record_id,
                "record_type": "aligned_satellite_band",
                "source": "derived_from_raw_satellite_band",
                "product_id": item_id,
                "acquisition_date": acquisition_date,
                "aoi_name": aoi_name,
                "crs": str(
                    alignment_report.get(
                        "reference_grid",
                        {},
                    ).get("crs", "")
                ),
                "resolution_meters": str(
                    alignment_report.get(
                        "reference_grid",
                        {},
                    ).get("resolution_meters", "")
                ),
                "relative_path": output_path.as_posix(),
                "sha256": str(output_record.get("sha256", "")),
                "parent_record_ids": parent_id,
                "processing_status": "aligned_10m",
                "created_at_utc": created_at,
            }

    temporary_path = manifest_path.with_suffix(
        f"{manifest_path.suffix}.tmp"
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

    temporary_path.replace(manifest_path)


def run_alignment(
    config_path: Path,
    download_report_override: Path | None,
    output_directory_override: Path | None,
    report_override: Path | None,
    reference_asset_override: str | None,
    overwrite: bool,
) -> tuple[dict[str, Any], Path, Path]:
    """Align all required raw bands to one AOI reference grid."""
    config = load_yaml_config(config_path)

    paths = require_mapping(config.get("paths"), "paths")
    acquisition = require_mapping(
        config.get("acquisition"),
        "acquisition",
    )
    processing = require_mapping(
        config.get("processing"),
        "processing",
    )

    bbox = BoundingBox.from_sequence(
        acquisition.get("bbox_wgs84", [])
    )

    target_epsg = int(processing.get("target_epsg"))
    target_resolution = float(
        processing.get("target_resolution_meters", 10.0)
    )

    output_directory = (
        output_directory_override
        if output_directory_override is not None
        else Path(
            str(
                processing.get(
                    "aligned_dir",
                    "data/processed/aligned",
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
                    "alignment_report",
                    "data/processed/aligned/alignment_report.json",
                )
            )
        )
    )

    download_report_path = (
        download_report_override
        if download_report_override is not None
        else Path(
            str(
                acquisition.get(
                    "band_download_report",
                    "data/raw/band_download_report.json",
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

    reference_asset = (
        reference_asset_override
        if reference_asset_override is not None
        else str(
            processing.get(
                "reference_asset",
                "B02_10m",
            )
        )
    )

    configured_assets = acquisition.get(
        "required_band_assets"
    )

    if (
        not isinstance(configured_assets, Sequence)
        or isinstance(configured_assets, (str, bytes))
        or not configured_assets
    ):
        raise ConfigurationError(
            "acquisition.required_band_assets must be a non-empty list."
        )

    required_assets = [
        str(asset_key)
        for asset_key in configured_assets
    ]

    download_report = load_json_mapping(
        download_report_path
    )

    reference_record = find_asset(
        download_report,
        "before",
        reference_asset,
    )
    reference_path = Path(
        str(reference_record.get("local_path", ""))
    )

    grid = create_reference_grid(
        reference_path=reference_path,
        bbox_wgs84=bbox,
        target_epsg=target_epsg,
        target_resolution=target_resolution,
    )

    continuous_resampling = parse_resampling_method(
        str(
            processing.get(
                "resampling_method_continuous",
                "bilinear",
            )
        )
    )
    categorical_resampling = parse_resampling_method(
        str(
            processing.get(
                "resampling_method_categorical",
                "nearest",
            )
        )
    )
    compression = str(
        processing.get(
            "output_compression",
            "deflate",
        )
    ).lower()

    periods = require_mapping(
        download_report.get("periods"),
        "download_report.periods",
    )

    period_results: dict[str, Any] = {}

    for period in ("before", "after"):
        period_source = require_mapping(
            periods.get(period),
            f"download_report.periods.{period}",
        )

        outputs: list[dict[str, Any]] = []

        for asset_key in required_assets:
            asset_record = find_asset(
                download_report,
                period,
                asset_key,
            )
            source_path = Path(
                str(asset_record.get("local_path", ""))
            )
            output_path = (
                output_directory
                / period
                / f"{asset_key}.tif"
            )

            resampling = (
                categorical_resampling
                if asset_key.startswith("SCL")
                else continuous_resampling
            )

            result = align_single_asset(
                source_path=source_path,
                output_path=output_path,
                asset_key=asset_key,
                grid=grid,
                resampling=resampling,
                compression=compression,
                overwrite=overwrite,
            )

            outputs.append(result)

            print(
                f"  [{period}] {asset_key}: "
                f"{result['status']} -> {output_path}"
            )

        period_results[period] = {
            "item_id": period_source.get("item_id"),
            "acquisition_datetime": period_source.get(
                "acquisition_datetime"
            ),
            "mgrs_tile": period_source.get("mgrs_tile"),
            "output_count": len(outputs),
            "outputs": outputs,
        }

    report = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "success",
        "source_download_report": str(
            download_report_path
        ),
        "source_download_report_sha256": calculate_sha256(
            download_report_path
        ),
        "reference_asset": reference_asset,
        "reference_grid": grid.to_dict(),
        "continuous_resampling": continuous_resampling.name,
        "categorical_resampling": categorical_resampling.name,
        "total_output_count": sum(
            result["output_count"]
            for result in period_results.values()
        ),
        "periods": period_results,
    }

    write_json_atomic(report, report_path)

    update_manifest_with_aligned_outputs(
        alignment_report=report,
        config_path=config_path,
        manifest_path=manifest_path,
    )

    return report, report_path, manifest_path


def print_aoi_summary(
    summary: AoiSummary,
    config_path: Path,
    updated: bool,
) -> None:
    """Print AOI validation output."""
    print("AOI validation succeeded")
    print(f"  Name: {summary.name}")
    print(f"  Target CRS: EPSG:{summary.utm_epsg}")
    print(
        "  Projected dimensions: "
        f"{summary.width_meters / 1000.0:.2f} km × "
        f"{summary.height_meters / 1000.0:.2f} km"
    )
    print(
        "  Geodesic area: "
        f"{summary.geodesic_area_square_km:.2f} km²"
    )
    print(
        f"  Configuration updated: "
        f"{config_path if updated else 'no'}"
    )


def print_alignment_summary(
    report: Mapping[str, Any],
    report_path: Path,
    manifest_path: Path,
) -> None:
    """Print aligned-raster completion information."""
    grid = require_mapping(
        report.get("reference_grid"),
        "alignment_report.reference_grid",
    )

    print("Sentinel-2 AOI alignment succeeded")
    print(f"  Status: {report.get('status')}")
    print(
        f"  Outputs created/validated: "
        f"{report.get('total_output_count')}"
    )
    print(
        f"  Grid: {grid.get('width')} × {grid.get('height')} pixels"
    )
    print(
        f"  CRS/resolution: {grid.get('crs')} at "
        f"{grid.get('resolution_meters')} m"
    )
    print(
        f"  Reflectance resampling: "
        f"{report.get('continuous_resampling')}"
    )
    print(
        f"  SCL resampling: "
        f"{report.get('categorical_resampling')}"
    )
    print(f"  Alignment report: {report_path}")
    print(f"  Dataset manifest: {manifest_path}")


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the GeoWatch geospatial CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate GeoWatch AOIs and align Sentinel-2 source rasters."
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    aoi_parser = subparsers.add_parser(
        "aoi",
        help="Validate and optionally save a WGS84 AOI.",
    )
    aoi_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data_config.yaml"),
    )
    aoi_parser.add_argument("--aoi-name", required=True)
    aoi_parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        required=True,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
    )
    aoi_parser.add_argument(
        "--write-config",
        action="store_true",
    )
    aoi_parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )

    align_parser = subparsers.add_parser(
        "align",
        help=(
            "Crop and align all selected bands to one common 10 m grid."
        ),
    )
    align_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data_config.yaml"),
    )
    align_parser.add_argument(
        "--download-report",
        type=Path,
        default=None,
    )
    align_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    align_parser.add_argument(
        "--report",
        type=Path,
        default=None,
    )
    align_parser.add_argument(
        "--reference-asset",
        default=None,
    )
    align_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing aligned outputs.",
    )
    align_parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )

    return parser


def main() -> int:
    """Run the requested GeoWatch geospatial command."""
    parser = build_argument_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    try:
        if args.command == "aoi":
            bbox = BoundingBox.from_sequence(args.bbox)
            summary = summarize_aoi(
                name=args.aoi_name,
                bbox=bbox,
            )

            if args.write_config:
                update_config_with_aoi(
                    config_path=args.config,
                    summary=summary,
                )

            print_aoi_summary(
                summary=summary,
                config_path=args.config,
                updated=args.write_config,
            )
            return 0

        if args.command == "align":
            report, report_path, manifest_path = run_alignment(
                config_path=args.config,
                download_report_override=args.download_report,
                output_directory_override=args.output_dir,
                report_override=args.report,
                reference_asset_override=args.reference_asset,
                overwrite=args.overwrite,
            )

            print_alignment_summary(
                report=report,
                report_path=report_path,
                manifest_path=manifest_path,
            )
            return 0

        parser.error(
            f"Unsupported command: {args.command}"
        )
        return 2

    except (
        FileNotFoundError,
        PermissionError,
        ConfigurationError,
        AlignmentError,
        RasterioIOError,
        ValueError,
        yaml.YAMLError,
    ) as error:
        LOGGER.error("%s", error)
        return 1

    except Exception:
        LOGGER.exception(
            "Unexpected geospatial processing failure."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
