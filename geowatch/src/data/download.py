"""Sentinel-2 scene discovery utilities for GeoWatch.

This module queries the Copernicus Data Space Ecosystem STAC catalogue for
Sentinel-2 Level-2A products intersecting the configured GeoWatch area of
interest.

This step discovers product metadata only. It does not download imagery.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from urllib.parse import urljoin

import requests
import yaml
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError
from pyproj import Transformer
from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as transform_geometry
from requests import Response, Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


LOGGER = logging.getLogger("geowatch.download")


class ConfigurationError(ValueError):
    """Raised when the GeoWatch configuration is missing required values."""


class StacResponseError(RuntimeError):
    """Raised when the STAC service returns an invalid response."""


@dataclass(frozen=True)
class SearchPeriod:
    """Date window used to discover satellite scenes.

    Attributes:
        label: Human-readable period name, such as ``before`` or ``after``.
        start_date: Inclusive beginning of the search window.
        end_date: Inclusive end of the search window.
    """

    label: str
    start_date: date
    end_date: date

    @property
    def stac_datetime_interval(self) -> str:
        """Return the period in STAC datetime interval format."""
        return (
            f"{self.start_date.isoformat()}T00:00:00Z/"
            f"{self.end_date.isoformat()}T23:59:59Z"
        )


@dataclass(frozen=True)
class DiscoverySettings:
    """Validated configuration for Sentinel-2 scene discovery."""

    stac_api_url: str
    collection: str
    aoi_name: str
    bbox_wgs84: tuple[float, float, float, float]
    cloud_cover_max_percent: float
    periods: tuple[SearchPeriod, ...]
    output_path: Path
    timeout_seconds: float
    retry_count: int
    page_size: int
    max_pages: int
    max_candidates_per_period: int


def require_mapping(
    value: object,
    context: str,
) -> Mapping[str, Any]:
    """Validate that a configuration value is a mapping.

    Args:
        value: Object to validate.
        context: Human-readable configuration location.

    Returns:
        The value as a mapping.

    Raises:
        ConfigurationError: If the value is not a mapping.
    """
    if not isinstance(value, Mapping):
        raise ConfigurationError(
            f"Configuration value '{context}' must be a mapping."
        )

    return value


def parse_date_value(
    value: object,
    context: str,
) -> date:
    """Parse an ISO-8601 date from YAML input.

    PyYAML may return ISO dates either as strings or as ``date`` objects.

    Args:
        value: Date-like configuration value.
        context: Human-readable configuration location.

    Returns:
        Parsed calendar date.

    Raises:
        ConfigurationError: If the value is missing or invalid.
    """
    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    if not isinstance(value, str):
        raise ConfigurationError(
            f"Configuration value '{context}' must be an ISO date."
        )

    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ConfigurationError(
            f"Configuration value '{context}' must use YYYY-MM-DD format; "
            f"received {value!r}."
        ) from error


def parse_bbox(value: object) -> tuple[float, float, float, float]:
    """Validate a STAC-compatible WGS84 bounding box.

    Args:
        value: Sequence ordered as west, south, east, north.

    Returns:
        Validated bounding-box tuple.

    Raises:
        ConfigurationError: If coordinates or ordering are invalid.
    """
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 4
    ):
        raise ConfigurationError(
            "acquisition.bbox_wgs84 must contain four values ordered as "
            "west, south, east, north."
        )

    try:
        west, south, east, north = (float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ConfigurationError(
            "All acquisition.bbox_wgs84 values must be numeric."
        ) from error

    if not -180.0 <= west < east <= 180.0:
        raise ConfigurationError(
            "WGS84 longitude values must satisfy "
            "-180 <= west < east <= 180."
        )

    if not -90.0 <= south < north <= 90.0:
        raise ConfigurationError(
            "WGS84 latitude values must satisfy "
            "-90 <= south < north <= 90."
        )

    return west, south, east, north


def parse_periods(value: object) -> tuple[SearchPeriod, ...]:
    """Parse configured before and after search periods.

    Args:
        value: Mapping of period labels to date windows.

    Returns:
        Validated periods ordered as before and after.

    Raises:
        ConfigurationError: If periods are missing, malformed, or overlap
            incorrectly.
    """
    periods_mapping = require_mapping(value, "acquisition.periods")
    parsed_periods: list[SearchPeriod] = []

    for label in ("before", "after"):
        period_mapping = require_mapping(
            periods_mapping.get(label),
            f"acquisition.periods.{label}",
        )

        start_date = parse_date_value(
            period_mapping.get("start_date"),
            f"acquisition.periods.{label}.start_date",
        )
        end_date = parse_date_value(
            period_mapping.get("end_date"),
            f"acquisition.periods.{label}.end_date",
        )

        if start_date > end_date:
            raise ConfigurationError(
                f"Period '{label}' starts after it ends."
            )

        parsed_periods.append(
            SearchPeriod(
                label=label,
                start_date=start_date,
                end_date=end_date,
            )
        )

    before_period, after_period = parsed_periods

    if before_period.end_date >= after_period.start_date:
        raise ConfigurationError(
            "The before period must finish before the after period starts."
        )

    return tuple(parsed_periods)


def load_yaml_config(config_path: Path) -> Mapping[str, Any]:
    """Load a GeoWatch YAML configuration file.

    Args:
        config_path: Path to the YAML configuration.

    Returns:
        Parsed root configuration mapping.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ConfigurationError: If the YAML root is invalid.
        yaml.YAMLError: If YAML parsing fails.
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file does not exist: {config_path}"
        )

    if not config_path.is_file():
        raise ConfigurationError(
            f"Configuration path is not a file: {config_path}"
        )

    with config_path.open("r", encoding="utf-8-sig") as config_file:
        config = yaml.safe_load(config_file)

    return require_mapping(config, "root")


def load_discovery_settings(
    config_path: Path,
    output_override: Path | None,
) -> DiscoverySettings:
    """Load and validate Sentinel-2 discovery settings.

    Args:
        config_path: GeoWatch YAML configuration.
        output_override: Optional CLI output-path override.

    Returns:
        Validated discovery settings.

    Raises:
        ConfigurationError: If required values are missing or invalid.
    """
    config = load_yaml_config(config_path)
    acquisition = require_mapping(
        config.get("acquisition"),
        "acquisition",
    )

    stac_api_url = str(
        acquisition.get(
            "stac_api_url",
            "https://stac.dataspace.copernicus.eu/v1",
        )
    ).rstrip("/")

    collection = str(
        acquisition.get("collection", "")
    ).strip()

    aoi_name = str(
        acquisition.get("aoi_name", "")
    ).strip()

    if not collection:
        raise ConfigurationError(
            "acquisition.collection cannot be empty."
        )

    if not aoi_name:
        raise ConfigurationError(
            "acquisition.aoi_name cannot be empty."
        )

    bbox = parse_bbox(acquisition.get("bbox_wgs84"))
    periods = parse_periods(acquisition.get("periods"))

    cloud_cover_max_percent = float(
        acquisition.get(
            "scene_search_cloud_cover_max_percent",
            30.0,
        )
    )

    if not 0.0 <= cloud_cover_max_percent <= 100.0:
        raise ConfigurationError(
            "scene_search_cloud_cover_max_percent must be between 0 and 100."
        )

    configured_output = Path(
        str(
            acquisition.get(
                "discovery_output",
                "data/raw/sentinel2_scene_catalog.json",
            )
        )
    )

    output_path = (
        output_override
        if output_override is not None
        else configured_output
    )

    timeout_seconds = float(
        acquisition.get("request_timeout_seconds", 60)
    )
    retry_count = int(
        acquisition.get("request_retries", 4)
    )
    page_size = int(
        acquisition.get("page_size", 100)
    )
    max_pages = int(
        acquisition.get("max_pages", 10)
    )
    max_candidates = int(
        acquisition.get("max_candidates_per_period", 25)
    )

    if timeout_seconds <= 0:
        raise ConfigurationError(
            "request_timeout_seconds must be greater than zero."
        )

    if retry_count < 0:
        raise ConfigurationError(
            "request_retries cannot be negative."
        )

    if not 1 <= page_size <= 1000:
        raise ConfigurationError(
            "page_size must be between 1 and 1000."
        )

    if max_pages <= 0:
        raise ConfigurationError(
            "max_pages must be greater than zero."
        )

    if max_candidates <= 0:
        raise ConfigurationError(
            "max_candidates_per_period must be greater than zero."
        )

    return DiscoverySettings(
        stac_api_url=stac_api_url,
        collection=collection,
        aoi_name=aoi_name,
        bbox_wgs84=bbox,
        cloud_cover_max_percent=cloud_cover_max_percent,
        periods=periods,
        output_path=output_path,
        timeout_seconds=timeout_seconds,
        retry_count=retry_count,
        page_size=page_size,
        max_pages=max_pages,
        max_candidates_per_period=max_candidates,
    )


def create_http_session(retry_count: int) -> Session:
    """Create an HTTP session with retry and connection-pooling support.

    STAC search POST requests are read-only, so retrying them is safe.

    Args:
        retry_count: Maximum retry attempts for transient failures.

    Returns:
        Configured Requests session.
    """
    retry_policy = Retry(
        total=retry_count,
        connect=retry_count,
        read=retry_count,
        status=retry_count,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retry_policy,
        pool_connections=4,
        pool_maxsize=4,
    )

    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/geo+json, application/json",
            "Content-Type": "application/json",
            "User-Agent": "GeoWatch/0.1 Sentinel-2 discovery",
        }
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


def decode_json_response(response: Response) -> Mapping[str, Any]:
    """Validate an HTTP response and decode its JSON body.

    Args:
        response: Requests response object.

    Returns:
        Parsed JSON mapping.

    Raises:
        StacResponseError: If the status or JSON body is invalid.
    """
    try:
        response.raise_for_status()
    except requests.HTTPError as error:
        response_excerpt = response.text[:500].replace("\n", " ")
        raise StacResponseError(
            f"STAC request failed with HTTP {response.status_code}: "
            f"{response_excerpt}"
        ) from error

    try:
        payload = response.json()
    except requests.JSONDecodeError as error:
        raise StacResponseError(
            "STAC service returned a non-JSON response."
        ) from error

    if not isinstance(payload, Mapping):
        raise StacResponseError(
            "STAC response root must be a JSON object."
        )

    return payload


def find_next_link(
    payload: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Return the STAC pagination link whose relation is ``next``."""
    links = payload.get("links", [])

    if not isinstance(links, Sequence):
        return None

    for link in links:
        if isinstance(link, Mapping) and link.get("rel") == "next":
            return link

    return None


def iter_stac_items(
    session: Session,
    search_url: str,
    initial_body: Mapping[str, Any],
    timeout_seconds: float,
    max_pages: int,
) -> Iterator[Mapping[str, Any]]:
    """Iterate through paginated STAC Item Search results.

    Args:
        session: Configured HTTP session.
        search_url: Initial STAC Item Search endpoint.
        initial_body: Initial POST request body.
        timeout_seconds: Per-request timeout.
        max_pages: Safety limit for pagination.

    Yields:
        STAC item mappings.

    Raises:
        StacResponseError: If the response structure is invalid.
    """
    request_url = search_url
    request_method = "POST"
    request_body: Mapping[str, Any] | None = initial_body

    for page_number in range(1, max_pages + 1):
        LOGGER.debug(
            "Requesting STAC page %d using %s %s",
            page_number,
            request_method,
            request_url,
        )

        if request_method == "POST":
            response = session.post(
                request_url,
                json=request_body,
                timeout=timeout_seconds,
            )
        else:
            response = session.get(
                request_url,
                timeout=timeout_seconds,
            )

        payload = decode_json_response(response)
        features = payload.get("features")

        if not isinstance(features, Sequence):
            raise StacResponseError(
                "STAC response does not contain a valid features array."
            )

        for feature in features:
            if isinstance(feature, Mapping):
                yield feature
            else:
                LOGGER.warning(
                    "Ignoring malformed STAC feature on page %d.",
                    page_number,
                )

        next_link = find_next_link(payload)

        if next_link is None:
            return

        next_href = next_link.get("href")

        if not isinstance(next_href, str) or not next_href.strip():
            raise StacResponseError(
                "STAC next-page link has no valid href."
            )

        request_url = urljoin(request_url, next_href)
        request_method = str(
            next_link.get("method", "GET")
        ).upper()

        if request_method not in {"GET", "POST"}:
            raise StacResponseError(
                f"Unsupported STAC pagination method: {request_method}"
            )

        if request_method == "POST":
            next_body = next_link.get("body")
            request_body = (
                next_body
                if isinstance(next_body, Mapping)
                else initial_body
            )
        else:
            request_body = None

    LOGGER.warning(
        "Stopped pagination after configured maximum of %d pages.",
        max_pages,
    )


def extract_cloud_cover(item: Mapping[str, Any]) -> float | None:
    """Extract the STAC ``eo:cloud_cover`` value from an item."""
    properties = item.get("properties")

    if not isinstance(properties, Mapping):
        return None

    cloud_cover = properties.get("eo:cloud_cover")

    if cloud_cover is None:
        return None

    try:
        return float(cloud_cover)
    except (TypeError, ValueError):
        return None


def extract_mgrs_tile(
    item_id: str,
    properties: Mapping[str, Any],
) -> str | None:
    """Extract a Sentinel-2 MGRS tile identifier.

    The method first checks standard metadata fields and then falls back to
    parsing product identifiers such as ``..._T44QKE_...``.
    """
    direct_tile = properties.get("s2:mgrs_tile")

    if isinstance(direct_tile, str) and direct_tile.strip():
        return direct_tile.removeprefix("T")

    zone = properties.get("mgrs:utm_zone")
    latitude_band = properties.get("mgrs:latitude_band")
    grid_square = properties.get("mgrs:grid_square")

    if (
        zone is not None
        and isinstance(latitude_band, str)
        and isinstance(grid_square, str)
    ):
        try:
            return (
                f"{int(zone):02d}"
                f"{latitude_band.upper()}"
                f"{grid_square.upper()}"
            )
        except (TypeError, ValueError):
            pass

    identifier_match = re.search(
        r"_T(?P<tile>\d{2}[A-Z]{3})_",
        item_id,
    )

    if identifier_match is not None:
        return identifier_match.group("tile")

    return None


def normalize_item(item: Mapping[str, Any]) -> dict[str, Any] | None:
    """Convert a STAC item into the GeoWatch discovery schema.

    Args:
        item: Raw STAC item.

    Returns:
        Normalized scene dictionary, or ``None`` if required metadata is
        missing.
    """
    item_id = item.get("id")
    properties = item.get("properties")

    if not isinstance(item_id, str) or not item_id.strip():
        LOGGER.warning("Ignoring STAC item without a valid ID.")
        return None

    if not isinstance(properties, Mapping):
        LOGGER.warning(
            "Ignoring STAC item %s without valid properties.",
            item_id,
        )
        return None

    acquisition_datetime = (
        properties.get("datetime")
        or properties.get("start_datetime")
    )

    if not isinstance(acquisition_datetime, str):
        LOGGER.warning(
            "Ignoring STAC item %s without acquisition datetime.",
            item_id,
        )
        return None

    cloud_cover = extract_cloud_cover(item)

    if cloud_cover is None:
        LOGGER.warning(
            "Ignoring STAC item %s without eo:cloud_cover metadata.",
            item_id,
        )
        return None

    assets = item.get("assets", {})
    links = item.get("links", [])

    selected_links: list[Mapping[str, Any]] = []

    if isinstance(links, Sequence):
        for link in links:
            if (
                isinstance(link, Mapping)
                and link.get("rel") in {"self", "collection", "parent"}
            ):
                selected_links.append(link)

    return {
        "item_id": item_id,
        "collection": item.get("collection"),
        "acquisition_datetime": acquisition_datetime,
        "cloud_cover_percent": round(cloud_cover, 4),
        "platform": properties.get("platform"),
        "constellation": properties.get("constellation"),
        "processing_level": properties.get("processing:level"),
        "product_type": properties.get("product:type"),
        "mgrs_tile": extract_mgrs_tile(item_id, properties),
        "bbox": item.get("bbox"),
        "geometry": item.get("geometry"),
        "assets": assets if isinstance(assets, Mapping) else {},
        "links": selected_links,
    }


def build_search_body(
    settings: DiscoverySettings,
    period: SearchPeriod,
) -> dict[str, Any]:
    """Build a CDSE STAC Item Search POST body.

    Args:
        settings: Validated discovery configuration.
        period: Date window to query.

    Returns:
        STAC POST body.
    """
    return {
        "collections": [settings.collection],
        "bbox": list(settings.bbox_wgs84),
        "datetime": period.stac_datetime_interval,
        "query": {
            "eo:cloud_cover": {
                "lte": settings.cloud_cover_max_percent,
            }
        },
        "sortby": [
            {
                "field": "properties.eo:cloud_cover",
                "direction": "asc",
            },
            {
                "field": "properties.datetime",
                "direction": "asc",
            },
        ],
        "limit": settings.page_size,
    }


def discover_period(
    session: Session,
    settings: DiscoverySettings,
    period: SearchPeriod,
) -> dict[str, Any]:
    """Discover and rank candidate scenes for one temporal period."""
    search_url = f"{settings.stac_api_url}/search"
    search_body = build_search_body(settings, period)

    candidates_by_id: dict[str, dict[str, Any]] = {}
    raw_item_count = 0

    for raw_item in iter_stac_items(
        session=session,
        search_url=search_url,
        initial_body=search_body,
        timeout_seconds=settings.timeout_seconds,
        max_pages=settings.max_pages,
    ):
        raw_item_count += 1
        normalized_item = normalize_item(raw_item)

        if normalized_item is None:
            continue

        cloud_cover = normalized_item["cloud_cover_percent"]

        if cloud_cover > settings.cloud_cover_max_percent:
            continue

        candidates_by_id[normalized_item["item_id"]] = normalized_item

    ranked_candidates = sorted(
        candidates_by_id.values(),
        key=lambda candidate: (
            candidate["cloud_cover_percent"],
            candidate["acquisition_datetime"],
            candidate["item_id"],
        ),
    )

    selected_candidates = ranked_candidates[
        : settings.max_candidates_per_period
    ]

    return {
        "label": period.label,
        "start_date": period.start_date.isoformat(),
        "end_date": period.end_date.isoformat(),
        "stac_datetime_interval": period.stac_datetime_interval,
        "raw_items_received": raw_item_count,
        "candidate_count": len(selected_candidates),
        "candidates": selected_candidates,
    }


def build_catalog(
    session: Session,
    settings: DiscoverySettings,
) -> dict[str, Any]:
    """Discover Sentinel-2 candidates for all configured periods."""
    period_results: dict[str, Any] = {}

    for period in settings.periods:
        LOGGER.info(
            "Searching period '%s': %s to %s",
            period.label,
            period.start_date,
            period.end_date,
        )

        period_result = discover_period(
            session=session,
            settings=settings,
            period=period,
        )

        if period_result["candidate_count"] == 0:
            raise StacResponseError(
                f"No suitable Sentinel-2 candidates were found for "
                f"period '{period.label}'."
            )

        period_results[period.label] = period_result

    return {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "provider": "Copernicus Data Space Ecosystem",
        "stac_api_url": settings.stac_api_url,
        "collection": settings.collection,
        "aoi": {
            "name": settings.aoi_name,
            "bbox_wgs84": list(settings.bbox_wgs84),
        },
        "scene_search_cloud_cover_max_percent": (
            settings.cloud_cover_max_percent
        ),
        "periods": period_results,
    }


def write_json_atomic(
    payload: Mapping[str, Any],
    output_path: Path,
) -> None:
    """Write JSON atomically to avoid partially written catalogue files."""
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


def print_discovery_summary(
    catalog: Mapping[str, Any],
    output_path: Path,
) -> None:
    """Print a concise candidate summary for manual verification."""
    print("Sentinel-2 discovery succeeded")
    print(f"  AOI: {catalog['aoi']['name']}")
    print(f"  Collection: {catalog['collection']}")
    print(
        "  Scene cloud threshold: "
        f"{catalog['scene_search_cloud_cover_max_percent']:.1f}%"
    )

    periods = catalog["periods"]

    for period_label in ("before", "after"):
        period = periods[period_label]
        candidates = period["candidates"]

        print(
            f"  {period_label.title()} period: "
            f"{period['start_date']} to {period['end_date']}"
        )
        print(
            f"    Candidates retained: {period['candidate_count']}"
        )

        for candidate in candidates[:5]:
            print(
                "    - "
                f"{candidate['acquisition_datetime']} | "
                f"cloud={candidate['cloud_cover_percent']:.2f}% | "
                f"tile={candidate['mgrs_tile'] or 'unknown'} | "
                f"{candidate['item_id']}"
            )

    print(f"  Catalogue written: {output_path}")


def calculate_file_sha256(
    file_path: Path,
    chunk_size_bytes: int = 1_048_576,
) -> str:
    """Calculate a SHA-256 checksum for a local file.

    Args:
        file_path: File whose contents will be hashed.
        chunk_size_bytes: Number of bytes read per iteration.

    Returns:
        Lowercase hexadecimal SHA-256 digest.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the path is not a file or chunk size is invalid.
    """
    import hashlib

    if not file_path.exists():
        raise FileNotFoundError(
            f"File does not exist: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"Checksum path is not a file: {file_path}"
        )

    if chunk_size_bytes <= 0:
        raise ValueError(
            "chunk_size_bytes must be greater than zero."
        )

    digest = hashlib.sha256()

    with file_path.open("rb") as input_file:
        while True:
            chunk = input_file.read(chunk_size_bytes)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def parse_candidate_datetime(
    candidate: Mapping[str, Any],
) -> datetime:
    """Parse a candidate acquisition timestamp as timezone-aware UTC.

    Args:
        candidate: Normalized Sentinel-2 candidate metadata.

    Returns:
        Parsed UTC datetime.

    Raises:
        StacResponseError: If acquisition metadata is missing or invalid.
    """
    value = candidate.get("acquisition_datetime")

    if not isinstance(value, str) or not value.strip():
        raise StacResponseError(
            "Candidate has no valid acquisition_datetime."
        )

    normalized_value = value.replace("Z", "+00:00")

    try:
        parsed_datetime = datetime.fromisoformat(normalized_value)
    except ValueError as error:
        raise StacResponseError(
            f"Invalid candidate datetime: {value!r}"
        ) from error

    if parsed_datetime.tzinfo is None:
        parsed_datetime = parsed_datetime.replace(
            tzinfo=timezone.utc
        )

    return parsed_datetime.astimezone(timezone.utc)


def infer_candidate_platform(
    candidate: Mapping[str, Any],
) -> str:
    """Infer the Sentinel platform from metadata or product identifier."""
    platform = candidate.get("platform")

    if isinstance(platform, str) and platform.strip():
        return platform.strip().upper()

    item_id = candidate.get("item_id")

    if isinstance(item_id, str) and item_id.startswith(
        ("S2A_", "S2B_", "S2C_")
    ):
        return item_id[:3]

    return "UNKNOWN"


def extract_candidate_mgrs_zone(
    candidate: Mapping[str, Any],
) -> int | None:
    """Extract the numerical UTM zone from an MGRS tile identifier."""
    mgrs_tile = candidate.get("mgrs_tile")

    if not isinstance(mgrs_tile, str):
        return None

    normalized_tile = mgrs_tile.strip().upper().removeprefix("T")

    if len(normalized_tile) < 2:
        return None

    try:
        return int(normalized_tile[:2])
    except ValueError:
        return None


def load_json_mapping(
    json_path: Path,
) -> Mapping[str, Any]:
    """Load a JSON file and validate that its root is an object."""
    if not json_path.exists():
        raise FileNotFoundError(
            f"JSON file does not exist: {json_path}"
        )

    if not json_path.is_file():
        raise ValueError(
            f"JSON path is not a file: {json_path}"
        )

    try:
        with json_path.open(
            "r",
            encoding="utf-8",
        ) as input_file:
            payload = json.load(input_file)
    except json.JSONDecodeError as error:
        raise StacResponseError(
            f"Invalid JSON file: {json_path}"
        ) from error

    if not isinstance(payload, Mapping):
        raise StacResponseError(
            f"JSON root must be an object: {json_path}"
        )

    return payload


def create_projected_aoi(
    bbox_wgs84: Sequence[float],
    target_epsg: int,
) -> tuple[BaseGeometry, Transformer]:
    """Create a projected AOI polygon and coordinate transformer.

    Areas are measured in the configured projected CRS rather than degrees.

    Args:
        bbox_wgs84: Bounding box ordered as west, south, east, north.
        target_epsg: Projected CRS EPSG code.

    Returns:
        Projected AOI geometry and reusable transformer.
    """
    west, south, east, north = (
        float(coordinate) for coordinate in bbox_wgs84
    )

    transformer = Transformer.from_crs(
        "EPSG:4326",
        f"EPSG:{target_epsg}",
        always_xy=True,
    )

    aoi_wgs84 = box(
        west,
        south,
        east,
        north,
    )

    projected_aoi = transform_geometry(
        transformer.transform,
        aoi_wgs84,
    )

    if projected_aoi.is_empty or projected_aoi.area <= 0.0:
        raise ValueError(
            "Projected AOI has no measurable area."
        )

    return projected_aoi, transformer


def calculate_candidate_overlap_ratio(
    candidate: Mapping[str, Any],
    projected_aoi: BaseGeometry,
    transformer: Transformer,
) -> float:
    """Calculate the fraction of the AOI covered by a STAC scene.

    Args:
        candidate: Normalized STAC candidate.
        projected_aoi: AOI polygon in the target projected CRS.
        transformer: WGS84-to-target-CRS transformer.

    Returns:
        AOI coverage ratio between zero and one.

    Raises:
        StacResponseError: If candidate geometry is missing or invalid.
    """
    geometry_mapping = candidate.get("geometry")

    if not isinstance(geometry_mapping, Mapping):
        raise StacResponseError(
            f"Candidate {candidate.get('item_id')} has no valid geometry."
        )

    try:
        scene_wgs84 = shape(geometry_mapping)

        if not scene_wgs84.is_valid:
            scene_wgs84 = scene_wgs84.buffer(0)

        projected_scene = transform_geometry(
            transformer.transform,
            scene_wgs84,
        )

        intersection = projected_aoi.intersection(
            projected_scene
        )

    except Exception as error:
        raise StacResponseError(
            "Failed to evaluate geometry for candidate "
            f"{candidate.get('item_id')}."
        ) from error

    if intersection.is_empty:
        return 0.0

    ratio = intersection.area / projected_aoi.area

    return max(0.0, min(1.0, float(ratio)))


def seasonal_day_distance(
    first_datetime: datetime,
    second_datetime: datetime,
) -> int:
    """Calculate circular calendar-day distance between two acquisitions.

    Using calendar-day distance helps reduce phenology and solar-angle
    differences between the before and after observations.
    """
    reference_year = 2000

    first_reference = date(
        reference_year,
        first_datetime.month,
        first_datetime.day,
    )
    second_reference = date(
        reference_year,
        second_datetime.month,
        second_datetime.day,
    )

    first_day = first_reference.timetuple().tm_yday
    second_day = second_reference.timetuple().tm_yday

    direct_distance = abs(first_day - second_day)

    return min(
        direct_distance,
        366 - direct_distance,
    )


def prepare_eligible_candidates(
    candidates: Sequence[Any],
    period_label: str,
    target_mgrs_zone: int,
    minimum_overlap_ratio: float,
    projected_aoi: BaseGeometry,
    transformer: Transformer,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Filter scene candidates using zone and spatial-coverage rules."""
    eligible: list[dict[str, Any]] = []

    rejection_counts = {
        "malformed": 0,
        "wrong_mgrs_zone": 0,
        "insufficient_aoi_overlap": 0,
    }

    for candidate_value in candidates:
        if not isinstance(candidate_value, Mapping):
            rejection_counts["malformed"] += 1
            continue

        candidate = dict(candidate_value)
        mgrs_zone = extract_candidate_mgrs_zone(candidate)

        if mgrs_zone != target_mgrs_zone:
            rejection_counts["wrong_mgrs_zone"] += 1
            continue

        try:
            overlap_ratio = calculate_candidate_overlap_ratio(
                candidate=candidate,
                projected_aoi=projected_aoi,
                transformer=transformer,
            )
        except StacResponseError as error:
            LOGGER.warning(
                "Rejecting %s candidate: %s",
                period_label,
                error,
            )
            rejection_counts["malformed"] += 1
            continue

        if overlap_ratio < minimum_overlap_ratio:
            rejection_counts[
                "insufficient_aoi_overlap"
            ] += 1
            continue

        candidate["aoi_overlap_ratio"] = round(
            overlap_ratio,
            6,
        )
        candidate["inferred_platform"] = (
            infer_candidate_platform(candidate)
        )

        eligible.append(candidate)

    return eligible, rejection_counts


def select_scene_pair(
    catalog_path: Path,
    config_path: Path,
    minimum_overlap_ratio: float,
) -> dict[str, Any]:
    """Select the best spatially and seasonally matched scene pair.

    Pair selection requires:

    * both scenes to cover the configured AOI,
    * both scenes to belong to the target UTM/MGRS zone,
    * both scenes to use the same MGRS tile.

    The ranking score combines scene cloud metadata, calendar-day distance,
    and a penalty when the satellite platforms differ.
    """
    if not 0.0 < minimum_overlap_ratio <= 1.0:
        raise ConfigurationError(
            "minimum_overlap_ratio must be greater than zero "
            "and no greater than one."
        )

    catalog = load_json_mapping(catalog_path)
    config = load_yaml_config(config_path)

    acquisition = require_mapping(
        config.get("acquisition"),
        "acquisition",
    )
    processing = require_mapping(
        config.get("processing"),
        "processing",
    )

    bbox_wgs84 = parse_bbox(
        acquisition.get("bbox_wgs84")
    )

    try:
        target_epsg = int(
            processing.get("target_epsg")
        )
    except (TypeError, ValueError) as error:
        raise ConfigurationError(
            "processing.target_epsg must be an integer."
        ) from error

    target_mgrs_zone = target_epsg % 100

    if not 1 <= target_mgrs_zone <= 60:
        raise ConfigurationError(
            f"Could not derive a UTM zone from EPSG:{target_epsg}."
        )

    projected_aoi, transformer = create_projected_aoi(
        bbox_wgs84=bbox_wgs84,
        target_epsg=target_epsg,
    )

    periods = require_mapping(
        catalog.get("periods"),
        "catalog.periods",
    )

    before_period = require_mapping(
        periods.get("before"),
        "catalog.periods.before",
    )
    after_period = require_mapping(
        periods.get("after"),
        "catalog.periods.after",
    )

    before_values = before_period.get("candidates")
    after_values = after_period.get("candidates")

    if not isinstance(before_values, Sequence):
        raise StacResponseError(
            "Before-period candidates must be an array."
        )

    if not isinstance(after_values, Sequence):
        raise StacResponseError(
            "After-period candidates must be an array."
        )

    eligible_before, rejected_before = (
        prepare_eligible_candidates(
            candidates=before_values,
            period_label="before",
            target_mgrs_zone=target_mgrs_zone,
            minimum_overlap_ratio=minimum_overlap_ratio,
            projected_aoi=projected_aoi,
            transformer=transformer,
        )
    )

    eligible_after, rejected_after = (
        prepare_eligible_candidates(
            candidates=after_values,
            period_label="after",
            target_mgrs_zone=target_mgrs_zone,
            minimum_overlap_ratio=minimum_overlap_ratio,
            projected_aoi=projected_aoi,
            transformer=transformer,
        )
    )

    if not eligible_before:
        raise StacResponseError(
            "No before-period candidate passed spatial validation."
        )

    if not eligible_after:
        raise StacResponseError(
            "No after-period candidate passed spatial validation."
        )

    ranked_pairs: list[
        tuple[
            tuple[float, float, str, str],
            dict[str, Any],
            dict[str, Any],
            dict[str, Any],
        ]
    ] = []

    for before_candidate in eligible_before:
        for after_candidate in eligible_after:
            if (
                before_candidate.get("mgrs_tile")
                != after_candidate.get("mgrs_tile")
            ):
                continue

            before_datetime = parse_candidate_datetime(
                before_candidate
            )
            after_datetime = parse_candidate_datetime(
                after_candidate
            )

            before_cloud = float(
                before_candidate["cloud_cover_percent"]
            )
            after_cloud = float(
                after_candidate["cloud_cover_percent"]
            )

            calendar_distance_days = seasonal_day_distance(
                before_datetime,
                after_datetime,
            )

            before_platform = str(
                before_candidate["inferred_platform"]
            )
            after_platform = str(
                after_candidate["inferred_platform"]
            )

            same_platform = (
                before_platform == after_platform
                and before_platform != "UNKNOWN"
            )

            cloud_score = before_cloud + after_cloud
            seasonal_penalty = (
                calendar_distance_days * 0.10
            )
            platform_penalty = (
                0.0 if same_platform else 5.0
            )

            pair_score = (
                cloud_score
                + seasonal_penalty
                + platform_penalty
            )

            minimum_pair_overlap = min(
                float(
                    before_candidate[
                        "aoi_overlap_ratio"
                    ]
                ),
                float(
                    after_candidate[
                        "aoi_overlap_ratio"
                    ]
                ),
            )

            score_details = {
                "total_score": round(pair_score, 6),
                "cloud_score": round(cloud_score, 6),
                "seasonal_penalty": round(
                    seasonal_penalty,
                    6,
                ),
                "platform_penalty": platform_penalty,
                "calendar_day_distance": (
                    calendar_distance_days
                ),
                "same_platform": same_platform,
                "minimum_pair_aoi_overlap_ratio": (
                    minimum_pair_overlap
                ),
            }

            ranking_key = (
                pair_score,
                -minimum_pair_overlap,
                str(before_candidate["item_id"]),
                str(after_candidate["item_id"]),
            )

            ranked_pairs.append(
                (
                    ranking_key,
                    before_candidate,
                    after_candidate,
                    score_details,
                )
            )

    if not ranked_pairs:
        raise StacResponseError(
            "No before/after pair shares the same validated MGRS tile."
        )

    ranked_pairs.sort(key=lambda pair: pair[0])

    (
        _,
        selected_before,
        selected_after,
        selected_score,
    ) = ranked_pairs[0]

    return {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "source_catalog": str(catalog_path),
        "source_catalog_sha256": calculate_file_sha256(
            catalog_path
        ),
        "aoi": {
            "name": acquisition.get("aoi_name"),
            "bbox_wgs84": list(bbox_wgs84),
            "target_epsg": target_epsg,
            "target_mgrs_zone": target_mgrs_zone,
        },
        "selection_rules": {
            "minimum_aoi_overlap_ratio": (
                minimum_overlap_ratio
            ),
            "same_mgrs_tile_required": True,
            "same_platform_preferred": True,
            "seasonal_alignment_preferred": True,
        },
        "candidate_statistics": {
            "before_discovered": len(before_values),
            "after_discovered": len(after_values),
            "before_eligible": len(eligible_before),
            "after_eligible": len(eligible_after),
            "before_rejections": rejected_before,
            "after_rejections": rejected_after,
            "valid_pair_count": len(ranked_pairs),
        },
        "selected_pair": {
            "mgrs_tile": selected_before.get("mgrs_tile"),
            "score": selected_score,
            "before": selected_before,
            "after": selected_after,
        },
    }


def print_scene_selection_summary(
    selection: Mapping[str, Any],
    output_path: Path,
) -> None:
    """Print the selected scene pair and validation statistics."""
    selected_pair = require_mapping(
        selection.get("selected_pair"),
        "selection.selected_pair",
    )

    before = require_mapping(
        selected_pair.get("before"),
        "selection.selected_pair.before",
    )
    after = require_mapping(
        selected_pair.get("after"),
        "selection.selected_pair.after",
    )
    score = require_mapping(
        selected_pair.get("score"),
        "selection.selected_pair.score",
    )
    statistics = require_mapping(
        selection.get("candidate_statistics"),
        "selection.candidate_statistics",
    )

    print("Sentinel-2 pair selection succeeded")
    print(
        f"  Selected MGRS tile: "
        f"{selected_pair.get('mgrs_tile')}"
    )
    print(
        f"  Eligible candidates: "
        f"before={statistics.get('before_eligible')}, "
        f"after={statistics.get('after_eligible')}"
    )
    print(
        "  Before: "
        f"{before.get('acquisition_datetime')} | "
        f"cloud={float(before.get('cloud_cover_percent')):.2f}% | "
        f"overlap={float(before.get('aoi_overlap_ratio')):.2%} | "
        f"{before.get('item_id')}"
    )
    print(
        "  After:  "
        f"{after.get('acquisition_datetime')} | "
        f"cloud={float(after.get('cloud_cover_percent')):.2f}% | "
        f"overlap={float(after.get('aoi_overlap_ratio')):.2%} | "
        f"{after.get('item_id')}"
    )
    print(
        "  Calendar-day distance: "
        f"{score.get('calendar_day_distance')} days"
    )
    print(
        f"  Pair score: {float(score.get('total_score')):.4f}"
    )
    print(f"  Selection written: {output_path}")

def get_required_environment_variable(name: str) -> str:
    """Return a required environment variable without logging its value.

    Args:
        name: Environment variable name.

    Returns:
        Non-empty environment variable value.

    Raises:
        ConfigurationError: If the variable is missing or blank.
    """
    value = os.environ.get(name, "").strip()

    if not value:
        raise ConfigurationError(
            f"Required environment variable '{name}' is not set."
        )

    return value


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split an S3 URI into bucket and object key.

    Args:
        uri: URI formatted as ``s3://bucket/object-key``.

    Returns:
        Bucket name and object key.

    Raises:
        ValueError: If the URI is malformed.
    """
    if not uri.startswith("s3://"):
        raise ValueError(
            f"Expected an s3:// URI; received {uri!r}."
        )

    remainder = uri[5:]
    bucket, separator, object_key = remainder.partition("/")

    if not separator or not bucket or not object_key:
        raise ValueError(
            f"Malformed S3 URI: {uri!r}"
        )

    return bucket, object_key


def create_cdse_s3_client(
    endpoint_url: str,
    access_key: str,
    secret_key: str,
    timeout_seconds: float,
    retry_count: int,
) -> Any:
    """Create a Boto3 S3 client for Copernicus Data Space.

    Args:
        endpoint_url: CDSE S3-compatible endpoint.
        access_key: Generated CDSE S3 access key.
        secret_key: Generated CDSE S3 secret key.
        timeout_seconds: Connection and read timeout.
        retry_count: Maximum SDK retry attempts.

    Returns:
        Configured Boto3 S3 client.
    """
    boto_config = BotoConfig(
        connect_timeout=timeout_seconds,
        read_timeout=timeout_seconds,
        retries={
            "max_attempts": retry_count,
            "mode": "standard",
        },
        s3={
            "addressing_style": "path",
        },
    )

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="default",
        config=boto_config,
    )


def get_scene_asset_s3_uri(
    scene: Mapping[str, Any],
    asset_key: str,
) -> str:
    """Return the S3 URI for a required scene asset.

    Args:
        scene: Selected scene metadata.
        asset_key: Required STAC asset key.

    Returns:
        S3 URI for the asset.

    Raises:
        StacResponseError: If the asset or URI is missing.
    """
    assets = scene.get("assets")

    if not isinstance(assets, Mapping):
        raise StacResponseError(
            f"Scene {scene.get('item_id')} has no valid assets mapping."
        )

    asset = assets.get(asset_key)

    if not isinstance(asset, Mapping):
        raise StacResponseError(
            f"Scene {scene.get('item_id')} does not contain "
            f"required asset '{asset_key}'."
        )

    href = asset.get("href")

    if not isinstance(href, str) or not href.startswith("s3://"):
        raise StacResponseError(
            f"Asset '{asset_key}' for scene "
            f"{scene.get('item_id')} has no valid S3 URI."
        )

    return href


def serialize_s3_head_response(
    asset_key: str,
    s3_uri: str,
    response: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert an S3 HEAD response into JSON-safe metadata."""
    content_length = response.get("ContentLength")

    if not isinstance(content_length, int) or content_length <= 0:
        raise StacResponseError(
            f"S3 object for '{asset_key}' has an invalid content length."
        )

    last_modified = response.get("LastModified")

    return {
        "asset_key": asset_key,
        "s3_uri": s3_uri,
        "content_length_bytes": content_length,
        "etag": str(response.get("ETag", "")).strip('"'),
        "content_type": response.get("ContentType"),
        "last_modified_utc": (
            last_modified.isoformat()
            if isinstance(last_modified, datetime)
            else None
        ),
    }


def download_s3_object_atomic(
    s3_client: Any,
    bucket: str,
    object_key: str,
    destination: Path,
    expected_size_bytes: int,
) -> str:
    """Download an S3 object atomically and return its SHA-256 checksum.

    Args:
        s3_client: Configured Boto3 S3 client.
        bucket: Source bucket name.
        object_key: Source object key.
        destination: Local destination path.
        expected_size_bytes: Size reported by S3 HEAD.

    Returns:
        SHA-256 checksum of the completed local file.

    Raises:
        StacResponseError: If the downloaded size is incorrect.
    """
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = destination.with_suffix(
        f"{destination.suffix}.part"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    s3_client.download_file(
        bucket,
        object_key,
        str(temporary_path),
    )

    downloaded_size = temporary_path.stat().st_size

    if downloaded_size != expected_size_bytes:
        temporary_path.unlink(missing_ok=True)

        raise StacResponseError(
            "Downloaded S3 object size mismatch: "
            f"expected {expected_size_bytes}, received {downloaded_size}."
        )

    temporary_path.replace(destination)

    return calculate_file_sha256(destination)


def verify_cdse_s3_assets(
    selection_path: Path,
    endpoint_url: str,
    raw_directory: Path,
    timeout_seconds: float,
    retry_count: int,
) -> dict[str, Any]:
    """Verify access to selected Sentinel-2 assets through CDSE S3.

    The function issues metadata-only HEAD requests for all required model
    inputs and downloads only each scene's small ``manifest.safe`` file.

    Args:
        selection_path: Selected bi-temporal pair JSON.
        endpoint_url: CDSE S3 endpoint.
        raw_directory: Root directory for immutable source data.
        timeout_seconds: S3 connection and read timeout.
        retry_count: Maximum SDK retry attempts.

    Returns:
        JSON-serializable verification report.
    """
    access_key = get_required_environment_variable(
        "CDSE_S3_ACCESS_KEY"
    )
    secret_key = get_required_environment_variable(
        "CDSE_S3_SECRET_KEY"
    )

    selection = load_json_mapping(selection_path)
    selected_pair = require_mapping(
        selection.get("selected_pair"),
        "selection.selected_pair",
    )

    required_assets = (
        "B02_10m",
        "B03_10m",
        "B04_10m",
        "B08_10m",
        "B11_20m",
        "B12_20m",
        "SCL_20m",
        "safe_manifest",
    )

    s3_client = create_cdse_s3_client(
        endpoint_url=endpoint_url,
        access_key=access_key,
        secret_key=secret_key,
        timeout_seconds=timeout_seconds,
        retry_count=retry_count,
    )

    period_reports: dict[str, Any] = {}

    try:
        for period in ("before", "after"):
            scene = require_mapping(
                selected_pair.get(period),
                f"selection.selected_pair.{period}",
            )

            item_id = scene.get("item_id")

            if not isinstance(item_id, str) or not item_id.strip():
                raise StacResponseError(
                    f"Selected '{period}' scene has no valid item ID."
                )

            asset_reports: list[dict[str, Any]] = []

            for asset_key in required_assets:
                s3_uri = get_scene_asset_s3_uri(
                    scene=scene,
                    asset_key=asset_key,
                )

                bucket, object_key = parse_s3_uri(s3_uri)

                head_response = s3_client.head_object(
                    Bucket=bucket,
                    Key=object_key,
                )

                asset_report = serialize_s3_head_response(
                    asset_key=asset_key,
                    s3_uri=s3_uri,
                    response=head_response,
                )

                if asset_key == "safe_manifest":
                    destination = (
                        raw_directory
                        / period
                        / item_id
                        / "manifest.safe"
                    )

                    checksum = download_s3_object_atomic(
                        s3_client=s3_client,
                        bucket=bucket,
                        object_key=object_key,
                        destination=destination,
                        expected_size_bytes=asset_report[
                            "content_length_bytes"
                        ],
                    )

                    asset_report["local_path"] = str(destination)
                    asset_report["sha256"] = checksum

                asset_reports.append(asset_report)

            period_reports[period] = {
                "item_id": item_id,
                "mgrs_tile": scene.get("mgrs_tile"),
                "verified_asset_count": len(asset_reports),
                "assets": asset_reports,
            }

    except (BotoCoreError, ClientError) as error:
        raise StacResponseError(
            "CDSE S3 access failed. Verify the credentials, their "
            "expiration date, endpoint connectivity, and selected paths. "
            f"SDK error: {error}"
        ) from error

    return {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "success",
        "endpoint_url": endpoint_url,
        "source_selection": str(selection_path),
        "source_selection_sha256": calculate_file_sha256(
            selection_path
        ),
        "required_assets": list(required_assets),
        "periods": period_reports,
    }


def print_s3_verification_summary(
    report: Mapping[str, Any],
    output_path: Path,
) -> None:
    """Print a concise S3 verification result."""
    print("CDSE S3 access verification succeeded")
    print(f"  Endpoint: {report.get('endpoint_url')}")

    periods = require_mapping(
        report.get("periods"),
        "verification.periods",
    )

    for period in ("before", "after"):
        period_report = require_mapping(
            periods.get(period),
            f"verification.periods.{period}",
        )

        print(
            f"  {period.title()}: "
            f"{period_report.get('verified_asset_count')} assets verified"
        )

        assets = period_report.get("assets")

        if isinstance(assets, Sequence):
            for asset in assets:
                if (
                    isinstance(asset, Mapping)
                    and asset.get("asset_key") == "safe_manifest"
                ):
                    print(
                        "    Manifest: "
                        f"{asset.get('local_path')}"
                    )
                    print(
                        "    SHA-256: "
                        f"{asset.get('sha256')}"
                    )

    print(f"  Verification report: {output_path}")

def infer_asset_resolution_meters(asset_key: str) -> int:
    """Infer spatial resolution from a Sentinel-2 STAC asset key.

    Args:
        asset_key: Asset key such as ``B02_10m`` or ``SCL_20m``.

    Returns:
        Resolution in metres.

    Raises:
        ValueError: If the resolution suffix is missing.
    """
    resolution_match = re.search(
        r"_(10|20|60)m$",
        asset_key,
    )

    if resolution_match is None:
        raise ValueError(
            f"Could not infer resolution from asset key: {asset_key}"
        )

    return int(resolution_match.group(1))


def get_s3_object_filename(s3_uri: str) -> str:
    """Return the filename component of an S3 object URI."""
    _, object_key = parse_s3_uri(s3_uri)
    filename = Path(object_key).name

    if not filename:
        raise ValueError(
            f"S3 URI does not contain a filename: {s3_uri}"
        )

    return filename


def download_s3_object_resumable(
    s3_client: Any,
    bucket: str,
    object_key: str,
    destination: Path,
    expected_size_bytes: int,
    chunk_size_bytes: int,
    force: bool,
) -> tuple[str, str]:
    """Download an S3 object with partial-file resume support.

    Completed files are written atomically. Interrupted downloads remain as
    ``.part`` files and continue from the previous byte position when the
    command is run again.

    Args:
        s3_client: Configured Boto3 S3 client.
        bucket: Source S3 bucket.
        object_key: Source object key.
        destination: Completed local file path.
        expected_size_bytes: Object size returned by S3 HEAD.
        chunk_size_bytes: Streaming read chunk size.
        force: Replace an existing completed or partial file.

    Returns:
        Download status and completed-file SHA-256 checksum.

    Raises:
        ValueError: If size or chunk configuration is invalid.
        StacResponseError: If the completed size does not match S3 metadata.
    """
    if expected_size_bytes <= 0:
        raise ValueError(
            "Expected S3 object size must be greater than zero."
        )

    if chunk_size_bytes <= 0:
        raise ValueError(
            "Download chunk size must be greater than zero."
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    partial_path = destination.with_suffix(
        f"{destination.suffix}.part"
    )

    if force:
        destination.unlink(missing_ok=True)
        partial_path.unlink(missing_ok=True)

    if destination.exists():
        completed_size = destination.stat().st_size

        if completed_size != expected_size_bytes:
            raise StacResponseError(
                f"Existing file has an unexpected size: {destination}. "
                "Use --force to replace it."
            )

        return (
            "already_present",
            calculate_file_sha256(destination),
        )

    existing_partial_size = (
        partial_path.stat().st_size
        if partial_path.exists()
        else 0
    )

    if existing_partial_size > expected_size_bytes:
        raise StacResponseError(
            f"Partial file is larger than the source object: {partial_path}. "
            "Use --force to restart."
        )

    if existing_partial_size == expected_size_bytes:
        partial_path.replace(destination)

        return (
            "recovered_complete_partial",
            calculate_file_sha256(destination),
        )

    request_arguments: dict[str, Any] = {
        "Bucket": bucket,
        "Key": object_key,
    }

    if existing_partial_size > 0:
        request_arguments["Range"] = (
            f"bytes={existing_partial_size}-"
        )

    response = s3_client.get_object(
        **request_arguments
    )

    response_body = response.get("Body")

    if response_body is None:
        raise StacResponseError(
            f"S3 returned no response body for object: {object_key}"
        )

    file_mode = (
        "ab"
        if existing_partial_size > 0
        else "wb"
    )

    try:
        with partial_path.open(file_mode) as output_file:
            for chunk in response_body.iter_chunks(
                chunk_size=chunk_size_bytes
            ):
                if chunk:
                    output_file.write(chunk)
    finally:
        response_body.close()

    final_size = partial_path.stat().st_size

    if final_size != expected_size_bytes:
        raise StacResponseError(
            "Incomplete S3 download. "
            f"Expected {expected_size_bytes} bytes but received "
            f"{final_size} bytes for {partial_path}. "
            "Run the command again to resume."
        )

    partial_path.replace(destination)

    status = (
        "resumed"
        if existing_partial_size > 0
        else "downloaded"
    )

    return status, calculate_file_sha256(destination)


def download_selected_band_assets(
    selection_path: Path,
    config_path: Path,
    endpoint_url: str,
    raw_directory: Path,
    chunk_size_bytes: int,
    timeout_seconds: float,
    retry_count: int,
    force: bool,
) -> dict[str, Any]:
    """Download selected Sentinel-2 source bands from CDSE S3.

    Args:
        selection_path: Selected scene-pair JSON.
        config_path: GeoWatch data configuration.
        endpoint_url: CDSE S3-compatible endpoint.
        raw_directory: Immutable raw-data directory.
        chunk_size_bytes: Streaming download chunk size.
        timeout_seconds: S3 request timeout.
        retry_count: SDK retry attempts.
        force: Replace existing files.

    Returns:
        JSON-serializable download report.
    """
    access_key = get_required_environment_variable(
        "CDSE_S3_ACCESS_KEY"
    )
    secret_key = get_required_environment_variable(
        "CDSE_S3_SECRET_KEY"
    )

    selection = load_json_mapping(selection_path)
    config = load_yaml_config(config_path)

    acquisition = require_mapping(
        config.get("acquisition"),
        "acquisition",
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

    required_assets = tuple(
        str(asset_key)
        for asset_key in configured_assets
    )

    selected_pair = require_mapping(
        selection.get("selected_pair"),
        "selection.selected_pair",
    )

    s3_client = create_cdse_s3_client(
        endpoint_url=endpoint_url,
        access_key=access_key,
        secret_key=secret_key,
        timeout_seconds=timeout_seconds,
        retry_count=retry_count,
    )

    period_reports: dict[str, Any] = {}
    total_size_bytes = 0

    try:
        for period in ("before", "after"):
            scene = require_mapping(
                selected_pair.get(period),
                f"selection.selected_pair.{period}",
            )

            item_id = scene.get("item_id")
            acquisition_datetime = scene.get(
                "acquisition_datetime"
            )

            if not isinstance(item_id, str) or not item_id.strip():
                raise StacResponseError(
                    f"Selected {period} scene has no item ID."
                )

            asset_reports: list[dict[str, Any]] = []

            for asset_key in required_assets:
                s3_uri = get_scene_asset_s3_uri(
                    scene=scene,
                    asset_key=asset_key,
                )

                bucket, object_key = parse_s3_uri(s3_uri)

                head_response = s3_client.head_object(
                    Bucket=bucket,
                    Key=object_key,
                )

                object_size = head_response.get(
                    "ContentLength"
                )

                if not isinstance(object_size, int) or object_size <= 0:
                    raise StacResponseError(
                        f"Invalid size for asset {asset_key}."
                    )

                filename = get_s3_object_filename(s3_uri)

                destination = (
                    raw_directory
                    / period
                    / item_id
                    / filename
                )

                status, checksum = (
                    download_s3_object_resumable(
                        s3_client=s3_client,
                        bucket=bucket,
                        object_key=object_key,
                        destination=destination,
                        expected_size_bytes=object_size,
                        chunk_size_bytes=chunk_size_bytes,
                        force=force,
                    )
                )

                resolution_meters = (
                    infer_asset_resolution_meters(asset_key)
                )

                asset_reports.append(
                    {
                        "asset_key": asset_key,
                        "resolution_meters": resolution_meters,
                        "s3_uri": s3_uri,
                        "local_path": str(destination),
                        "content_length_bytes": object_size,
                        "etag": str(
                            head_response.get("ETag", "")
                        ).strip('"'),
                        "sha256": checksum,
                        "download_status": status,
                    }
                )

                total_size_bytes += object_size

                print(
                    f"  [{period}] {asset_key}: "
                    f"{status} -> {destination}"
                )

            period_reports[period] = {
                "item_id": item_id,
                "acquisition_datetime": acquisition_datetime,
                "mgrs_tile": scene.get("mgrs_tile"),
                "asset_count": len(asset_reports),
                "assets": asset_reports,
            }

    except (BotoCoreError, ClientError) as error:
        raise StacResponseError(
            f"CDSE S3 band download failed: {error}"
        ) from error

    return {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "success",
        "endpoint_url": endpoint_url,
        "source_selection": str(selection_path),
        "source_selection_sha256": calculate_file_sha256(
            selection_path
        ),
        "required_assets": list(required_assets),
        "total_asset_count": sum(
            period["asset_count"]
            for period in period_reports.values()
        ),
        "total_size_bytes": total_size_bytes,
        "periods": period_reports,
    }


def update_manifest_from_download_report(
    report: Mapping[str, Any],
    config_path: Path,
    manifest_path: Path,
) -> None:
    """Add or update downloaded raw-band records in the dataset manifest.

    Args:
        report: Completed band-download report.
        config_path: GeoWatch YAML configuration.
        manifest_path: Dataset manifest CSV path.
    """
    config = load_yaml_config(config_path)

    project = require_mapping(
        config.get("project"),
        "project",
    )
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

    if not dataset_version:
        raise ConfigurationError(
            "project.dataset_version cannot be empty."
        )

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

    if manifest_path.exists():
        with manifest_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as manifest_file:
            reader = csv.DictReader(manifest_file)

            for row in reader:
                record_id = row.get("record_id", "").strip()

                if record_id:
                    existing_records[record_id] = {
                        field: row.get(field, "")
                        for field in fieldnames
                    }

    periods = require_mapping(
        report.get("periods"),
        "download_report.periods",
    )

    created_at_utc = datetime.now(
        timezone.utc
    ).isoformat()

    for period in ("before", "after"):
        period_report = require_mapping(
            periods.get(period),
            f"download_report.periods.{period}",
        )

        item_id = str(
            period_report.get("item_id", "")
        )
        acquisition_datetime = str(
            period_report.get("acquisition_datetime", "")
        )

        acquisition_date = (
            acquisition_datetime[:10]
            if len(acquisition_datetime) >= 10
            else ""
        )

        assets = period_report.get("assets")

        if not isinstance(assets, Sequence):
            raise ValueError(
                f"Download report assets are invalid for {period}."
            )

        for asset_value in assets:
            asset = require_mapping(
                asset_value,
                f"download_report.periods.{period}.asset",
            )

            asset_key = str(asset.get("asset_key", ""))
            local_path = Path(
                str(asset.get("local_path", ""))
            )

            record_id = (
                f"raw-{period}-{item_id}-{asset_key}"
            )

            existing_records[record_id] = {
                "dataset_version": dataset_version,
                "record_id": record_id,
                "record_type": "raw_satellite_band",
                "source": "copernicus_data_space_s3",
                "product_id": item_id,
                "acquisition_date": acquisition_date,
                "aoi_name": aoi_name,
                "crs": "",
                "resolution_meters": str(
                    asset.get("resolution_meters", "")
                ),
                "relative_path": local_path.as_posix(),
                "sha256": str(asset.get("sha256", "")),
                "parent_record_ids": (
                    "selected_scene_pair"
                ),
                "processing_status": "downloaded",
                "created_at_utc": created_at_utc,
            }

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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

        for record_id in sorted(existing_records):
            writer.writerow(existing_records[record_id])

    temporary_path.replace(manifest_path)


def format_file_size(size_bytes: int) -> str:
    """Format a byte count as a human-readable value."""
    size = float(size_bytes)

    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.2f} {unit}"

        size /= 1024.0

    return f"{size_bytes} B"


def print_band_download_summary(
    report: Mapping[str, Any],
    report_path: Path,
    manifest_path: Path,
) -> None:
    """Print a concise raw-band download summary."""
    print("Sentinel-2 band download succeeded")
    print(
        f"  Assets: {report.get('total_asset_count')}"
    )
    print(
        "  Total source size: "
        f"{format_file_size(int(report.get('total_size_bytes', 0)))}"
    )
    print(f"  Download report: {report_path}")
    print(f"  Dataset manifest: {manifest_path}")

def affine_to_list(transform: Any) -> list[float]:
    """Serialize the six meaningful coefficients of an affine transform."""
    return [
        float(transform.a),
        float(transform.b),
        float(transform.c),
        float(transform.d),
        float(transform.e),
        float(transform.f),
    ]


def numeric_sequences_match(
    first: Sequence[float],
    second: Sequence[float],
    absolute_tolerance: float = 1e-6,
) -> bool:
    """Return whether two numeric sequences match within a tolerance."""
    if len(first) != len(second):
        return False

    return all(
        math.isclose(
            float(left),
            float(right),
            rel_tol=0.0,
            abs_tol=absolute_tolerance,
        )
        for left, right in zip(first, second)
    )


def inspect_raw_raster(
    asset: Mapping[str, Any],
    target_epsg: int,
) -> tuple[dict[str, Any], list[str]]:
    """Inspect one downloaded Sentinel-2 raster.

    Args:
        asset: Asset record from the band-download report.
        target_epsg: Expected projected CRS.

    Returns:
        Raster metadata report and validation errors.
    """
    errors: list[str] = []

    asset_key = str(asset.get("asset_key", "")).strip()
    local_path = Path(str(asset.get("local_path", "")))
    expected_checksum = str(asset.get("sha256", "")).strip()

    try:
        expected_resolution = int(
            asset.get(
                "resolution_meters",
                infer_asset_resolution_meters(asset_key),
            )
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid expected resolution for asset {asset_key}."
        ) from error

    raster_report: dict[str, Any] = {
        "asset_key": asset_key,
        "local_path": str(local_path),
        "expected_resolution_meters": expected_resolution,
        "exists": local_path.is_file(),
    }

    if not local_path.is_file():
        errors.append(
            f"{asset_key}: file does not exist: {local_path}"
        )
        return raster_report, errors

    actual_checksum = calculate_file_sha256(local_path)

    raster_report["size_bytes"] = local_path.stat().st_size
    raster_report["sha256"] = actual_checksum
    raster_report["checksum_matches_report"] = (
        actual_checksum == expected_checksum
    )

    if actual_checksum != expected_checksum:
        errors.append(
            f"{asset_key}: SHA-256 does not match the download report."
        )

    try:
        with rasterio.open(local_path) as dataset:
            crs_epsg = (
                dataset.crs.to_epsg()
                if dataset.crs is not None
                else None
            )

            pixel_width = abs(float(dataset.transform.a))
            pixel_height = abs(float(dataset.transform.e))

            sample_height = min(256, dataset.height)
            sample_width = min(256, dataset.width)

            sample = dataset.read(
                1,
                out_shape=(
                    sample_height,
                    sample_width,
                ),
                masked=True,
                resampling=Resampling.nearest,
            )

            valid_count = int(sample.count())
            sample_size = int(sample.size)

            if valid_count > 0:
                sample_min = float(sample.min())
                sample_max = float(sample.max())
            else:
                sample_min = None
                sample_max = None

            raster_report.update(
                {
                    "driver": dataset.driver,
                    "crs": (
                        dataset.crs.to_string()
                        if dataset.crs is not None
                        else None
                    ),
                    "epsg": crs_epsg,
                    "width": dataset.width,
                    "height": dataset.height,
                    "band_count": dataset.count,
                    "dtype": dataset.dtypes[0],
                    "nodata": dataset.nodata,
                    "pixel_width_meters": pixel_width,
                    "pixel_height_meters": pixel_height,
                    "transform": affine_to_list(
                        dataset.transform
                    ),
                    "bounds": [
                        float(dataset.bounds.left),
                        float(dataset.bounds.bottom),
                        float(dataset.bounds.right),
                        float(dataset.bounds.top),
                    ],
                    "sample_valid_fraction": (
                        valid_count / sample_size
                        if sample_size > 0
                        else 0.0
                    ),
                    "sample_min": sample_min,
                    "sample_max": sample_max,
                }
            )

            if dataset.driver != "JP2OpenJPEG":
                errors.append(
                    f"{asset_key}: unexpected raster driver "
                    f"{dataset.driver!r}; expected 'JP2OpenJPEG'."
                )

            if dataset.count != 1:
                errors.append(
                    f"{asset_key}: expected one raster band, "
                    f"found {dataset.count}."
                )

            if crs_epsg != target_epsg:
                errors.append(
                    f"{asset_key}: expected EPSG:{target_epsg}, "
                    f"found {crs_epsg}."
                )

            if not math.isclose(
                pixel_width,
                expected_resolution,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                errors.append(
                    f"{asset_key}: pixel width is {pixel_width} m, "
                    f"expected {expected_resolution} m."
                )

            if not math.isclose(
                pixel_height,
                expected_resolution,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                errors.append(
                    f"{asset_key}: pixel height is {pixel_height} m, "
                    f"expected {expected_resolution} m."
                )

            if dataset.width <= 0 or dataset.height <= 0:
                errors.append(
                    f"{asset_key}: raster dimensions are invalid."
                )

            if valid_count == 0:
                errors.append(
                    f"{asset_key}: diagnostic sample contains "
                    "no valid pixels."
                )

    except RasterioIOError as error:
        errors.append(
            f"{asset_key}: Rasterio could not open the file: {error}"
        )

    return raster_report, errors


def validate_raster_group_alignment(
    rasters: Sequence[Mapping[str, Any]],
    group_name: str,
) -> list[str]:
    """Validate that all rasters in a group share one pixel grid."""
    errors: list[str] = []

    if not rasters:
        return [f"{group_name}: no rasters were available."]

    reference = rasters[0]

    required_fields = (
        "width",
        "height",
        "epsg",
        "transform",
        "bounds",
    )

    if any(field not in reference for field in required_fields):
        return [
            f"{group_name}: reference raster metadata is incomplete."
        ]

    for raster in rasters[1:]:
        asset_key = raster.get("asset_key")

        if raster.get("width") != reference.get("width"):
            errors.append(
                f"{group_name}/{asset_key}: width does not match."
            )

        if raster.get("height") != reference.get("height"):
            errors.append(
                f"{group_name}/{asset_key}: height does not match."
            )

        if raster.get("epsg") != reference.get("epsg"):
            errors.append(
                f"{group_name}/{asset_key}: CRS does not match."
            )

        first_transform = reference.get("transform")
        second_transform = raster.get("transform")

        if (
            not isinstance(first_transform, Sequence)
            or not isinstance(second_transform, Sequence)
            or not numeric_sequences_match(
                first_transform,
                second_transform,
            )
        ):
            errors.append(
                f"{group_name}/{asset_key}: affine transform "
                "does not match."
            )

        first_bounds = reference.get("bounds")
        second_bounds = raster.get("bounds")

        if (
            not isinstance(first_bounds, Sequence)
            or not isinstance(second_bounds, Sequence)
            or not numeric_sequences_match(
                first_bounds,
                second_bounds,
            )
        ):
            errors.append(
                f"{group_name}/{asset_key}: raster bounds "
                "do not match."
            )

    return errors


def validate_raw_band_download(
    download_report_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    """Validate downloaded JP2 files and their spatial alignment."""
    download_report = load_json_mapping(
        download_report_path
    )
    config = load_yaml_config(config_path)

    processing = require_mapping(
        config.get("processing"),
        "processing",
    )
    acquisition = require_mapping(
        config.get("acquisition"),
        "acquisition",
    )

    try:
        target_epsg = int(processing.get("target_epsg"))
    except (TypeError, ValueError) as error:
        raise ConfigurationError(
            "processing.target_epsg must be an integer."
        ) from error

    configured_assets = acquisition.get(
        "required_band_assets"
    )

    if (
        not isinstance(configured_assets, Sequence)
        or isinstance(configured_assets, (str, bytes))
    ):
        raise ConfigurationError(
            "acquisition.required_band_assets must be a list."
        )

    required_asset_keys = {
        str(asset_key)
        for asset_key in configured_assets
    }

    periods = require_mapping(
        download_report.get("periods"),
        "download_report.periods",
    )

    errors: list[str] = []
    period_reports: dict[str, Any] = {}

    for period in ("before", "after"):
        period_value = require_mapping(
            periods.get(period),
            f"download_report.periods.{period}",
        )

        assets = period_value.get("assets")

        if not isinstance(assets, Sequence):
            raise ValueError(
                f"Download report assets are invalid for {period}."
            )

        discovered_asset_keys = {
            str(asset.get("asset_key"))
            for asset in assets
            if isinstance(asset, Mapping)
        }

        missing_assets = sorted(
            required_asset_keys - discovered_asset_keys
        )
        unexpected_assets = sorted(
            discovered_asset_keys - required_asset_keys
        )

        if missing_assets:
            errors.append(
                f"{period}: missing assets: {missing_assets}"
            )

        if unexpected_assets:
            errors.append(
                f"{period}: unexpected assets: {unexpected_assets}"
            )

        raster_reports: list[dict[str, Any]] = []

        for asset_value in assets:
            asset = require_mapping(
                asset_value,
                f"download_report.{period}.asset",
            )

            raster_report, raster_errors = inspect_raw_raster(
                asset=asset,
                target_epsg=target_epsg,
            )

            raster_reports.append(raster_report)
            errors.extend(
                f"{period}: {message}"
                for message in raster_errors
            )

        ten_meter_rasters = [
            raster
            for raster in raster_reports
            if raster.get(
                "expected_resolution_meters"
            ) == 10
        ]
        twenty_meter_rasters = [
            raster
            for raster in raster_reports
            if raster.get(
                "expected_resolution_meters"
            ) == 20
        ]

        errors.extend(
            validate_raster_group_alignment(
                ten_meter_rasters,
                f"{period}/10m",
            )
        )
        errors.extend(
            validate_raster_group_alignment(
                twenty_meter_rasters,
                f"{period}/20m",
            )
        )

        period_reports[period] = {
            "item_id": period_value.get("item_id"),
            "mgrs_tile": period_value.get("mgrs_tile"),
            "raster_count": len(raster_reports),
            "rasters": raster_reports,
        }

    for resolution in (10, 20):
        before_rasters = [
            raster
            for raster in period_reports["before"]["rasters"]
            if raster.get(
                "expected_resolution_meters"
            ) == resolution
        ]
        after_rasters = [
            raster
            for raster in period_reports["after"]["rasters"]
            if raster.get(
                "expected_resolution_meters"
            ) == resolution
        ]

        if before_rasters and after_rasters:
            errors.extend(
                validate_raster_group_alignment(
                    [
                        before_rasters[0],
                        after_rasters[0],
                    ],
                    f"cross_temporal/{resolution}m",
                )
            )

    return {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": (
            "success"
            if not errors
            else "failed"
        ),
        "source_download_report": str(
            download_report_path
        ),
        "source_download_report_sha256": (
            calculate_file_sha256(download_report_path)
        ),
        "target_epsg": target_epsg,
        "expected_asset_count": (
            len(required_asset_keys) * 2
        ),
        "validated_raster_count": sum(
            period["raster_count"]
            for period in period_reports.values()
        ),
        "error_count": len(errors),
        "errors": errors,
        "periods": period_reports,
    }


def print_raw_validation_summary(
    report: Mapping[str, Any],
    output_path: Path,
) -> None:
    """Print the result of raw Sentinel-2 raster validation."""
    print("Raw Sentinel-2 raster validation completed")
    print(f"  Status: {report.get('status')}")
    print(
        "  Rasters validated: "
        f"{report.get('validated_raster_count')}"
    )
    print(f"  Target CRS: EPSG:{report.get('target_epsg')}")
    print(f"  Errors: {report.get('error_count')}")
    print(f"  Validation report: {output_path}")

    errors = report.get("errors")

    if isinstance(errors, Sequence) and errors:
        for error in errors[:20]:
            print(f"    - {error}")

def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Discover Sentinel-2 L2A scene candidates using the "
            "Copernicus Data Space STAC API."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data_config.yaml"),
        help=(
            "Path to the GeoWatch data configuration. "
            "Default: configs/data_config.yaml"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output JSON path. When omitted, "
            "acquisition.discovery_output is used."
        ),
    )

    parser.add_argument(
        "--mode",
        choices=("discover", "select", "verify-s3", "download-bands", "validate-raw"),
        default="discover",
        help=(
            "Operation to perform: discover scene metadata or select "
            "the best spatially valid bi-temporal pair."
        ),
    )

    parser.add_argument(
        "--catalog",
        type=Path,
        default=None,
        help=(
            "Scene catalogue used by select mode. When omitted, "
            "acquisition.discovery_output is used."
        ),
    )

    parser.add_argument(
        "--selection-output",
        type=Path,
        default=None,
        help=(
            "Output JSON used by select mode. When omitted, "
            "acquisition.scene_selection_output is used."
        ),
    )

    parser.add_argument(
        "--minimum-overlap-ratio",
        type=float,
        default=None,
        help=(
            "Minimum fraction of the AOI that each scene must cover. "
            "When omitted, acquisition.minimum_aoi_overlap_ratio is used."
        ),
    )

    parser.add_argument(
        "--selected-pair",
        type=Path,
        default=None,
        help=(
            "Selected scene-pair JSON used by verify-s3 mode. "
            "Defaults to acquisition.scene_selection_output."
        ),
    )

    parser.add_argument(
        "--s3-verification-output",
        type=Path,
        default=None,
        help=(
            "Output report used by verify-s3 mode. "
            "Defaults to acquisition.s3_verification_output."
        ),
    )
    parser.add_argument(
        "--download-report",
        type=Path,
        default=None,
        help=(
            "Output JSON for download-bands mode. Defaults to "
            "acquisition.band_download_report."
        ),
    )

    parser.add_argument(
        "--chunk-size-mb",
        type=int,
        default=None,
        help=(
            "Streaming download chunk size in MB. Defaults to "
            "acquisition.download_chunk_size_mb."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing completed and partial band files.",
    )
    parser.add_argument(
        "--raw-validation-output",
        type=Path,
        default=None,
        help=(
            "Output JSON for validate-raw mode. Defaults to "
            "acquisition.raw_validation_output."
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
    """Run Sentinel-2 scene discovery.

    Returns:
        Process exit code. Zero indicates success.
    """
    parser = build_argument_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    try:
        settings = load_discovery_settings(
            config_path=args.config,
            output_override=args.output,
        )

        if args.mode == "validate-raw":
            config = load_yaml_config(args.config)

            acquisition = require_mapping(
                config.get("acquisition"),
                "acquisition",
            )

            download_report_path = Path(
                str(
                    acquisition.get(
                        "band_download_report",
                        "data/raw/band_download_report.json",
                    )
                )
            )

            validation_output_path = (
                args.raw_validation_output
                if args.raw_validation_output is not None
                else Path(
                    str(
                        acquisition.get(
                            "raw_validation_output",
                            "data/raw/raw_raster_validation.json",
                        )
                    )
                )
            )

            validation_report = validate_raw_band_download(
                download_report_path=download_report_path,
                config_path=args.config,
            )

            write_json_atomic(
                payload=validation_report,
                output_path=validation_output_path,
            )

            print_raw_validation_summary(
                report=validation_report,
                output_path=validation_output_path,
            )

            if validation_report["status"] != "success":
                raise StacResponseError(
                    "Raw-raster validation failed. "
                    "Review the generated validation report."
                )

            return 0
        if args.mode == "download-bands":
            config = load_yaml_config(args.config)

            acquisition = require_mapping(
                config.get("acquisition"),
                "acquisition",
            )
            paths = require_mapping(
                config.get("paths"),
                "paths",
            )

            selected_pair_path = Path(
                str(
                    acquisition.get(
                        "scene_selection_output",
                        "data/raw/selected_scene_pair.json",
                    )
                )
            )

            download_report_path = (
                args.download_report
                if args.download_report is not None
                else Path(
                    str(
                        acquisition.get(
                            "band_download_report",
                            "data/raw/band_download_report.json",
                        )
                    )
                )
            )

            configured_chunk_mb = int(
                acquisition.get(
                    "download_chunk_size_mb",
                    8,
                )
            )

            chunk_size_mb = (
                args.chunk_size_mb
                if args.chunk_size_mb is not None
                else configured_chunk_mb
            )

            if chunk_size_mb <= 0:
                raise ConfigurationError(
                    "Download chunk size must be greater than zero."
                )

            raw_directory = Path(
                str(paths.get("raw_dir", "data/raw"))
            )
            manifest_path = Path(
                str(
                    paths.get(
                        "manifest_path",
                        "data/manifest.csv",
                    )
                )
            )

            endpoint_url = str(
                acquisition.get(
                    "s3_endpoint_url",
                    "https://eodata.dataspace.copernicus.eu",
                )
            ).rstrip("/")

            report = download_selected_band_assets(
                selection_path=selected_pair_path,
                config_path=args.config,
                endpoint_url=endpoint_url,
                raw_directory=raw_directory,
                chunk_size_bytes=(
                    chunk_size_mb * 1024 * 1024
                ),
                timeout_seconds=settings.timeout_seconds,
                retry_count=settings.retry_count,
                force=args.force,
            )

            write_json_atomic(
                payload=report,
                output_path=download_report_path,
            )

            update_manifest_from_download_report(
                report=report,
                config_path=args.config,
                manifest_path=manifest_path,
            )

            print_band_download_summary(
                report=report,
                report_path=download_report_path,
                manifest_path=manifest_path,
            )

            return 0
        if args.mode == "verify-s3":
            config = load_yaml_config(args.config)
            acquisition = require_mapping(
                config.get("acquisition"),
                "acquisition",
            )

            selected_pair_path = (
                args.selected_pair
                if args.selected_pair is not None
                else Path(
                    str(
                        acquisition.get(
                            "scene_selection_output",
                            "data/raw/selected_scene_pair.json",
                        )
                    )
                )
            )

            verification_output = (
                args.s3_verification_output
                if args.s3_verification_output is not None
                else Path(
                    str(
                        acquisition.get(
                            "s3_verification_output",
                            "data/raw/s3_access_verification.json",
                        )
                    )
                )
            )

            endpoint_url = str(
                acquisition.get(
                    "s3_endpoint_url",
                    "https://eodata.dataspace.copernicus.eu",
                )
            ).rstrip("/")

            report = verify_cdse_s3_assets(
                selection_path=selected_pair_path,
                endpoint_url=endpoint_url,
                raw_directory=Path("data/raw"),
                timeout_seconds=settings.timeout_seconds,
                retry_count=settings.retry_count,
            )

            write_json_atomic(
                payload=report,
                output_path=verification_output,
            )

            print_s3_verification_summary(
                report=report,
                output_path=verification_output,
            )

            return 0
        if args.mode == "select":
            config = load_yaml_config(args.config)
            acquisition = require_mapping(
                config.get("acquisition"),
                "acquisition",
            )

            catalog_path = (
                args.catalog
                if args.catalog is not None
                else settings.output_path
            )

            configured_selection_output = Path(
                str(
                    acquisition.get(
                        "scene_selection_output",
                        "data/raw/selected_scene_pair.json",
                    )
                )
            )

            selection_output = (
                args.selection_output
                if args.selection_output is not None
                else configured_selection_output
            )

            configured_overlap_ratio = float(
                acquisition.get(
                    "minimum_aoi_overlap_ratio",
                    0.99,
                )
            )

            minimum_overlap_ratio = (
                args.minimum_overlap_ratio
                if args.minimum_overlap_ratio is not None
                else configured_overlap_ratio
            )

            selection = select_scene_pair(
                catalog_path=catalog_path,
                config_path=args.config,
                minimum_overlap_ratio=minimum_overlap_ratio,
            )

            write_json_atomic(
                payload=selection,
                output_path=selection_output,
            )

            print_scene_selection_summary(
                selection=selection,
                output_path=selection_output,
            )

            return 0

        with create_http_session(
            retry_count=settings.retry_count
        ) as session:
            catalog = build_catalog(
                session=session,
                settings=settings,
            )

        write_json_atomic(
            payload=catalog,
            output_path=settings.output_path,
        )

        print_discovery_summary(
            catalog=catalog,
            output_path=settings.output_path,
        )

        return 0

    except (
        FileNotFoundError,
        PermissionError,
        ConfigurationError,
        StacResponseError,
        requests.RequestException,
        yaml.YAMLError,
    ) as error:
        LOGGER.error("%s", error)
        return 1

    except Exception:
        LOGGER.exception(
            "Unexpected Sentinel-2 discovery failure."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())




