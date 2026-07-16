"""Download, verify, extract and document the OSCD benchmark dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class AcquisitionError(RuntimeError):
    """Raised when OSCD acquisition or validation fails."""


@dataclass(frozen=True)
class ArchiveSpec:
    """One immutable OSCD source archive."""

    key: str
    filename: str
    url: str
    expected_md5: str
    extracted_directory: str


ARCHIVES = (
    ArchiveSpec(
        key="images",
        filename=(
            "Onera Satellite Change Detection dataset - Images.zip"
        ),
        url=(
            "https://partage.imt.fr/index.php/s/"
            "gKRaWgRnLMfwMGo/download"
        ),
        expected_md5="c50d4a2941da64e03a47ac4dec63d915",
        extracted_directory=(
            "Onera Satellite Change Detection dataset - Images"
        ),
    ),
    ArchiveSpec(
        key="train_labels",
        filename=(
            "Onera Satellite Change Detection dataset - "
            "Train Labels.zip"
        ),
        url=(
            "https://partage.mines-telecom.fr/index.php/s/"
            "2D6n03k58ygBSpu/download"
        ),
        expected_md5="4d2965af8170c705ebad3d6ee71b6990",
        extracted_directory=(
            "Onera Satellite Change Detection dataset - Train Labels"
        ),
    ),
    ArchiveSpec(
        key="test_labels",
        filename=(
            "Onera Satellite Change Detection dataset - "
            "Test Labels.zip"
        ),
        url=(
            "https://partage.imt.fr/index.php/s/"
            "gpStKn4Mpgfnr63/download"
        ),
        expected_md5="8177d437793c522653c442aa4e66c617",
        extracted_directory=(
            "Onera Satellite Change Detection dataset - Test Labels"
        ),
    ),
)


def require_mapping(
    value: object,
    context: str,
) -> Mapping[str, Any]:
    """Return a mapping or raise a clear configuration error."""
    if not isinstance(value, Mapping):
        raise AcquisitionError(
            f"Configuration value '{context}' must be a mapping."
        )

    return value


def load_config(path: Path) -> dict[str, Any]:
    """Load the GeoWatch YAML configuration."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Configuration file does not exist: {path}"
        )

    with path.open("r", encoding="utf-8-sig") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise AcquisitionError(
            "Configuration root must be a YAML mapping."
        )

    return config


def calculate_md5(path: Path) -> str:
    """Calculate a file's MD5 checksum."""
    digest = hashlib.md5()

    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def calculate_sha256(path: Path) -> str:
    """Calculate a file's SHA-256 checksum."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def create_http_session() -> requests.Session:
    """Create an HTTP session with bounded retries."""
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=2,
        pool_maxsize=2,
    )

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "GeoWatch-OSCD-Acquisition/1.0",
        }
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


def download_archive(
    session: requests.Session,
    spec: ArchiveSpec,
    destination: Path,
    force: bool,
) -> str:
    """Download one archive atomically with partial-download resume."""
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

    if destination.is_file():
        actual_md5 = calculate_md5(destination)

        if actual_md5 == spec.expected_md5:
            return "already_present"

        destination.unlink()

    existing_size = (
        partial_path.stat().st_size
        if partial_path.exists()
        else 0
    )

    headers: dict[str, str] = {}

    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"

    response = session.get(
        spec.url,
        headers=headers,
        stream=True,
        timeout=(30, 180),
        allow_redirects=True,
    )

    if response.status_code not in {200, 206}:
        raise AcquisitionError(
            f"Download failed for {spec.filename}: "
            f"HTTP {response.status_code}"
        )

    if existing_size > 0 and response.status_code != 206:
        partial_path.unlink(missing_ok=True)
        existing_size = 0

    file_mode = "ab" if existing_size > 0 else "wb"

    with partial_path.open(file_mode) as output:
        for chunk in response.iter_content(
            chunk_size=8 * 1024 * 1024
        ):
            if chunk:
                output.write(chunk)

    partial_path.replace(destination)

    actual_md5 = calculate_md5(destination)

    if actual_md5 != spec.expected_md5:
        destination.unlink(missing_ok=True)

        raise AcquisitionError(
            f"MD5 mismatch for {spec.filename}. "
            f"Expected {spec.expected_md5}, received {actual_md5}."
        )

    return "resumed" if existing_size > 0 else "downloaded"


def safely_extract_zip(
    archive_path: Path,
    destination: Path,
    force: bool,
    expected_directory: Path,
) -> str:
    """Extract a ZIP archive while preventing path traversal."""
    if expected_directory.is_dir() and not force:
        return "already_extracted"

    if force and expected_directory.exists():
        shutil.rmtree(expected_directory)

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination_root = destination.resolve()

    with zipfile.ZipFile(archive_path) as archive:
        invalid_members: list[str] = []

        for member in archive.infolist():
            member_path = (
                destination / member.filename
            ).resolve()

            if (
                member_path != destination_root
                and destination_root
                not in member_path.parents
            ):
                invalid_members.append(member.filename)

        if invalid_members:
            raise AcquisitionError(
                "Archive contains unsafe paths: "
                f"{invalid_members[:10]}"
            )

        archive.extractall(destination)

    if not expected_directory.is_dir():
        raise AcquisitionError(
            "Expected extracted directory was not created: "
            f"{expected_directory}"
        )

    return "extracted"


def find_tif_files(directory: Path) -> list[Path]:
    """Return TIFF files regardless of extension capitalization."""
    return sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".tif", ".tiff"}
        ]
    )


def validate_region(
    region_name: str,
    images_root: Path,
    labels_root: Path,
) -> dict[str, Any]:
    """Validate one OSCD region's dates, bands and change mask."""
    image_region = images_root / region_name
    label_region = labels_root / region_name

    first_date_directory = image_region / "imgs_1_rect"
    second_date_directory = image_region / "imgs_2_rect"
    dates_path = image_region / "dates.txt"
    mask_path = label_region / "cm" / "cm.png"

    required_paths = (
        image_region,
        label_region,
        first_date_directory,
        second_date_directory,
        dates_path,
        mask_path,
    )

    missing = [
        str(path)
        for path in required_paths
        if not path.exists()
    ]

    if missing:
        raise AcquisitionError(
            f"Region '{region_name}' is incomplete: {missing}"
        )

    first_bands = find_tif_files(
        first_date_directory
    )
    second_bands = find_tif_files(
        second_date_directory
    )

    if len(first_bands) != 13:
        raise AcquisitionError(
            f"{region_name}/date-1 contains "
            f"{len(first_bands)} bands; expected 13."
        )

    if len(second_bands) != 13:
        raise AcquisitionError(
            f"{region_name}/date-2 contains "
            f"{len(second_bands)} bands; expected 13."
        )

    dates = [
        line.strip()
        for line in dates_path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
        if line.strip()
    ]

    if len(dates) < 2:
        raise AcquisitionError(
            f"Region '{region_name}' has invalid dates.txt."
        )

    return {
        "region": region_name,
        "date_1_band_count": len(first_bands),
        "date_2_band_count": len(second_bands),
        "dates": dates,
        "mask_path": str(mask_path),
    }


def validate_dataset(
    raw_directory: Path,
) -> dict[str, Any]:
    """Validate the complete extracted OSCD directory structure."""
    images_root = raw_directory / ARCHIVES[0].extracted_directory
    train_root = raw_directory / ARCHIVES[1].extracted_directory
    test_root = raw_directory / ARCHIVES[2].extracted_directory

    for path in (images_root, train_root, test_root):
        if not path.is_dir():
            raise AcquisitionError(
                f"Required OSCD directory is missing: {path}"
            )

    train_regions = sorted(
        path.name
        for path in train_root.iterdir()
        if path.is_dir()
    )
    test_regions = sorted(
        path.name
        for path in test_root.iterdir()
        if path.is_dir()
    )
    image_regions = sorted(
        path.name
        for path in images_root.iterdir()
        if path.is_dir()
    )

    if len(train_regions) != 14:
        raise AcquisitionError(
            f"Expected 14 train regions; found "
            f"{len(train_regions)}."
        )

    if len(test_regions) != 10:
        raise AcquisitionError(
            f"Expected 10 test regions; found "
            f"{len(test_regions)}."
        )

    if len(image_regions) != 24:
        raise AcquisitionError(
            f"Expected 24 image regions; found "
            f"{len(image_regions)}."
        )

    all_label_regions = train_regions + test_regions

    if set(all_label_regions) != set(image_regions):
        raise AcquisitionError(
            "Image regions and labelled regions do not match."
        )

    train_validation = [
        validate_region(
            region_name=region,
            images_root=images_root,
            labels_root=train_root,
        )
        for region in train_regions
    ]
    test_validation = [
        validate_region(
            region_name=region,
            images_root=images_root,
            labels_root=test_root,
        )
        for region in test_regions
    ]

    return {
        "image_region_count": len(image_regions),
        "train_region_count": len(train_regions),
        "test_region_count": len(test_regions),
        "total_region_count": len(all_label_regions),
        "validated_date_rasters": (
            len(all_label_regions) * 2
        ),
        "validated_band_files": (
            len(all_label_regions) * 2 * 13
        ),
        "train_regions": train_validation,
        "test_regions": test_validation,
    }


def write_dataset_card(path: Path) -> None:
    """Write OSCD provenance, usage and limitation documentation."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    content = """# OSCD Dataset Card

## Dataset

Onera Satellite Change Detection Dataset (OSCD).

OSCD contains 24 registered pairs of Sentinel-2 multispectral images
captured between 2015 and 2018. Each region provides 13 spectral bands
at native 10 m, 20 m and 60 m resolutions.

## Labels

Pixel-level urban-change labels are provided for:

- 14 training regions
- 10 testing regions

The primary labelled changes are urban developments such as new
buildings and roads. Original masks use 0 for unchanged pixels and
255 for changed pixels.

## GeoWatch intended use

OSCD is GeoWatch's primary standardized multispectral benchmark. It is
kept separate from the custom Hyderabad operational AOI. Metrics from
OSCD must be labelled as benchmark results and must not be presented as
Hyderabad operational performance.

GeoWatch model inputs will later use:

- B02
- B03
- B04
- B08
- B11
- B12

Band resampling and preprocessing are performed in a derived directory.
The downloaded raw dataset remains immutable.

## Limitations

- Only 24 geographic regions are available.
- Labels primarily represent urban changes.
- Sentinel-2 resolution limits detection of very small buildings.
- Seasonal, atmospheric and radiometric differences may resemble change.
- Dataset results do not establish global operational performance.

## Legal and redistribution note

The official project page does not provide a clear machine-readable
licence statement. Use the dataset for research and benchmarking, retain
the required citation, and verify redistribution or commercial-use terms
before publishing dataset copies.

## Citation

Rodrigo Caye Daudt, Bertrand Le Saux, Alexandre Boulch and Yann
Gousseau. Urban Change Detection for Multispectral Earth Observation
Using Convolutional Neural Networks. IGARSS 2018.

DOI: 10.1109/IGARSS.2018.8518015

Official project page:
https://rcdaudt.github.io/oscd/
"""

    temporary_path = path.with_suffix(
        f"{path.suffix}.tmp"
    )
    temporary_path.write_text(
        content,
        encoding="utf-8",
    )
    temporary_path.replace(path)


def write_json_atomic(
    payload: Mapping[str, Any],
    path: Path,
) -> None:
    """Write a JSON report atomically."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        f"{path.suffix}.tmp"
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

    temporary_path.replace(path)


def main() -> int:
    """Run OSCD acquisition."""
    parser = argparse.ArgumentParser(
        description=(
            "Download, checksum, extract and validate the OSCD dataset."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data_config.yaml"),
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        benchmark = require_mapping(
            config.get("benchmark"),
            "benchmark",
        )
        oscd = require_mapping(
            benchmark.get("oscd"),
            "benchmark.oscd",
        )

        archive_directory = Path(
            str(oscd.get("archive_dir"))
        )
        raw_directory = Path(
            str(oscd.get("raw_dir"))
        )
        manifest_path = Path(
            str(oscd.get("manifest_path"))
        )
        dataset_card_path = Path(
            str(oscd.get("dataset_card_path"))
        )

        session = create_http_session()
        archive_reports: list[dict[str, Any]] = []

        for spec in ARCHIVES:
            archive_path = (
                archive_directory / spec.filename
            )

            download_status = download_archive(
                session=session,
                spec=spec,
                destination=archive_path,
                force=args.force_download,
            )

            expected_directory = (
                raw_directory
                / spec.extracted_directory
            )

            extraction_status = safely_extract_zip(
                archive_path=archive_path,
                destination=raw_directory,
                force=args.force_extract,
                expected_directory=expected_directory,
            )

            archive_reports.append(
                {
                    "key": spec.key,
                    "filename": spec.filename,
                    "source_url": spec.url,
                    "local_path": str(archive_path),
                    "download_status": download_status,
                    "extraction_status": extraction_status,
                    "size_bytes": archive_path.stat().st_size,
                    "expected_md5": spec.expected_md5,
                    "actual_md5": calculate_md5(
                        archive_path
                    ),
                    "sha256": calculate_sha256(
                        archive_path
                    ),
                }
            )

            print(
                f"  {spec.key}: {download_status}, "
                f"{extraction_status}"
            )

        validation = validate_dataset(
            raw_directory
        )

        write_dataset_card(
            dataset_card_path
        )

        manifest = {
            "schema_version": "1.0",
            "generated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "status": "success",
            "dataset_name": (
                "Onera Satellite Change Detection Dataset"
            ),
            "dataset_short_name": "OSCD",
            "official_project_page": (
                "https://rcdaudt.github.io/oscd/"
            ),
            "citation_doi": (
                "10.1109/IGARSS.2018.8518015"
            ),
            "sensor": "Sentinel-2",
            "spectral_band_count": 13,
            "native_resolutions_meters": [
                10,
                20,
                60,
            ],
            "archives": archive_reports,
            "validation": validation,
            "raw_directory": str(raw_directory),
            "dataset_card": str(dataset_card_path),
        }

        write_json_atomic(
            manifest,
            manifest_path,
        )

        print("OSCD acquisition completed")
        print(
            "  Image regions:",
            validation["image_region_count"],
        )
        print(
            "  Train regions:",
            validation["train_region_count"],
        )
        print(
            "  Test regions:",
            validation["test_region_count"],
        )
        print(
            "  Band files validated:",
            validation["validated_band_files"],
        )
        print("  Manifest:", manifest_path)
        print("  Dataset card:", dataset_card_path)

        return 0

    except (
        AcquisitionError,
        FileNotFoundError,
        PermissionError,
        requests.RequestException,
        zipfile.BadZipFile,
        yaml.YAMLError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

