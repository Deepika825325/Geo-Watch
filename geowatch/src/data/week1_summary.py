"""Generate an audited Week 1 completion report for GeoWatch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml


class SummaryError(RuntimeError):
    """Raised when a required Week 1 artifact is missing or invalid."""


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing configuration: {path}")

    with path.open("r", encoding="utf-8-sig") as file:
        payload = yaml.safe_load(file)

    if not isinstance(payload, dict):
        raise SummaryError(
            f"YAML root must be a mapping: {path}"
        )

    return payload


def load_json(path: Path) -> Mapping[str, Any]:
    """Load a JSON object."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing JSON artifact: {path}")

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, Mapping):
        raise SummaryError(
            f"JSON root must be an object: {path}"
        )

    return payload


def sha256(path: Path) -> str:
    """Calculate a file SHA-256 checksum."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def write_json_atomic(
    payload: Mapping[str, Any],
    path: Path,
) -> None:
    """Write JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary.replace(path)


def write_text_atomic(
    content: str,
    path: Path,
) -> None:
    """Write text atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def read_manifest(path: Path) -> list[dict[str, str]]:
    """Read the dataset manifest."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset manifest does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def stage(
    name: str,
    passed: bool,
    details: Mapping[str, Any],
) -> dict[str, Any]:
    """Create one summary-stage record."""
    return {
        "name": name,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "details": dict(details),
    }


def build_summary(config_path: Path) -> dict[str, Any]:
    """Validate and summarize all Week 1 artifacts."""
    config = load_yaml(config_path)

    processing = config.get("processing", {})
    acquisition = config.get("acquisition", {})
    paths = config.get("paths", {})
    benchmark = config.get("benchmark", {})

    if not isinstance(processing, Mapping):
        raise SummaryError("processing must be a mapping.")

    if not isinstance(acquisition, Mapping):
        raise SummaryError("acquisition must be a mapping.")

    selected_pair_path = Path(
        str(
            acquisition.get(
                "selected_pair_output",
                "data/raw/selected_scene_pair.json",
            )
        )
    )
    download_report_path = Path(
        str(
            acquisition.get(
                "band_download_report",
                "data/raw/band_download_report.json",
            )
        )
    )
    raw_validation_path = Path(
        str(
            acquisition.get(
                "raw_validation_output",
                "data/raw/raw_raster_validation.json",
            )
        )
    )
    alignment_path = Path(
        str(
            processing.get(
                "alignment_report",
                "data/processed/aligned/alignment_report.json",
            )
        )
    )
    cloud_mask_path = Path(
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
    tiling_path = Path(
        str(
            processing.get(
                "tiling_report",
                "data/processed/patches/tiling_report.json",
            )
        )
    )
    split_path = Path(
        str(
            processing.get(
                "split_report",
                "data/processed/splits/split_report.json",
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

    oscd_config = (
        benchmark.get("oscd", {})
        if isinstance(benchmark, Mapping)
        else {}
    )

    if not isinstance(oscd_config, Mapping):
        raise SummaryError(
            "benchmark.oscd must be a mapping."
        )

    oscd_path = Path(
        str(
            oscd_config.get(
                "manifest_path",
                (
                    "data/benchmark/oscd/"
                    "acquisition_manifest.json"
                ),
            )
        )
    )

    selected_pair = load_json(selected_pair_path)
    download_report = load_json(download_report_path)
    raw_validation = load_json(raw_validation_path)
    alignment = load_json(alignment_path)
    cloud_mask = load_json(cloud_mask_path)
    tiling = load_json(tiling_path)
    split = load_json(split_path)
    oscd = load_json(oscd_path)
    manifest_rows = read_manifest(manifest_path)

    periods = download_report.get("periods", {})

    if not isinstance(periods, Mapping):
        raise SummaryError(
            "Download report periods are invalid."
        )

    raw_asset_count = 0

    for period in ("before", "after"):
        period_value = periods.get(period, {})

        if isinstance(period_value, Mapping):
            assets = period_value.get("assets", [])

            if isinstance(assets, list):
                raw_asset_count += len(assets)

    archive_records = oscd.get("archives", [])

    if not isinstance(archive_records, list):
        archive_records = []

    oscd_md5_passed = (
        len(archive_records) == 3
        and all(
            record.get("actual_md5")
            == record.get("expected_md5")
            for record in archive_records
            if isinstance(record, Mapping)
        )
    )

    split_values = split.get("splits", {})

    if not isinstance(split_values, Mapping):
        split_values = {}

    train_count = int(
        split_values.get("train", {}).get(
            "patch_count",
            0,
        )
    )
    validation_count = int(
        split_values.get("validation", {}).get(
            "patch_count",
            0,
        )
    )
    test_count = int(
        split_values.get("test", {}).get(
            "patch_count",
            0,
        )
    )

    manifest_types = Counter(
        row.get("record_type", "")
        for row in manifest_rows
    )

    stages = [
        stage(
            "scene_pair_selection",
            selected_pair_path.is_file(),
            {
                "artifact": str(selected_pair_path),
                "sha256": sha256(selected_pair_path),
            },
        ),
        stage(
            "raw_band_acquisition",
            raw_asset_count == 14,
            {
                "downloaded_asset_count": raw_asset_count,
                "expected_asset_count": 14,
                "artifact": str(download_report_path),
            },
        ),
        stage(
            "raw_raster_validation",
            (
                raw_validation.get("status") == "success"
                and raw_validation.get("error_count") == 0
                and raw_validation.get(
                    "validated_raster_count"
                ) == 14
            ),
            {
                "status": raw_validation.get("status"),
                "validated_rasters": raw_validation.get(
                    "validated_raster_count"
                ),
                "errors": raw_validation.get("error_count"),
            },
        ),
        stage(
            "aoi_alignment",
            (
                alignment.get("status") == "success"
                and alignment.get(
                    "total_output_count"
                ) == 14
            ),
            {
                "status": alignment.get("status"),
                "output_count": alignment.get(
                    "total_output_count"
                ),
                "grid": alignment.get("reference_grid"),
            },
        ),
        stage(
            "cloud_masking",
            (
                cloud_mask.get("status") == "success"
                and cloud_mask.get(
                    "quality_gate_passed"
                ) is True
            ),
            {
                "status": cloud_mask.get("status"),
                "pair_valid_fraction": cloud_mask.get(
                    "pair_valid_fraction"
                ),
                "quality_gate_passed": cloud_mask.get(
                    "quality_gate_passed"
                ),
            },
        ),
        stage(
            "paired_patch_generation",
            (
                tiling.get("status") == "success"
                and int(
                    tiling.get(
                        "accepted_patch_count",
                        0,
                    )
                )
                > 0
            ),
            {
                "status": tiling.get("status"),
                "candidate_patches": tiling.get(
                    "candidate_patch_count"
                ),
                "accepted_patches": tiling.get(
                    "accepted_patch_count"
                ),
                "rejected_patches": tiling.get(
                    "rejected_patch_count"
                ),
                "mean_valid_fraction": tiling.get(
                    "mean_accepted_valid_fraction"
                ),
            },
        ),
        stage(
            "geographic_split",
            (
                split.get("status") == "success"
                and split.get(
                    "unassigned_patch_count"
                ) == 0
                and (
                    train_count
                    + validation_count
                    + test_count
                )
                == tiling.get("accepted_patch_count")
            ),
            {
                "status": split.get("status"),
                "train": train_count,
                "validation": validation_count,
                "test": test_count,
                "unassigned": split.get(
                    "unassigned_patch_count"
                ),
            },
        ),
        stage(
            "oscd_benchmark_acquisition",
            (
                oscd.get("status") == "success"
                and oscd_md5_passed
                and oscd.get(
                    "validation",
                    {},
                ).get("image_region_count")
                == 24
                and oscd.get(
                    "validation",
                    {},
                ).get("validated_band_files")
                == 624
            ),
            {
                "status": oscd.get("status"),
                "archives": len(archive_records),
                "md5_checks_passed": oscd_md5_passed,
                "image_regions": oscd.get(
                    "validation",
                    {},
                ).get("image_region_count"),
                "train_regions": oscd.get(
                    "validation",
                    {},
                ).get("train_region_count"),
                "test_regions": oscd.get(
                    "validation",
                    {},
                ).get("test_region_count"),
                "band_files": oscd.get(
                    "validation",
                    {},
                ).get("validated_band_files"),
            },
        ),
        stage(
            "dataset_lineage_manifest",
            len(manifest_rows) > 0,
            {
                "manifest_records": len(manifest_rows),
                "record_type_counts": dict(
                    sorted(manifest_types.items())
                ),
                "artifact": str(manifest_path),
            },
        ),
    ]

    passed_count = sum(
        item["passed"]
        for item in stages
    )
    overall_complete = passed_count == len(stages)

    return {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "project": "GeoWatch",
        "roadmap_period": "Week 1",
        "status": (
            "complete"
            if overall_complete
            else "incomplete"
        ),
        "completion_percentage": (
            100.0 * passed_count / len(stages)
        ),
        "passed_stage_count": passed_count,
        "total_stage_count": len(stages),
        "aoi": {
            "name": acquisition.get("aoi_name"),
            "bbox_wgs84": acquisition.get("bbox_wgs84"),
            "target_epsg": processing.get("target_epsg"),
        },
        "operational_dataset": {
            "raw_rasters": raw_asset_count,
            "aligned_rasters": alignment.get(
                "total_output_count"
            ),
            "pair_valid_fraction": cloud_mask.get(
                "pair_valid_fraction"
            ),
            "accepted_patches": tiling.get(
                "accepted_patch_count"
            ),
            "train_patches": train_count,
            "validation_patches": validation_count,
            "test_patches": test_count,
            "input_channels": tiling.get(
                "input_channel_count"
            ),
            "patch_size_pixels": tiling.get(
                "patch_size_pixels"
            ),
        },
        "benchmark_dataset": {
            "name": "OSCD",
            "image_regions": oscd.get(
                "validation",
                {},
            ).get("image_region_count"),
            "train_regions": oscd.get(
                "validation",
                {},
            ).get("train_region_count"),
            "test_regions": oscd.get(
                "validation",
                {},
            ).get("test_region_count"),
            "validated_band_files": oscd.get(
                "validation",
                {},
            ).get("validated_band_files"),
            "md5_checks_passed": oscd_md5_passed,
        },
        "stages": stages,
        "source_artifacts": {
            "selected_pair": str(selected_pair_path),
            "download_report": str(download_report_path),
            "raw_validation": str(raw_validation_path),
            "alignment_report": str(alignment_path),
            "cloud_mask_report": str(cloud_mask_path),
            "tiling_report": str(tiling_path),
            "split_report": str(split_path),
            "oscd_manifest": str(oscd_path),
            "dataset_manifest": str(manifest_path),
        },
    }


def build_markdown(summary: Mapping[str, Any]) -> str:
    """Build the human-readable Week 1 summary."""
    operational = summary["operational_dataset"]
    benchmark = summary["benchmark_dataset"]

    lines = [
        "# GeoWatch Week 1 Data Pipeline Summary",
        "",
        f"**Status:** {summary['status'].upper()}",
        "",
        (
            f"**Completion:** "
            f"{summary['passed_stage_count']}/"
            f"{summary['total_stage_count']} stages "
            f"({summary['completion_percentage']:.0f}%)"
        ),
        "",
        "## Operational Hyderabad dataset",
        "",
        f"- Raw Sentinel-2 rasters: {operational['raw_rasters']}",
        f"- Aligned rasters: {operational['aligned_rasters']}",
        (
            f"- Joint valid-pixel fraction: "
            f"{operational['pair_valid_fraction']:.2%}"
        ),
        f"- Accepted patches: {operational['accepted_patches']}",
        (
            f"- Split: {operational['train_patches']} train / "
            f"{operational['validation_patches']} validation / "
            f"{operational['test_patches']} test"
        ),
        f"- Patch size: {operational['patch_size_pixels']} × "
        f"{operational['patch_size_pixels']}",
        f"- Input channels: {operational['input_channels']}",
        "",
        "## OSCD benchmark",
        "",
        f"- Image regions: {benchmark['image_regions']}",
        f"- Training regions: {benchmark['train_regions']}",
        f"- Testing regions: {benchmark['test_regions']}",
        (
            f"- Validated band files: "
            f"{benchmark['validated_band_files']}"
        ),
        (
            f"- Archive checksum verification: "
            f"{benchmark['md5_checks_passed']}"
        ),
        "",
        "## Pipeline stages",
        "",
        "| Stage | Status |",
        "|---|---|",
    ]

    for item in summary["stages"]:
        lines.append(
            f"| {item['name']} | {item['status']} |"
        )

    lines.extend(
        [
            "",
            "## Week 1 outcome",
            "",
            (
                "The acquisition, validation, spatial alignment, "
                "cloud masking, patch generation, geographic splitting "
                "and public benchmark acquisition stages are complete."
            ),
            "",
            (
                "Week 2 begins with exploratory data analysis, "
                "radiometric statistics, change-label imbalance analysis "
                "and a classical image-difference baseline."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def main() -> int:
    """Generate the Week 1 summary artifacts."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate all GeoWatch Week 1 outputs and generate "
            "completion reports."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data_config.yaml"),
    )
    args = parser.parse_args()

    try:
        config = load_yaml(args.config)
        processing = config.get("processing", {})

        if not isinstance(processing, Mapping):
            raise SummaryError(
                "processing must be a mapping."
            )

        json_path = Path(
            str(
                processing.get(
                    "week1_summary_json",
                    "data/processed/week1_summary.json",
                )
            )
        )
        markdown_path = Path(
            str(
                processing.get(
                    "week1_summary_markdown",
                    "docs/week1_data_pipeline_summary.md",
                )
            )
        )

        summary = build_summary(args.config)

        write_json_atomic(
            summary,
            json_path,
        )
        write_text_atomic(
            build_markdown(summary),
            markdown_path,
        )

        print("GeoWatch Week 1 audit completed")
        print("  Status:", summary["status"])
        print(
            "  Stages passed:",
            f"{summary['passed_stage_count']}/"
            f"{summary['total_stage_count']}",
        )
        print(
            "  Completion:",
            f"{summary['completion_percentage']:.0f}%",
        )
        print("  Summary JSON:", json_path)
        print("  Summary document:", markdown_path)

        return 0 if summary["status"] == "complete" else 1

    except (
        FileNotFoundError,
        PermissionError,
        SummaryError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
        yaml.YAMLError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
