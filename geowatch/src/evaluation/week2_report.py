"""Generate the GeoWatch Week 2 classical-baseline comparison report.

The report consolidates:

* OSCD exploratory data analysis;
* Band Difference + Otsu results;
* CVA + PCA + K-Means results.

Only labelled OSCD regions contribute quantitative metrics. GeoWatch's
unlabelled Hyderabad AOI is deliberately excluded from all reported scores.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


LOGGER = logging.getLogger(
    "geowatch.week2_report"
)


class ReportError(RuntimeError):
    """Raised when Week 2 reports are missing or inconsistent."""


@dataclass(frozen=True)
class BaselineSummary:
    """Test metrics and metadata for one classical baseline."""

    identifier: str
    display_name: str
    precision: float
    recall: float
    f1_score: float
    iou: float
    accuracy: float
    change_prevalence: float
    predicted_change_fraction: float
    report_path: Path


def load_json(
    path: Path,
) -> Mapping[str, Any]:
    """Load and validate a JSON object."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Required JSON report does not exist: {path}"
        )

    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8",
            )
        )
    except json.JSONDecodeError as error:
        raise ReportError(
            f"Invalid JSON report: {path}"
        ) from error

    if not isinstance(
        payload,
        Mapping,
    ):
        raise ReportError(
            f"JSON root must be an object: {path}"
        )

    return payload


def require_mapping(
    payload: Mapping[str, Any],
    key: str,
    source: Path,
) -> Mapping[str, Any]:
    """Read one required nested mapping."""
    value = payload.get(key)

    if not isinstance(
        value,
        Mapping,
    ):
        raise ReportError(
            f"{source} is missing mapping {key!r}."
        )

    return value


def require_probability(
    payload: Mapping[str, Any],
    key: str,
    source: Path,
) -> float:
    """Read a metric constrained to the interval zero through one."""
    value = payload.get(key)

    if not isinstance(
        value,
        (int, float),
    ):
        raise ReportError(
            f"{source} has invalid metric {key!r}: {value!r}"
        )

    numeric_value = float(value)

    if not 0.0 <= numeric_value <= 1.0:
        raise ReportError(
            f"{source} metric {key!r} is outside [0, 1]: "
            f"{numeric_value}"
        )

    return numeric_value


def validate_common_report(
    report: Mapping[str, Any],
    source: Path,
    expected_identifier: str,
) -> None:
    """Validate fields shared by both baseline reports."""
    if report.get("status") != "success":
        raise ReportError(
            f"Baseline report is not successful: {source}"
        )

    if report.get("dataset") != "OSCD":
        raise ReportError(
            f"Baseline report does not use OSCD: {source}"
        )

    if (
        report.get("baseline")
        != expected_identifier
    ):
        raise ReportError(
            f"Expected baseline {expected_identifier!r} "
            f"in {source}; found {report.get('baseline')!r}."
        )

    if (
        report.get(
            "custom_hyderabad_aoi_used_for_metrics"
        )
        is not False
    ):
        raise ReportError(
            f"Hyderabad AOI was not explicitly excluded in {source}."
        )

    region_counts = require_mapping(
        report,
        "region_counts",
        source,
    )

    expected_counts = {
        "train": 14,
        "test": 10,
        "overall": 24,
    }

    if dict(region_counts) != expected_counts:
        raise ReportError(
            f"Unexpected OSCD region counts in {source}: "
            f"{dict(region_counts)}"
        )


def validate_otsu_protocol(
    report: Mapping[str, Any],
    source: Path,
) -> None:
    """Validate that Otsu threshold fitting did not leak test data."""
    method = require_mapping(
        report,
        "method",
        source,
    )

    if (
        method.get("threshold_source")
        != "OSCD training image pixels only"
    ):
        raise ReportError(
            "Otsu threshold was not fitted exclusively "
            f"from training imagery: {source}"
        )

    for key in (
        "training_labels_used_for_threshold",
        "test_images_used_for_threshold",
        "test_labels_used_for_threshold",
    ):
        if method.get(key) is not False:
            raise ReportError(
                f"Otsu protocol field {key!r} is not False "
                f"in {source}."
            )


def validate_cva_protocol(
    report: Mapping[str, Any],
    source: Path,
) -> None:
    """Validate that CVA model fitting did not leak test data."""
    method = require_mapping(
        report,
        "method",
        source,
    )

    expected_locations = {
        "pca_fitted_on": (
            "OSCD training image pixels only"
        ),
        "kmeans_fitted_on": (
            "OSCD training image pixels only"
        ),
    }

    for key, expected_value in expected_locations.items():
        if method.get(key) != expected_value:
            raise ReportError(
                f"Invalid CVA protocol field {key!r} "
                f"in {source}."
            )

    for key in (
        "training_labels_used_for_fitting",
        "test_images_used_for_fitting",
        "test_labels_used_for_fitting",
    ):
        if method.get(key) is not False:
            raise ReportError(
                f"CVA protocol field {key!r} is not False "
                f"in {source}."
            )


def parse_baseline(
    path: Path,
    expected_identifier: str,
    display_name: str,
) -> tuple[BaselineSummary, Mapping[str, Any]]:
    """Load and summarize one baseline result."""
    report = load_json(path)

    validate_common_report(
        report=report,
        source=path,
        expected_identifier=expected_identifier,
    )

    if expected_identifier == "band_difference_otsu":
        validate_otsu_protocol(
            report,
            path,
        )
    elif expected_identifier == "cva_pca_kmeans":
        validate_cva_protocol(
            report,
            path,
        )
    else:
        raise ReportError(
            f"Unsupported baseline identifier: "
            f"{expected_identifier}"
        )

    metrics = require_mapping(
        report,
        "test_metrics",
        path,
    )

    summary = BaselineSummary(
        identifier=expected_identifier,
        display_name=display_name,
        precision=require_probability(
            metrics,
            "precision",
            path,
        ),
        recall=require_probability(
            metrics,
            "recall",
            path,
        ),
        f1_score=require_probability(
            metrics,
            "f1_score",
            path,
        ),
        iou=require_probability(
            metrics,
            "iou",
            path,
        ),
        accuracy=require_probability(
            metrics,
            "accuracy",
            path,
        ),
        change_prevalence=require_probability(
            metrics,
            "change_prevalence",
            path,
        ),
        predicted_change_fraction=require_probability(
            metrics,
            "predicted_change_fraction",
            path,
        ),
        report_path=path,
    )

    return summary, report


def validate_eda(
    path: Path,
) -> Mapping[str, Any]:
    """Validate the OSCD EDA artifact."""
    report = load_json(path)

    if report.get("status") != "success":
        raise ReportError(
            f"EDA report is not successful: {path}"
        )

    if report.get("dataset") != "OSCD":
        raise ReportError(
            f"EDA report does not use OSCD: {path}"
        )

    if (
        report.get(
            "custom_hyderabad_aoi_used_for_metrics"
        )
        is not False
    ):
        raise ReportError(
            "Hyderabad AOI was not excluded from EDA metrics."
        )

    counts = require_mapping(
        report,
        "region_counts",
        path,
    )

    if dict(counts) != {
        "train": 14,
        "test": 10,
        "overall": 24,
    }:
        raise ReportError(
            f"Unexpected EDA region counts: {dict(counts)}"
        )

    return report


def relative_improvement(
    improved: float,
    reference: float,
) -> float | None:
    """Return relative improvement, or None for a zero reference."""
    if reference == 0.0:
        return None

    return (
        improved - reference
    ) / reference


def format_relative(
    value: float | None,
) -> str:
    """Format a relative improvement safely."""
    if value is None:
        return "undefined because the reference value was zero"

    return f"{value:.2%}"


def build_markdown(
    eda: Mapping[str, Any],
    otsu: BaselineSummary,
    cva: BaselineSummary,
    cva_report: Mapping[str, Any],
) -> str:
    """Build the final Week 2 Markdown report."""
    overall_eda = require_mapping(
        eda,
        "overall",
        Path("EDA report"),
    )
    train_eda = require_mapping(
        eda,
        "train",
        Path("EDA report"),
    )
    test_eda = require_mapping(
        eda,
        "test",
        Path("EDA report"),
    )

    change_fraction = float(
        overall_eda[
            "pixel_weighted_change_fraction"
        ]
    )
    imbalance_ratio = float(
        overall_eda[
            "unchanged_to_change_ratio"
        ]
    )

    baselines = sorted(
        [otsu, cva],
        key=lambda item: (
            item.f1_score,
            item.iou,
        ),
        reverse=True,
    )
    winner = baselines[0]
    runner_up = baselines[1]

    f1_relative = relative_improvement(
        winner.f1_score,
        runner_up.f1_score,
    )
    iou_relative = relative_improvement(
        winner.iou,
        runner_up.iou,
    )

    cva_method = require_mapping(
        cva_report,
        "method",
        cva.report_path,
    )

    explained_variance = [
        float(value)
        for value in cva_method[
            "pca_explained_variance_ratio"
        ]
    ]
    explained_variance_total = sum(
        explained_variance
    )

    cluster_magnitudes = [
        float(value)
        for value in cva_method[
            "cluster_mean_cva_magnitude"
        ]
    ]

    cluster_gap = abs(
        cluster_magnitudes[1]
        - cluster_magnitudes[0]
    )
    cluster_gap_relative = (
        cluster_gap
        / max(cluster_magnitudes)
        if max(cluster_magnitudes) > 0
        else 0.0
    )

    lines = [
        "# GeoWatch Week 2 — EDA and Classical Baselines",
        "",
        "## Evaluation protocol",
        "",
        (
            "- Quantitative evaluation uses only the labelled "
            "OSCD benchmark."
        ),
        (
            "- The unlabelled Hyderabad AOI is excluded from "
            "all reported metrics."
        ),
        (
            "- OSCD's official 14-region training and 10-region "
            "testing split is preserved."
        ),
        (
            "- Thresholds, PCA, scaling and clustering are fitted "
            "using training imagery only."
        ),
        (
            "- Precision, recall, F1 and IoU are reported for the "
            "positive change class."
        ),
        (
            "- Overall pixel accuracy is retained only as a "
            "secondary diagnostic."
        ),
        "",
        "## Dataset analysis",
        "",
        (
            f"- Overall changed-pixel fraction: "
            f"**{change_fraction:.4%}**"
        ),
        (
            f"- Training changed-pixel fraction: "
            f"**{float(train_eda['pixel_weighted_change_fraction']):.4%}**"
        ),
        (
            f"- Test changed-pixel fraction: "
            f"**{float(test_eda['pixel_weighted_change_fraction']):.4%}**"
        ),
        (
            f"- Overall unchanged-to-change ratio: "
            f"**{imbalance_ratio:.2f}:1**"
        ),
        "",
        (
            "The severe class imbalance explains why overall pixel "
            "accuracy is not an appropriate headline metric. A method "
            "can classify most unchanged pixels correctly while still "
            "performing poorly on the change class."
        ),
        "",
        "## OSCD test results",
        "",
        (
            "| Baseline | Precision | Recall | F1 | IoU | "
            "Accuracy* | Predicted change |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for baseline in (
        otsu,
        cva,
    ):
        lines.append(
            f"| {baseline.display_name} "
            f"| {baseline.precision:.6f} "
            f"| {baseline.recall:.6f} "
            f"| {baseline.f1_score:.6f} "
            f"| {baseline.iou:.6f} "
            f"| {baseline.accuracy:.6f} "
            f"| {baseline.predicted_change_fraction:.2%} |"
        )

    lines.extend(
        [
            "",
            (
                "\\*Accuracy is shown only as a secondary diagnostic "
                "because unchanged pixels dominate the dataset."
            ),
            "",
            "## Result interpretation",
            "",
            (
                f"**{winner.display_name}** is the stronger Week 2 "
                f"baseline by both F1 and IoU."
            ),
            "",
            (
                f"- F1 improvement over {runner_up.display_name}: "
                f"**{format_relative(f1_relative)}** relative "
                f"improvement, or "
                f"**{100 * (winner.f1_score - runner_up.f1_score):.2f} "
                "percentage points**."
            ),
            (
                f"- IoU improvement over {runner_up.display_name}: "
                f"**{format_relative(iou_relative)}** relative "
                f"improvement, or "
                f"**{100 * (winner.iou - runner_up.iou):.2f} "
                "percentage points**."
            ),
            (
                f"- {cva.display_name} achieved higher recall "
                f"({cva.recall:.6f}) but only "
                f"{cva.precision:.6f} precision, indicating substantial "
                "over-prediction."
            ),
            (
                f"- The first three PCA components retained "
                f"{explained_variance_total:.2%} of training-vector "
                "variance."
            ),
            (
                f"- The two K-Means clusters differed in mean CVA "
                f"magnitude by only {cluster_gap_relative:.2%}, "
                "suggesting weak separation between unchanged and "
                "changed pixels under the unsupervised cluster rule."
            ),
            "",
            "## Baseline limitations",
            "",
            (
                "- Radiometric and seasonal differences can be mistaken "
                "for real land-cover change."
            ),
            (
                "- Neither baseline learns spatial context, object shape "
                "or semantic land-use patterns."
            ),
            (
                "- No morphological cleanup was applied, preserving a "
                "simple and reproducible comparison."
            ),
            (
                "- The methods operate on four native 10 m bands: "
                "B02, B03, B04 and B08."
            ),
            "",
            "## Week 3 target",
            "",
            (
                "The Siamese U-Net must exceed the strongest classical "
                f"test baseline of **F1={winner.f1_score:.6f}** and "
                f"**IoU={winner.iou:.6f}** while producing more "
                "spatially coherent change masks."
            ),
            "",
            "## Source artifacts",
            "",
            f"- EDA: `{eda['outputs']['region_statistics_csv']}`",
            f"- Otsu report: `{otsu.report_path}`",
            f"- CVA report: `{cva.report_path}`",
            "",
        ]
    )

    return "\n".join(lines)


def write_text_atomic(
    content: str,
    output_path: Path,
) -> None:
    """Write the Markdown report atomically."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        f"{output_path.suffix}.tmp"
    )

    temporary_path.write_text(
        content,
        encoding="utf-8",
    )

    temporary_path.replace(
        output_path
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the report-generator CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate the GeoWatch Week 2 comparison report "
            "from validated EDA and baseline JSON artifacts."
        )
    )

    parser.add_argument(
        "--eda-report",
        type=Path,
        default=Path(
            "reports/week2/eda/"
            "oscd_dataset_statistics.json"
        ),
    )
    parser.add_argument(
        "--otsu-report",
        type=Path,
        default=Path(
            "reports/week2/baselines/"
            "band_diff_otsu/"
            "band_diff_otsu_report.json"
        ),
    )
    parser.add_argument(
        "--cva-report",
        type=Path,
        default=Path(
            "reports/week2/baselines/"
            "cva_pca_kmeans/"
            "cva_pca_kmeans_report.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/week2_baseline_report.md"
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


def main() -> int:
    """Generate the validated Week 2 comparison report."""
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
        eda = validate_eda(
            args.eda_report
        )

        otsu, _ = parse_baseline(
            path=args.otsu_report,
            expected_identifier=(
                "band_difference_otsu"
            ),
            display_name=(
                "Band Difference + Otsu"
            ),
        )

        cva, cva_report = parse_baseline(
            path=args.cva_report,
            expected_identifier=(
                "cva_pca_kmeans"
            ),
            display_name=(
                "CVA + PCA + K-Means"
            ),
        )

        markdown = build_markdown(
            eda=eda,
            otsu=otsu,
            cva=cva,
            cva_report=cva_report,
        )

        write_text_atomic(
            content=markdown,
            output_path=args.output,
        )

        winner = max(
            (otsu, cva),
            key=lambda item: (
                item.f1_score,
                item.iou,
            ),
        )

        print(
            "Week 2 baseline report generated"
        )
        print(
            "  Status: success"
        )
        print(
            "  Strongest baseline:",
            winner.display_name,
        )
        print(
            "  Best test F1:",
            f"{winner.f1_score:.6f}",
        )
        print(
            "  Best test IoU:",
            f"{winner.iou:.6f}",
        )
        print(
            "  Output:",
            args.output,
        )

        return 0

    except (
        FileNotFoundError,
        PermissionError,
        ReportError,
        KeyError,
        TypeError,
        ValueError,
        OSError,
    ) as error:
        LOGGER.error(
            "%s",
            error,
        )
        return 1

    except Exception:
        LOGGER.exception(
            "Unexpected Week 2 report-generation failure."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
