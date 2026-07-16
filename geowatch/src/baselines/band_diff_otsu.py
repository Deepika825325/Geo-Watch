"""Band-difference and Otsu baseline for GeoWatch.

This classical baseline converts multispectral before/after observations into
one scalar change-intensity image and then applies a global Otsu threshold.

Protocol:

* input bands: Sentinel-2 B02, B03, B04 and B08;
* difference: normalized absolute difference per band;
* fusion: mean difference across bands;
* threshold fitting: OSCD training image pixels only;
* quantitative evaluation: OSCD labelled regions only;
* test labels: used only after the threshold is frozen;
* Hyderabad AOI: excluded from quantitative metrics.

No morphological post-processing is applied so this remains a simple,
reproducible and interpretable baseline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.data.eda import (
    find_band_file,
    list_region_directories,
    read_single_band,
    resolve_oscd_paths,
)
from src.evaluation.metrics import (
    ChangeMetrics,
    ConfusionCounts,
    calculate_change_metrics,
    load_mask,
    safe_divide,
)


LOGGER = logging.getLogger(
    "geowatch.band_diff_otsu"
)


class BaselineError(RuntimeError):
    """Raised when the classical baseline cannot run safely."""


@dataclass(frozen=True)
class RegionResult:
    """Metrics and output paths for one OSCD region."""

    region: str
    split: str
    threshold: float
    width: int
    height: int
    evaluated_pixels: int
    ignored_pixels: int
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    precision: float
    recall: float
    f1_score: float
    iou: float
    accuracy: float
    change_prevalence: float
    predicted_change_fraction: float
    prediction_path: str
    difference_preview_path: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable record."""
        return asdict(self)


@dataclass(frozen=True)
class DifferenceResult:
    """One scalar change image and its valid-pixel mask."""

    difference: np.ndarray
    valid_mask: np.ndarray


def stable_region_seed(
    base_seed: int,
    region: str,
) -> int:
    """Create a deterministic region-specific random seed."""
    digest = hashlib.sha256(
        region.encode("utf-8")
    ).digest()

    region_value = int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )

    return (
        int(base_seed)
        + region_value
    ) % (2**32)


def load_difference_image(
    images_root: Path,
    region: str,
    bands: Sequence[str],
    epsilon: float,
) -> DifferenceResult:
    """Create a normalized multispectral absolute-difference image.

    For each selected band, the normalized absolute difference is:

        abs(after - before) / (abs(after) + abs(before) + epsilon)

    This bounds the difference approximately within 0–1 and makes it less
    dependent on the absolute reflectance magnitude of each band.
    """
    if not bands:
        raise BaselineError(
            "At least one input band is required."
        )

    if epsilon <= 0:
        raise BaselineError(
            "epsilon must be greater than zero."
        )

    image_region = (
        images_root
        / region
    )
    before_directory = (
        image_region
        / "imgs_1_rect"
    )
    after_directory = (
        image_region
        / "imgs_2_rect"
    )

    differences: list[np.ndarray] = []
    validity_masks: list[np.ndarray] = []
    expected_shape: tuple[int, int] | None = None

    for band_name in bands:
        before_path = find_band_file(
            before_directory,
            band_name,
        )
        after_path = find_band_file(
            after_directory,
            band_name,
        )

        before = read_single_band(
            before_path
        )
        after = read_single_band(
            after_path
        )

        if before.shape != after.shape:
            raise BaselineError(
                f"Before/after shapes differ for {region} "
                f"{band_name}: {before.shape} versus {after.shape}."
            )

        if expected_shape is None:
            expected_shape = before.shape
        elif before.shape != expected_shape:
            raise BaselineError(
                f"Selected bands do not share one shape for {region}: "
                f"expected {expected_shape}, received {before.shape} "
                f"for {band_name}."
            )

        finite = (
            np.isfinite(before)
            & np.isfinite(after)
        )

        nonempty_observation = (
            (before != 0)
            | (after != 0)
        )

        valid = (
            finite
            & nonempty_observation
        )

        numerator = np.abs(
            after - before
        )
        denominator = (
            np.abs(after)
            + np.abs(before)
            + epsilon
        )

        normalized_difference = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(
                numerator,
                dtype=np.float32,
            ),
            where=valid,
        )

        normalized_difference = np.clip(
            normalized_difference,
            0.0,
            1.0,
        ).astype(
            np.float32,
            copy=False,
        )

        differences.append(
            normalized_difference
        )
        validity_masks.append(
            valid
        )

    difference_stack = np.stack(
        differences,
        axis=0,
    )
    valid_stack = np.stack(
        validity_masks,
        axis=0,
    )

    jointly_valid = np.all(
        valid_stack,
        axis=0,
    )

    fused_difference = np.mean(
        difference_stack,
        axis=0,
        dtype=np.float32,
    )

    fused_difference[
        ~jointly_valid
    ] = np.nan

    return DifferenceResult(
        difference=fused_difference,
        valid_mask=jointly_valid,
    )


def deterministic_sample(
    values: np.ndarray,
    maximum_samples: int,
    seed: int,
) -> np.ndarray:
    """Select a deterministic sample from valid scalar values."""
    if maximum_samples <= 0:
        raise BaselineError(
            "maximum_samples must be greater than zero."
        )

    flattened = np.asarray(
        values,
        dtype=np.float32,
    ).reshape(-1)

    finite_values = flattened[
        np.isfinite(flattened)
    ]

    if finite_values.size == 0:
        raise BaselineError(
            "No finite difference pixels are available for sampling."
        )

    if finite_values.size <= maximum_samples:
        return finite_values

    generator = np.random.default_rng(
        seed
    )
    indices = generator.choice(
        finite_values.size,
        size=maximum_samples,
        replace=False,
    )

    return finite_values[
        indices
    ]


def calculate_otsu_threshold(
    values: np.ndarray,
    histogram_bins: int,
) -> float:
    """Calculate Otsu's threshold from scalar training differences.

    Otsu searches for the threshold that maximizes between-class variance
    in the one-dimensional difference histogram.
    """
    if histogram_bins < 2:
        raise BaselineError(
            "histogram_bins must be at least 2."
        )

    finite_values = np.asarray(
        values,
        dtype=np.float64,
    )
    finite_values = finite_values[
        np.isfinite(finite_values)
    ]

    if finite_values.size == 0:
        raise BaselineError(
            "No finite values were supplied to Otsu thresholding."
        )

    minimum = float(
        np.min(finite_values)
    )
    maximum = float(
        np.max(finite_values)
    )

    if maximum <= minimum:
        raise BaselineError(
            "Otsu thresholding requires a non-constant "
            "difference distribution."
        )

    histogram, edges = np.histogram(
        finite_values,
        bins=histogram_bins,
        range=(0.0, 1.0),
    )

    histogram = histogram.astype(
        np.float64
    )

    histogram_total = float(
        histogram.sum()
    )

    if histogram_total <= 0:
        raise BaselineError(
            "Otsu histogram contains no observations."
        )

    probabilities = (
        histogram
        / histogram_total
    )

    centers = (
        edges[:-1]
        + edges[1:]
    ) / 2.0

    cumulative_probability = np.cumsum(
        probabilities
    )
    cumulative_mean = np.cumsum(
        probabilities * centers
    )
    global_mean = float(
        cumulative_mean[-1]
    )

    denominator = (
        cumulative_probability
        * (
            1.0
            - cumulative_probability
        )
    )

    between_class_variance = np.full(
        denominator.shape,
        -np.inf,
        dtype=np.float64,
    )

    valid = denominator > 0

    between_class_variance[valid] = (
        (
            global_mean
            * cumulative_probability[valid]
            - cumulative_mean[valid]
        )
        ** 2
    ) / denominator[valid]

    best_index = int(
        np.argmax(
            between_class_variance
        )
    )

    threshold = float(
        edges[
            best_index + 1
        ]
    )

    if not 0.0 <= threshold <= 1.0:
        raise BaselineError(
            f"Calculated invalid Otsu threshold: {threshold}"
        )

    return threshold


def metrics_from_counts(
    counts: ConfusionCounts,
) -> ChangeMetrics:
    """Calculate change-class metrics from aggregated counts."""
    tp = counts.true_positive
    fp = counts.false_positive
    fn = counts.false_negative
    tn = counts.true_negative

    return ChangeMetrics(
        precision=safe_divide(
            tp,
            tp + fp,
            0.0,
        ),
        recall=safe_divide(
            tp,
            tp + fn,
            0.0,
        ),
        f1_score=safe_divide(
            2 * tp,
            2 * tp + fp + fn,
            0.0,
        ),
        iou=safe_divide(
            tp,
            tp + fp + fn,
            0.0,
        ),
        accuracy=safe_divide(
            tp + tn,
            counts.evaluated_pixels,
            0.0,
        ),
        change_prevalence=safe_divide(
            tp + fn,
            counts.evaluated_pixels,
            0.0,
        ),
        predicted_change_fraction=safe_divide(
            tp + fp,
            counts.evaluated_pixels,
            0.0,
        ),
        counts=counts,
    )


def aggregate_region_metrics(
    results: Sequence[RegionResult],
) -> ChangeMetrics:
    """Micro-average metrics by summing pixel confusion counts."""
    if not results:
        raise BaselineError(
            "Cannot aggregate an empty result collection."
        )

    counts = ConfusionCounts(
        true_positive=sum(
            result.true_positive
            for result in results
        ),
        false_positive=sum(
            result.false_positive
            for result in results
        ),
        false_negative=sum(
            result.false_negative
            for result in results
        ),
        true_negative=sum(
            result.true_negative
            for result in results
        ),
        ignored_pixels=sum(
            result.ignored_pixels
            for result in results
        ),
        evaluated_pixels=sum(
            result.evaluated_pixels
            for result in results
        ),
    )

    return metrics_from_counts(
        counts
    )


def save_prediction(
    prediction: np.ndarray,
    output_path: Path,
) -> None:
    """Save a binary prediction using 0/255 PNG encoding."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    encoded = (
        prediction.astype(
            np.uint8
        )
        * 255
    )

    temporary_path = output_path.with_suffix(
        ".tmp.png"
    )

    Image.fromarray(
        encoded,
        mode="L",
    ).save(
        temporary_path,
        format="PNG",
    )

    temporary_path.replace(
        output_path
    )


def save_difference_preview(
    difference: np.ndarray,
    valid_mask: np.ndarray,
    output_path: Path,
) -> None:
    """Save a display-only 8-bit difference preview."""
    preview = np.zeros(
        difference.shape,
        dtype=np.uint8,
    )

    scaled = np.clip(
        difference,
        0.0,
        1.0,
    )

    preview[
        valid_mask
    ] = np.round(
        255.0
        * scaled[
            valid_mask
        ]
    ).astype(
        np.uint8
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        ".tmp.png"
    )

    Image.fromarray(
        preview,
        mode="L",
    ).save(
        temporary_path,
        format="PNG",
    )

    temporary_path.replace(
        output_path
    )


def evaluate_region(
    region: str,
    split: str,
    images_root: Path,
    labels_root: Path,
    bands: Sequence[str],
    threshold: float,
    epsilon: float,
    output_directory: Path,
) -> RegionResult:
    """Generate and evaluate one OSCD region prediction."""
    difference_result = load_difference_image(
        images_root=images_root,
        region=region,
        bands=bands,
        epsilon=epsilon,
    )

    label_path = (
        labels_root
        / region
        / "cm"
        / "cm.png"
    )
    ground_truth = load_mask(
        label_path
    )

    if (
        ground_truth.shape
        != difference_result.difference.shape
    ):
        raise BaselineError(
            f"Difference and label shapes do not match for {region}: "
            f"{difference_result.difference.shape} versus "
            f"{ground_truth.shape}."
        )

    prediction = (
        difference_result.difference
        > threshold
    ) & difference_result.valid_mask

    ground_truth_for_evaluation = (
        ground_truth.astype(
            np.int16,
            copy=True,
        )
    )
    ground_truth_for_evaluation[
        ~difference_result.valid_mask
    ] = -1

    metrics = calculate_change_metrics(
        ground_truth=ground_truth_for_evaluation,
        prediction=prediction.astype(
            np.uint8
        ),
        ground_truth_change_values=(),
        prediction_change_values=(),
        ground_truth_ignore_values=(-1,),
        prediction_ignore_values=(),
        zero_division=0.0,
    )

    prediction_path = (
        output_directory
        / "predictions"
        / split
        / f"{region}.png"
    )
    preview_path = (
        output_directory
        / "difference_previews"
        / split
        / f"{region}.png"
    )

    save_prediction(
        prediction=prediction,
        output_path=prediction_path,
    )
    save_difference_preview(
        difference=difference_result.difference,
        valid_mask=difference_result.valid_mask,
        output_path=preview_path,
    )

    return RegionResult(
        region=region,
        split=split,
        threshold=threshold,
        width=int(
            ground_truth.shape[1]
        ),
        height=int(
            ground_truth.shape[0]
        ),
        evaluated_pixels=(
            metrics.counts.evaluated_pixels
        ),
        ignored_pixels=(
            metrics.counts.ignored_pixels
        ),
        true_positive=(
            metrics.counts.true_positive
        ),
        false_positive=(
            metrics.counts.false_positive
        ),
        false_negative=(
            metrics.counts.false_negative
        ),
        true_negative=(
            metrics.counts.true_negative
        ),
        precision=metrics.precision,
        recall=metrics.recall,
        f1_score=metrics.f1_score,
        iou=metrics.iou,
        accuracy=metrics.accuracy,
        change_prevalence=(
            metrics.change_prevalence
        ),
        predicted_change_fraction=(
            metrics.predicted_change_fraction
        ),
        prediction_path=str(
            prediction_path
        ),
        difference_preview_path=str(
            preview_path
        ),
    )


def write_csv_atomic(
    results: Sequence[RegionResult],
    output_path: Path,
) -> None:
    """Write per-region metrics atomically."""
    if not results:
        raise BaselineError(
            "Cannot write an empty result table."
        )

    rows = [
        result.to_dict()
        for result in results
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
    ) as file:
        writer = csv.DictWriter(
            file,
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
    """Write the baseline report atomically."""
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


def create_training_histogram(
    samples: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    """Create a diagnostic histogram of training differences."""
    figure, axis = plt.subplots(
        figsize=(9.0, 5.5)
    )

    axis.hist(
        samples,
        bins=256,
    )
    axis.axvline(
        threshold,
        linestyle="--",
        linewidth=2,
        label=(
            f"Otsu threshold = "
            f"{threshold:.6f}"
        ),
    )
    axis.set_title(
        "OSCD training difference distribution"
    )
    axis.set_xlabel(
        "Mean normalized absolute difference"
    )
    axis.set_ylabel(
        "Sampled pixel count"
    )
    axis.legend()
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


def metrics_payload(
    metrics: ChangeMetrics,
) -> dict[str, Any]:
    """Serialize aggregated positive-class metrics."""
    return {
        "metric_scope": "positive_change_class",
        "averaging": "micro_pixel_aggregation",
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1_score": metrics.f1_score,
        "iou": metrics.iou,
        "accuracy": metrics.accuracy,
        "accuracy_role": (
            "secondary_diagnostic_only"
        ),
        "change_prevalence": (
            metrics.change_prevalence
        ),
        "predicted_change_fraction": (
            metrics.predicted_change_fraction
        ),
        "counts": asdict(
            metrics.counts
        ),
    }


def run_baseline(
    oscd_root: Path,
    output_directory: Path,
    bands: Sequence[str],
    histogram_bins: int,
    samples_per_train_region: int,
    seed: int,
    epsilon: float,
) -> dict[str, Any]:
    """Fit the training-only Otsu threshold and evaluate OSCD."""
    paths = resolve_oscd_paths(
        oscd_root
    )

    train_regions = sorted(
        path.name
        for path in list_region_directories(
            paths.train_labels_root
        )
    )
    test_regions = sorted(
        path.name
        for path in list_region_directories(
            paths.test_labels_root
        )
    )

    if len(train_regions) != 14:
        raise BaselineError(
            f"Expected 14 training regions; "
            f"found {len(train_regions)}."
        )

    if len(test_regions) != 10:
        raise BaselineError(
            f"Expected 10 test regions; "
            f"found {len(test_regions)}."
        )

    normalized_bands = tuple(
        str(band).strip().upper()
        for band in bands
    )

    if len(set(normalized_bands)) != len(
        normalized_bands
    ):
        raise BaselineError(
            "Input bands must be unique."
        )

    print(
        "Building training-only "
        "difference distribution"
    )

    training_samples: list[np.ndarray] = []

    for region in train_regions:
        difference_result = load_difference_image(
            images_root=paths.images_root,
            region=region,
            bands=normalized_bands,
            epsilon=epsilon,
        )

        sample = deterministic_sample(
            values=difference_result.difference[
                difference_result.valid_mask
            ],
            maximum_samples=(
                samples_per_train_region
            ),
            seed=stable_region_seed(
                base_seed=seed,
                region=region,
            ),
        )
        training_samples.append(
            sample
        )

        print(
            f"  [threshold sample] {region}: "
            f"{sample.size} pixels"
        )

    pooled_samples = np.concatenate(
        training_samples
    )

    threshold = calculate_otsu_threshold(
        values=pooled_samples,
        histogram_bins=histogram_bins,
    )

    print(
        f"Frozen training-only Otsu threshold: "
        f"{threshold:.8f}"
    )

    results: list[RegionResult] = []

    for (
        split,
        regions,
        labels_root,
    ) in (
        (
            "train",
            train_regions,
            paths.train_labels_root,
        ),
        (
            "test",
            test_regions,
            paths.test_labels_root,
        ),
    ):
        for region in regions:
            result = evaluate_region(
                region=region,
                split=split,
                images_root=paths.images_root,
                labels_root=labels_root,
                bands=normalized_bands,
                threshold=threshold,
                epsilon=epsilon,
                output_directory=(
                    output_directory
                ),
            )
            results.append(result)

            print(
                f"  [{split}] {region}: "
                f"F1={result.f1_score:.4f}, "
                f"IoU={result.iou:.4f}, "
                f"predicted-change="
                f"{result.predicted_change_fraction:.2%}"
            )

    train_results = [
        result
        for result in results
        if result.split == "train"
    ]
    test_results = [
        result
        for result in results
        if result.split == "test"
    ]

    train_metrics = aggregate_region_metrics(
        train_results
    )
    test_metrics = aggregate_region_metrics(
        test_results
    )

    region_metrics_path = (
        output_directory
        / "oscd_region_metrics.csv"
    )
    report_path = (
        output_directory
        / "band_diff_otsu_report.json"
    )
    histogram_path = (
        output_directory
        / "training_difference_histogram.png"
    )

    write_csv_atomic(
        results=results,
        output_path=region_metrics_path,
    )
    create_training_histogram(
        samples=pooled_samples,
        threshold=threshold,
        output_path=histogram_path,
    )

    report = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "success",
        "baseline": "band_difference_otsu",
        "dataset": "OSCD",
        "quantitative_scope": (
            "OSCD labelled regions only"
        ),
        "custom_hyderabad_aoi_used_for_metrics": False,
        "method": {
            "bands": list(
                normalized_bands
            ),
            "difference": (
                "abs(after-before) / "
                "(abs(after)+abs(before)+epsilon)"
            ),
            "band_fusion": (
                "mean across selected bands"
            ),
            "threshold_method": (
                "global Otsu"
            ),
            "threshold_value": threshold,
            "threshold_source": (
                "OSCD training image pixels only"
            ),
            "training_labels_used_for_threshold": False,
            "test_images_used_for_threshold": False,
            "test_labels_used_for_threshold": False,
            "morphological_postprocessing": False,
            "epsilon": epsilon,
            "histogram_bins": histogram_bins,
            "samples_per_train_region": (
                samples_per_train_region
            ),
            "random_seed": seed,
            "sampled_training_pixels": int(
                pooled_samples.size
            ),
        },
        "region_counts": {
            "train": len(
                train_results
            ),
            "test": len(
                test_results
            ),
            "overall": len(
                results
            ),
        },
        "train_metrics": metrics_payload(
            train_metrics
        ),
        "test_metrics": metrics_payload(
            test_metrics
        ),
        "primary_reported_result": {
            "split": "OSCD test",
            "averaging": (
                "micro pixel aggregation"
            ),
            "precision": (
                test_metrics.precision
            ),
            "recall": (
                test_metrics.recall
            ),
            "f1_score": (
                test_metrics.f1_score
            ),
            "iou": test_metrics.iou,
        },
        "outputs": {
            "region_metrics_csv": str(
                region_metrics_path
            ),
            "training_histogram": str(
                histogram_path
            ),
            "prediction_directory": str(
                output_directory
                / "predictions"
            ),
            "difference_preview_directory": str(
                output_directory
                / "difference_previews"
            ),
        },
    }

    write_json_atomic(
        payload=report,
        output_path=report_path,
    )

    return {
        **report,
        "report_path": str(
            report_path
        ),
    }


def print_summary(
    report: Mapping[str, Any],
) -> None:
    """Print the final baseline result."""
    train = report["train_metrics"]
    test = report["test_metrics"]

    print(
        "Band-difference + Otsu baseline completed"
    )
    print(
        "  Status:",
        report["status"],
    )
    print(
        "  Threshold:",
        f"{report['method']['threshold_value']:.8f}",
    )
    print(
        "  Threshold source:",
        report["method"]["threshold_source"],
    )
    print(
        "  Train F1:",
        f"{train['f1_score']:.6f}",
    )
    print(
        "  Train IoU:",
        f"{train['iou']:.6f}",
    )
    print(
        "  Test precision:",
        f"{test['precision']:.6f}",
    )
    print(
        "  Test recall:",
        f"{test['recall']:.6f}",
    )
    print(
        "  Test F1:",
        f"{test['f1_score']:.6f}",
    )
    print(
        "  Test IoU:",
        f"{test['iou']:.6f}",
    )
    print(
        "  Test accuracy:",
        f"{test['accuracy']:.6f}",
        "(secondary diagnostic)",
    )
    print(
        "  Report:",
        report["report_path"],
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the baseline command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a multispectral band-difference and "
            "training-only global Otsu baseline on OSCD."
        )
    )

    parser.add_argument(
        "--oscd-root",
        type=Path,
        default=Path(
            "data/benchmark/oscd/raw"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "reports/week2/baselines/"
            "band_diff_otsu"
        ),
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        default=[
            "B02",
            "B03",
            "B04",
            "B08",
        ],
    )
    parser.add_argument(
        "--histogram-bins",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--samples-per-train-region",
        type=int,
        default=250_000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-6,
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
    """Run the band-difference and Otsu baseline."""
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
        report = run_baseline(
            oscd_root=args.oscd_root,
            output_directory=args.output_dir,
            bands=args.bands,
            histogram_bins=args.histogram_bins,
            samples_per_train_region=(
                args.samples_per_train_region
            ),
            seed=args.seed,
            epsilon=args.epsilon,
        )

        print_summary(
            report
        )

        return 0

    except (
        BaselineError,
        FileNotFoundError,
        PermissionError,
        ValueError,
        TypeError,
        OSError,
        csv.Error,
        json.JSONDecodeError,
    ) as error:
        LOGGER.error(
            "%s",
            error,
        )
        return 1

    except Exception:
        LOGGER.exception(
            "Unexpected band-difference baseline failure."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
