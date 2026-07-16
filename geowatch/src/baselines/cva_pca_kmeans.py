"""CVA, PCA and K-Means classical baseline for GeoWatch.

The model is fitted using OSCD training images only:

1. Construct normalized signed change vectors from before/after bands.
2. Standardize the training change vectors.
3. Fit PCA on sampled training pixels.
4. Fit a two-cluster K-Means model in PCA space.
5. Identify the change cluster using mean original CVA magnitude.
6. Freeze the complete pipeline before evaluating OSCD test regions.

Ground-truth labels are never used to fit the scaler, PCA, K-Means model,
or to identify the change cluster. GeoWatch's Hyderabad AOI is excluded
from all quantitative metrics.
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
    "geowatch.cva_pca_kmeans"
)


class CVABaselineError(RuntimeError):
    """Raised when the CVA baseline cannot be completed safely."""


@dataclass(frozen=True)
class CVAImage:
    """CVA features, magnitude and valid-pixel mask for one region."""

    vectors: np.ndarray
    magnitude: np.ndarray
    valid_mask: np.ndarray


@dataclass(frozen=True)
class RegionResult:
    """Metrics and output artifacts for one OSCD region."""

    region: str
    split: str
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
    magnitude_preview_path: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible result dictionary."""
        return asdict(self)


def stable_region_seed(
    base_seed: int,
    region: str,
) -> int:
    """Create a deterministic random seed for one region."""
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


def load_cva_image(
    images_root: Path,
    region: str,
    bands: Sequence[str],
    epsilon: float,
) -> CVAImage:
    """Load a normalized signed change-vector image.

    Each selected band is converted to:

        (after - before) /
        (abs(after) + abs(before) + epsilon)

    Unlike an absolute-difference image, this preserves the direction of
    spectral change. PCA can therefore learn correlated positive and negative
    change directions before clustering.
    """
    if not bands:
        raise CVABaselineError(
            "At least one Sentinel-2 band must be provided."
        )

    if epsilon <= 0:
        raise CVABaselineError(
            "epsilon must be greater than zero."
        )

    image_region = images_root / region
    before_directory = image_region / "imgs_1_rect"
    after_directory = image_region / "imgs_2_rect"

    vectors: list[np.ndarray] = []
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
            raise CVABaselineError(
                f"Before/after shapes differ for {region} "
                f"{band_name}: {before.shape} versus {after.shape}."
            )

        if expected_shape is None:
            expected_shape = before.shape
        elif before.shape != expected_shape:
            raise CVABaselineError(
                f"Selected bands do not share one shape for "
                f"{region}. Expected {expected_shape}, received "
                f"{before.shape} for {band_name}."
            )

        finite = (
            np.isfinite(before)
            & np.isfinite(after)
        )
        nonempty = (
            (before != 0)
            | (after != 0)
        )
        valid = finite & nonempty

        denominator = (
            np.abs(after)
            + np.abs(before)
            + epsilon
        )

        signed_difference = np.divide(
            after - before,
            denominator,
            out=np.zeros_like(
                after,
                dtype=np.float32,
            ),
            where=valid,
        )

        signed_difference = np.clip(
            signed_difference,
            -1.0,
            1.0,
        ).astype(
            np.float32,
            copy=False,
        )

        vectors.append(
            signed_difference
        )
        validity_masks.append(
            valid
        )

    vector_cube = np.stack(
        vectors,
        axis=-1,
    )
    valid_stack = np.stack(
        validity_masks,
        axis=0,
    )
    jointly_valid = np.all(
        valid_stack,
        axis=0,
    )

    magnitude = np.linalg.norm(
        vector_cube,
        axis=-1,
    ).astype(
        np.float32,
        copy=False,
    )

    vector_cube[
        ~jointly_valid
    ] = 0.0
    magnitude[
        ~jointly_valid
    ] = np.nan

    return CVAImage(
        vectors=vector_cube,
        magnitude=magnitude,
        valid_mask=jointly_valid,
    )


def sample_rows(
    values: np.ndarray,
    maximum_samples: int,
    seed: int,
) -> np.ndarray:
    """Select deterministic rows from a two-dimensional feature matrix."""
    if values.ndim != 2:
        raise CVABaselineError(
            f"Expected a two-dimensional feature matrix, "
            f"received shape {values.shape}."
        )

    if maximum_samples <= 0:
        raise CVABaselineError(
            "maximum_samples must be greater than zero."
        )

    if values.shape[0] == 0:
        raise CVABaselineError(
            "No valid CVA pixels are available for sampling."
        )

    if values.shape[0] <= maximum_samples:
        return values.astype(
            np.float32,
            copy=False,
        )

    generator = np.random.default_rng(
        seed
    )
    indices = generator.choice(
        values.shape[0],
        size=maximum_samples,
        replace=False,
    )

    return values[
        indices
    ].astype(
        np.float32,
        copy=False,
    )


def metrics_from_counts(
    counts: ConfusionCounts,
) -> ChangeMetrics:
    """Calculate positive-class metrics from aggregate confusion counts."""
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


def aggregate_metrics(
    results: Sequence[RegionResult],
) -> ChangeMetrics:
    """Micro-average metrics across multiple OSCD regions."""
    if not results:
        raise CVABaselineError(
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


def save_binary_prediction(
    prediction: np.ndarray,
    output_path: Path,
) -> None:
    """Save a binary change prediction as a 0/255 PNG."""
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


def save_magnitude_preview(
    magnitude: np.ndarray,
    valid_mask: np.ndarray,
    output_path: Path,
) -> None:
    """Save a percentile-stretched CVA magnitude preview."""
    valid_values = magnitude[
        valid_mask
        & np.isfinite(magnitude)
    ]

    if valid_values.size == 0:
        raise CVABaselineError(
            "No valid magnitude values are available for preview."
        )

    upper = float(
        np.percentile(
            valid_values,
            98.0,
        )
    )

    if upper <= 0:
        upper = 1.0

    preview = np.zeros(
        magnitude.shape,
        dtype=np.uint8,
    )

    preview[
        valid_mask
    ] = np.round(
        255.0
        * np.clip(
            magnitude[
                valid_mask
            ] / upper,
            0.0,
            1.0,
        )
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
    epsilon: float,
    scaler: StandardScaler,
    pca: PCA,
    kmeans: KMeans,
    change_cluster: int,
    output_directory: Path,
) -> RegionResult:
    """Predict and evaluate one OSCD region using the frozen pipeline."""
    cva_image = load_cva_image(
        images_root=images_root,
        region=region,
        bands=bands,
        epsilon=epsilon,
    )

    valid_vectors = cva_image.vectors[
        cva_image.valid_mask
    ]

    standardized = scaler.transform(
        valid_vectors
    )
    projected = pca.transform(
        standardized
    )
    cluster_labels = kmeans.predict(
        projected
    )

    prediction = np.zeros(
        cva_image.valid_mask.shape,
        dtype=bool,
    )
    prediction[
        cva_image.valid_mask
    ] = (
        cluster_labels
        == change_cluster
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

    if ground_truth.shape != prediction.shape:
        raise CVABaselineError(
            f"Prediction and label shapes differ for {region}: "
            f"{prediction.shape} versus {ground_truth.shape}."
        )

    evaluation_ground_truth = ground_truth.astype(
        np.int16,
        copy=True,
    )
    evaluation_ground_truth[
        ~cva_image.valid_mask
    ] = -1

    metrics = calculate_change_metrics(
        ground_truth=evaluation_ground_truth,
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
    magnitude_path = (
        output_directory
        / "magnitude_previews"
        / split
        / f"{region}.png"
    )

    save_binary_prediction(
        prediction,
        prediction_path,
    )
    save_magnitude_preview(
        magnitude=cva_image.magnitude,
        valid_mask=cva_image.valid_mask,
        output_path=magnitude_path,
    )

    return RegionResult(
        region=region,
        split=split,
        width=int(
            prediction.shape[1]
        ),
        height=int(
            prediction.shape[0]
        ),
        evaluated_pixels=metrics.counts.evaluated_pixels,
        ignored_pixels=metrics.counts.ignored_pixels,
        true_positive=metrics.counts.true_positive,
        false_positive=metrics.counts.false_positive,
        false_negative=metrics.counts.false_negative,
        true_negative=metrics.counts.true_negative,
        precision=metrics.precision,
        recall=metrics.recall,
        f1_score=metrics.f1_score,
        iou=metrics.iou,
        accuracy=metrics.accuracy,
        change_prevalence=metrics.change_prevalence,
        predicted_change_fraction=(
            metrics.predicted_change_fraction
        ),
        prediction_path=str(
            prediction_path
        ),
        magnitude_preview_path=str(
            magnitude_path
        ),
    )


def write_csv_atomic(
    results: Sequence[RegionResult],
    output_path: Path,
) -> None:
    """Write per-region metrics atomically."""
    if not results:
        raise CVABaselineError(
            "Cannot write an empty metrics table."
        )

    rows = [
        result.to_dict()
        for result in results
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
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(
            rows
        )

    temporary_path.replace(
        output_path
    )


def write_json_atomic(
    payload: Mapping[str, Any],
    output_path: Path,
) -> None:
    """Write a JSON report atomically."""
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


def create_pca_cluster_figure(
    projected_samples: np.ndarray,
    cluster_labels: np.ndarray,
    output_path: Path,
    maximum_points: int,
    seed: int,
) -> None:
    """Create a diagnostic PC1/PC2 cluster projection."""
    if projected_samples.shape[1] < 2:
        raise CVABaselineError(
            "At least two PCA components are required for plotting."
        )

    count = projected_samples.shape[0]

    if count > maximum_points:
        generator = np.random.default_rng(
            seed
        )
        indices = generator.choice(
            count,
            size=maximum_points,
            replace=False,
        )
    else:
        indices = np.arange(
            count
        )

    figure, axis = plt.subplots(
        figsize=(8.0, 6.5)
    )

    scatter = axis.scatter(
        projected_samples[
            indices,
            0,
        ],
        projected_samples[
            indices,
            1,
        ],
        c=cluster_labels[
            indices
        ],
        s=4,
        alpha=0.35,
    )

    axis.set_title(
        "OSCD training CVA clusters in PCA space"
    )
    axis.set_xlabel(
        "Principal component 1"
    )
    axis.set_ylabel(
        "Principal component 2"
    )
    axis.grid(
        alpha=0.25,
    )

    figure.colorbar(
        scatter,
        ax=axis,
        label="K-Means cluster",
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
    plt.close(
        figure
    )


def metrics_payload(
    metrics: ChangeMetrics,
) -> dict[str, Any]:
    """Serialize aggregate positive-class metrics."""
    return {
        "metric_scope": "positive_change_class",
        "averaging": "micro_pixel_aggregation",
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1_score": metrics.f1_score,
        "iou": metrics.iou,
        "accuracy": metrics.accuracy,
        "accuracy_role": "secondary_diagnostic_only",
        "change_prevalence": metrics.change_prevalence,
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
    pca_components: int,
    samples_per_train_region: int,
    seed: int,
    epsilon: float,
    kmeans_n_init: int,
) -> dict[str, Any]:
    """Fit CVA, PCA and K-Means on training images and evaluate OSCD."""
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
        raise CVABaselineError(
            f"Expected 14 training regions; "
            f"found {len(train_regions)}."
        )

    if len(test_regions) != 10:
        raise CVABaselineError(
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
        raise CVABaselineError(
            "Input bands must be unique."
        )

    if not (
        2
        <= pca_components
        <= len(normalized_bands)
    ):
        raise CVABaselineError(
            "pca_components must be between 2 and "
            "the number of selected bands."
        )

    if kmeans_n_init <= 0:
        raise CVABaselineError(
            "kmeans_n_init must be greater than zero."
        )

    print(
        "Sampling training-only CVA vectors"
    )

    sampled_vectors: list[np.ndarray] = []

    for region in train_regions:
        cva_image = load_cva_image(
            images_root=paths.images_root,
            region=region,
            bands=normalized_bands,
            epsilon=epsilon,
        )

        region_vectors = cva_image.vectors[
            cva_image.valid_mask
        ]

        sample = sample_rows(
            values=region_vectors,
            maximum_samples=(
                samples_per_train_region
            ),
            seed=stable_region_seed(
                seed,
                region,
            ),
        )

        sampled_vectors.append(
            sample
        )

        print(
            f"  [training sample] {region}: "
            f"{sample.shape[0]} vectors"
        )

    training_vectors = np.concatenate(
        sampled_vectors,
        axis=0,
    )

    scaler = StandardScaler(
        copy=True,
    )
    standardized_vectors = scaler.fit_transform(
        training_vectors
    )

    pca = PCA(
        n_components=pca_components,
        svd_solver="full",
    )
    projected_vectors = pca.fit_transform(
        standardized_vectors
    )

    kmeans = KMeans(
        n_clusters=2,
        random_state=seed,
        n_init=kmeans_n_init,
        max_iter=300,
        algorithm="lloyd",
    )
    training_clusters = kmeans.fit_predict(
        projected_vectors
    )

    original_magnitude = np.linalg.norm(
        training_vectors,
        axis=1,
    )

    cluster_magnitude_means: list[float] = []

    for cluster_index in range(2):
        cluster_members = (
            training_clusters
            == cluster_index
        )

        if not np.any(
            cluster_members
        ):
            raise CVABaselineError(
                f"K-Means produced empty cluster {cluster_index}."
            )

        cluster_magnitude_means.append(
            float(
                np.mean(
                    original_magnitude[
                        cluster_members
                    ]
                )
            )
        )

    change_cluster = int(
        np.argmax(
            cluster_magnitude_means
        )
    )

    print(
        "Frozen change cluster:",
        change_cluster,
    )
    print(
        "Cluster mean CVA magnitudes:",
        [
            round(value, 8)
            for value in cluster_magnitude_means
        ],
    )
    print(
        "PCA explained variance:",
        [
            round(
                float(value),
                8,
            )
            for value in pca.explained_variance_ratio_
        ],
    )

    results: list[RegionResult] = []

    for split, regions, labels_root in (
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
                epsilon=epsilon,
                scaler=scaler,
                pca=pca,
                kmeans=kmeans,
                change_cluster=change_cluster,
                output_directory=output_directory,
            )

            results.append(
                result
            )

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

    train_metrics = aggregate_metrics(
        train_results
    )
    test_metrics = aggregate_metrics(
        test_results
    )

    metrics_path = (
        output_directory
        / "oscd_region_metrics.csv"
    )
    report_path = (
        output_directory
        / "cva_pca_kmeans_report.json"
    )
    figure_path = (
        output_directory
        / "training_pca_clusters.png"
    )

    write_csv_atomic(
        results=results,
        output_path=metrics_path,
    )

    create_pca_cluster_figure(
        projected_samples=projected_vectors,
        cluster_labels=training_clusters,
        output_path=figure_path,
        maximum_points=50_000,
        seed=seed,
    )

    report = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "success",
        "baseline": "cva_pca_kmeans",
        "dataset": "OSCD",
        "quantitative_scope": (
            "OSCD labelled regions only"
        ),
        "custom_hyderabad_aoi_used_for_metrics": False,
        "method": {
            "bands": list(
                normalized_bands
            ),
            "change_vector": (
                "(after-before) / "
                "(abs(after)+abs(before)+epsilon)"
            ),
            "standardization": (
                "StandardScaler fitted on training samples only"
            ),
            "pca_components": pca_components,
            "pca_fitted_on": (
                "OSCD training image pixels only"
            ),
            "pca_explained_variance_ratio": [
                float(value)
                for value in pca.explained_variance_ratio_
            ],
            "clustering": "K-Means with two clusters",
            "kmeans_fitted_on": (
                "OSCD training image pixels only"
            ),
            "change_cluster_rule": (
                "cluster with larger mean original CVA magnitude"
            ),
            "change_cluster": change_cluster,
            "cluster_mean_cva_magnitude": (
                cluster_magnitude_means
            ),
            "training_labels_used_for_fitting": False,
            "test_images_used_for_fitting": False,
            "test_labels_used_for_fitting": False,
            "samples_per_train_region": (
                samples_per_train_region
            ),
            "sampled_training_vectors": int(
                training_vectors.shape[0]
            ),
            "random_seed": seed,
            "kmeans_n_init": kmeans_n_init,
            "epsilon": epsilon,
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
            "averaging": "micro pixel aggregation",
            "precision": test_metrics.precision,
            "recall": test_metrics.recall,
            "f1_score": test_metrics.f1_score,
            "iou": test_metrics.iou,
        },
        "outputs": {
            "region_metrics_csv": str(
                metrics_path
            ),
            "training_pca_figure": str(
                figure_path
            ),
            "prediction_directory": str(
                output_directory
                / "predictions"
            ),
            "magnitude_preview_directory": str(
                output_directory
                / "magnitude_previews"
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
    """Print the final baseline summary."""
    train = report["train_metrics"]
    test = report["test_metrics"]

    print("CVA + PCA + K-Means baseline completed")
    print("  Status:", report["status"])
    print(
        "  Change cluster:",
        report["method"]["change_cluster"],
    )
    print(
        "  PCA explained variance:",
        [
            round(value, 6)
            for value in report["method"][
                "pca_explained_variance_ratio"
            ]
        ],
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
    """Build the CVA baseline CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Fit a training-only CVA, PCA and K-Means "
            "change-detection baseline and evaluate it on OSCD."
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
            "cva_pca_kmeans"
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
        "--pca-components",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--samples-per-train-region",
        type=int,
        default=25_000,
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
        "--kmeans-n-init",
        type=int,
        default=10,
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
    """Run the CVA, PCA and K-Means baseline."""
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
            pca_components=args.pca_components,
            samples_per_train_region=(
                args.samples_per_train_region
            ),
            seed=args.seed,
            epsilon=args.epsilon,
            kmeans_n_init=args.kmeans_n_init,
        )

        print_summary(
            report
        )

        return 0

    except (
        CVABaselineError,
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
            "Unexpected CVA baseline failure."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
