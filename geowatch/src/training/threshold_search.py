"""Validation-only probability-threshold search for GeoWatch."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from src.training.train import (
    TrainingRuntime,
    build_training_runtime,
    load_training_checkpoint,
    load_training_config,
    move_batch_to_device,
    require_mapping,
)


EXPECTED_VALIDATION_REGIONS = frozenset(
    {
        "hongkong",
        "mumbai",
        "paris",
    }
)


@dataclass(frozen=True)
class ThresholdMetrics:
    """Positive-change metrics for one probability threshold."""

    threshold: float
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    precision: float
    recall: float
    f1: float
    iou: float
    accuracy: float


def safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    """Divide two values while returning zero for an empty denominator."""

    if denominator == 0.0:
        return 0.0

    return numerator / denominator


def build_threshold_grid(
    minimum: float,
    maximum: float,
    step: float,
) -> tuple[float, ...]:
    """Build a deterministic inclusive threshold grid."""

    if not 0.0 <= minimum < maximum <= 1.0:
        raise ValueError(
            "Threshold bounds must satisfy "
            "0 <= minimum < maximum <= 1."
        )

    if step <= 0.0:
        raise ValueError(
            "Threshold step must be positive."
        )

    count = int(
        np.floor(
            (maximum - minimum)
            / step
            + 1.0e-9
        )
    )

    thresholds = tuple(
        round(
            minimum + index * step,
            10,
        )
        for index in range(count + 1)
    )

    if not thresholds:
        raise ValueError(
            "Threshold grid cannot be empty."
        )

    return thresholds


def validate_prediction_arrays(
    probabilities: NDArray[np.float32],
    targets: NDArray[np.uint8],
) -> None:
    """Validate flattened probabilities and binary targets."""

    if probabilities.ndim != 1:
        raise ValueError(
            "Probabilities must be one-dimensional."
        )

    if targets.ndim != 1:
        raise ValueError(
            "Targets must be one-dimensional."
        )

    if probabilities.shape != targets.shape:
        raise ValueError(
            "Probabilities and targets must have identical shapes."
        )

    if probabilities.size == 0:
        raise ValueError(
            "Validation predictions cannot be empty."
        )

    if not np.isfinite(
        probabilities
    ).all():
        raise ValueError(
            "Probabilities must be finite."
        )

    if (
        probabilities.min() < 0.0
        or probabilities.max() > 1.0
    ):
        raise ValueError(
            "Probabilities must be between zero and one."
        )

    unique_targets = set(
        np.unique(
            targets
        ).tolist()
    )

    if not unique_targets.issubset(
        {
            0,
            1,
        }
    ):
        raise ValueError(
            "Targets must be binary."
        )


def calculate_threshold_metrics(
    probabilities: NDArray[np.float32],
    targets: NDArray[np.uint8],
    threshold: float,
) -> ThresholdMetrics:
    """Calculate aggregate positive-change metrics at one threshold."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "Threshold must be between zero and one."
        )

    predictions = probabilities >= threshold
    truth = targets.astype(
        np.bool_,
        copy=False,
    )

    true_positive = int(
        np.count_nonzero(
            predictions
            & truth
        )
    )
    false_positive = int(
        np.count_nonzero(
            predictions
            & ~truth
        )
    )
    false_negative = int(
        np.count_nonzero(
            ~predictions
            & truth
        )
    )
    true_negative = int(
        np.count_nonzero(
            ~predictions
            & ~truth
        )
    )

    precision = safe_divide(
        true_positive,
        true_positive + false_positive,
    )
    recall = safe_divide(
        true_positive,
        true_positive + false_negative,
    )
    f1 = safe_divide(
        2.0 * true_positive,
        2.0 * true_positive
        + false_positive
        + false_negative,
    )
    iou = safe_divide(
        true_positive,
        true_positive
        + false_positive
        + false_negative,
    )
    accuracy = safe_divide(
        true_positive + true_negative,
        probabilities.size,
    )

    return ThresholdMetrics(
        threshold=float(
            threshold
        ),
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        true_negative=true_negative,
        precision=precision,
        recall=recall,
        f1=f1,
        iou=iou,
        accuracy=accuracy,
    )


def select_best_threshold(
    results: Sequence[ThresholdMetrics],
    objective: str,
) -> ThresholdMetrics:
    """Select the best threshold using deterministic tie-breaking."""

    if not results:
        raise ValueError(
            "Threshold results cannot be empty."
        )

    if objective not in {
        "f1",
        "iou",
    }:
        raise ValueError(
            "Objective must be either 'f1' or 'iou'."
        )

    def ranking_key(
        result: ThresholdMetrics,
    ) -> tuple[
        float,
        float,
        float,
        float,
    ]:
        primary = (
            result.f1
            if objective == "f1"
            else result.iou
        )
        secondary = (
            result.iou
            if objective == "f1"
            else result.f1
        )

        return (
            primary,
            secondary,
            result.precision,
            -abs(
                result.threshold - 0.5
            ),
        )

    return max(
        results,
        key=ranking_key,
    )


def validate_evaluation_protocol(
    config: Mapping[str, Any],
    runtime: TrainingRuntime,
) -> tuple[str, ...]:
    """Validate the sealed validation-only evaluation contract."""

    protocol = require_mapping(
        config.get(
            "protocol"
        ),
        "protocol",
    )
    dataset = require_mapping(
        config.get(
            "dataset"
        ),
        "dataset",
    )

    if protocol.get(
        "official_test_regions_sealed"
    ) is not True:
        raise ValueError(
            "Official test regions must remain sealed."
        )

    configured_validation_regions = tuple(
        str(region).strip().lower()
        for region in dataset[
            "validation_regions"
        ]
    )

    configured_training_regions = tuple(
        str(region).strip().lower()
        for region in dataset[
            "train_regions"
        ]
    )

    if (
        frozenset(
            configured_validation_regions
        )
        != EXPECTED_VALIDATION_REGIONS
    ):
        raise ValueError(
            "Threshold search is restricted to "
            "Hong Kong, Mumbai and Paris."
        )

    if set(
        configured_training_regions
    ).intersection(
        configured_validation_regions
    ):
        raise ValueError(
            "Training and validation regions overlap."
        )

    runtime_validation_regions = tuple(
        str(region).strip().lower()
        for region
        in runtime.validation_dataset.region_names
    )

    if (
        frozenset(
            runtime_validation_regions
        )
        != EXPECTED_VALIDATION_REGIONS
    ):
        raise ValueError(
            "Runtime validation regions do not match "
            "the frozen Week 5 protocol."
        )

    return configured_validation_regions


def collect_validation_predictions(
    runtime: TrainingRuntime,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.uint8],
]:
    """Collect flattened validation probabilities and targets."""

    probability_batches: list[
        NDArray[np.float32]
    ] = []
    target_batches: list[
        NDArray[np.uint8]
    ] = []

    runtime.model.eval()

    with torch.inference_mode():
        for batch in runtime.validation_loader:
            before, after, targets = move_batch_to_device(
                batch=batch,
                device=runtime.device,
            )

            with torch.autocast(
                device_type=runtime.device.type,
                enabled=runtime.mixed_precision_enabled,
            ):
                logits: Tensor = runtime.model(
                    before,
                    after,
                )

            probabilities = (
                torch.sigmoid(
                    logits
                )
                .to(
                    dtype=torch.float32
                )
                .detach()
                .cpu()
                .reshape(
                    -1
                )
                .numpy()
            )

            binary_targets = (
                (
                    targets > 0
                )
                .to(
                    dtype=torch.uint8
                )
                .detach()
                .cpu()
                .reshape(
                    -1
                )
                .numpy()
            )

            probability_batches.append(
                probabilities
            )
            target_batches.append(
                binary_targets
            )

    all_probabilities = np.concatenate(
        probability_batches
    ).astype(
        np.float32,
        copy=False,
    )
    all_targets = np.concatenate(
        target_batches
    ).astype(
        np.uint8,
        copy=False,
    )

    validate_prediction_arrays(
        probabilities=all_probabilities,
        targets=all_targets,
    )

    return (
        all_probabilities,
        all_targets,
    )


def calculate_file_sha256(
    path: Path,
) -> str:
    """Calculate the SHA-256 digest of one file."""

    return sha256(
        path.read_bytes()
    ).hexdigest()


def write_threshold_csv(
    path: Path,
    results: Sequence[ThresholdMetrics],
) -> None:
    """Write the complete threshold metric curve."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        asdict(
            results[0]
        ).keys()
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
            lineterminator="\\n",
        )
        writer.writeheader()

        for result in results:
            writer.writerow(
                asdict(
                    result
                )
            )


def write_summary_json(
    path: Path,
    summary: Mapping[str, Any],
) -> None:
    """Write the threshold-search summary as formatted JSON."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the threshold-search command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Select a GeoWatch probability threshold "
            "using only the frozen validation regions."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--minimum-threshold",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--maximum-threshold",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--objective",
        choices=(
            "f1",
            "iou",
        ),
        default="f1",
    )
    parser.add_argument(
        "--device",
        choices=(
            "auto",
            "cpu",
            "cuda",
        ),
        default="cuda",
    )
    parser.add_argument(
        "--validation-batch-size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
    )

    return parser


def main() -> None:
    """Execute validation-only threshold selection."""

    arguments = build_argument_parser().parse_args()

    config = load_training_config(
        arguments.config
    )

    runtime = build_training_runtime(
        config=config,
        device_override=arguments.device,
        validation_batch_size_override=(
            arguments.validation_batch_size
        ),
        num_workers_override=arguments.num_workers,
        disable_pretrained=True,
    )

    resume = load_training_checkpoint(
        path=arguments.checkpoint,
        runtime=runtime,
        strict_config=True,
        restore_rng=False,
    )

    validation_regions = validate_evaluation_protocol(
        config=config,
        runtime=runtime,
    )

    probabilities, targets = collect_validation_predictions(
        runtime
    )

    thresholds = build_threshold_grid(
        minimum=arguments.minimum_threshold,
        maximum=arguments.maximum_threshold,
        step=arguments.step,
    )

    results = tuple(
        calculate_threshold_metrics(
            probabilities=probabilities,
            targets=targets,
            threshold=threshold,
        )
        for threshold in thresholds
    )

    best_result = select_best_threshold(
        results=results,
        objective=arguments.objective,
    )

    metrics_config = require_mapping(
        config.get(
            "metrics"
        ),
        "metrics",
    )

    default_threshold = float(
        metrics_config[
            "threshold"
        ]
    )

    default_result = calculate_threshold_metrics(
        probabilities=probabilities,
        targets=targets,
        threshold=default_threshold,
    )

    csv_path = (
        arguments.output_dir
        / "threshold_metrics.csv"
    )
    summary_path = (
        arguments.output_dir
        / "threshold_search_summary.json"
    )

    write_threshold_csv(
        path=csv_path,
        results=results,
    )

    summary = {
        "protocol": {
            "validation_regions": list(
                validation_regions
            ),
            "official_test_regions_accessed": False,
            "official_test_labels_accessed": False,
        },
        "checkpoint": {
            "path": str(
                arguments.checkpoint
            ),
            "epoch": resume.next_epoch - 1,
            "sha256": calculate_file_sha256(
                arguments.checkpoint
            ),
        },
        "search": {
            "objective": arguments.objective,
            "minimum_threshold": arguments.minimum_threshold,
            "maximum_threshold": arguments.maximum_threshold,
            "step": arguments.step,
            "evaluated_thresholds": len(
                results
            ),
            "validation_pixels": int(
                probabilities.size
            ),
            "positive_validation_pixels": int(
                targets.sum()
            ),
        },
        "default_threshold_metrics": asdict(
            default_result
        ),
        "best_threshold_metrics": asdict(
            best_result
        ),
        "improvement_from_default": {
            "precision": (
                best_result.precision
                - default_result.precision
            ),
            "recall": (
                best_result.recall
                - default_result.recall
            ),
            "f1": (
                best_result.f1
                - default_result.f1
            ),
            "iou": (
                best_result.iou
                - default_result.iou
            ),
        },
    }

    write_summary_json(
        path=summary_path,
        summary=summary,
    )

    ranked_results = sorted(
        results,
        key=lambda result: (
            result.f1,
            result.iou,
            result.precision,
        ),
        reverse=True,
    )

    print(
        "GeoWatch validation threshold search completed"
    )
    print(
        "  Checkpoint epoch:",
        resume.next_epoch - 1,
    )
    print(
        "  Validation regions:",
        ", ".join(
            validation_regions
        ),
    )
    print(
        "  Validation pixels:",
        probabilities.size,
    )
    print(
        "  Positive validation pixels:",
        int(
            targets.sum()
        ),
    )
    print(
        "  Default threshold:",
        default_result.threshold,
    )
    print(
        "  Default precision:",
        default_result.precision,
    )
    print(
        "  Default recall:",
        default_result.recall,
    )
    print(
        "  Default F1:",
        default_result.f1,
    )
    print(
        "  Default IoU:",
        default_result.iou,
    )
    print(
        "  Best threshold:",
        best_result.threshold,
    )
    print(
        "  Best precision:",
        best_result.precision,
    )
    print(
        "  Best recall:",
        best_result.recall,
    )
    print(
        "  Best F1:",
        best_result.f1,
    )
    print(
        "  Best IoU:",
        best_result.iou,
    )
    print(
        "  F1 improvement:",
        best_result.f1
        - default_result.f1,
    )
    print(
        "  IoU improvement:",
        best_result.iou
        - default_result.iou,
    )
    print("")
    print(
        "Top five validation thresholds"
    )

    for rank, result in enumerate(
        ranked_results[:5],
        start=1,
    ):
        print(
            f"  {rank}. "
            f"threshold={result.threshold:.2f}, "
            f"precision={result.precision:.6f}, "
            f"recall={result.recall:.6f}, "
            f"f1={result.f1:.6f}, "
            f"iou={result.iou:.6f}"
        )

    print("")
    print(
        "  CSV:",
        csv_path,
    )
    print(
        "  Summary:",
        summary_path,
    )
    print(
        "  Official test regions accessed:",
        False,
    )
    print(
        "  Official test labels accessed:",
        False,
    )


if __name__ == "__main__":
    main()
