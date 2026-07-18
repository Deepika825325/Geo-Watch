from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


FROZEN_THRESHOLD = 0.76
FROZEN_CHECKPOINT_EPOCH = 24
FROZEN_CHECKPOINT_SHA256 = (
    "61e53ba86bc108d6ccbbb636c8da88c967e07da41471504878c74df6a903ea94"
)


def require_mapping(
    value: Any,
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(
        value,
        Mapping,
    ):
        raise TypeError(
            f"{name} must be a mapping."
        )

    return value


def safe_divide(
    numerator: int | float,
    denominator: int | float,
) -> float:
    if denominator == 0:
        return 0.0

    return float(
        numerator
        / denominator
    )


def calculate_sha256(
    path: Path,
) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def load_json(
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(
            f"JSON root must be a mapping: {path}"
        )

    return payload


def determine_error_direction(
    false_positive: int,
    false_negative: int,
) -> str:
    if false_positive > false_negative:
        return "false_positive_dominant"

    if false_negative > false_positive:
        return "false_negative_dominant"

    return "balanced"


def determine_rank_tier(
    rank: int,
    total_regions: int,
) -> str:
    if rank <= 3:
        return "top"

    if rank > total_regions - 3:
        return "bottom"

    return "middle"


def analyze_region(
    region: Mapping[str, Any],
    rank: int,
    total_regions: int,
) -> dict[str, Any]:
    evaluated_pixels = int(
        region[
            "evaluated_pixels"
        ]
    )

    true_positive = int(
        region[
            "true_positive"
        ]
    )

    false_positive = int(
        region[
            "false_positive"
        ]
    )

    false_negative = int(
        region[
            "false_negative"
        ]
    )

    true_negative = int(
        region[
            "true_negative"
        ]
    )

    prevalence = float(
        region[
            "change_prevalence"
        ]
    )

    predicted_fraction = float(
        region[
            "predicted_change_fraction"
        ]
    )

    if (
        true_positive
        + false_positive
        + false_negative
        + true_negative
        != evaluated_pixels
    ):
        raise RuntimeError(
            f"Confusion counts are invalid for {region['region']}."
        )

    prediction_bias = (
        "overprediction"
        if predicted_fraction > prevalence
        else (
            "underprediction"
            if predicted_fraction < prevalence
            else "balanced"
        )
    )

    return {
        "rank": rank,
        "rank_tier": determine_rank_tier(
            rank,
            total_regions,
        ),
        "region": str(
            region[
                "region"
            ]
        ),
        "height": int(
            region[
                "height"
            ]
        ),
        "width": int(
            region[
                "width"
            ]
        ),
        "patch_count": int(
            region[
                "patch_count"
            ]
        ),
        "precision": float(
            region[
                "precision"
            ]
        ),
        "recall": float(
            region[
                "recall"
            ]
        ),
        "f1": float(
            region[
                "f1"
            ]
        ),
        "iou": float(
            region[
                "iou"
            ]
        ),
        "accuracy": float(
            region[
                "accuracy"
            ]
        ),
        "change_prevalence": prevalence,
        "predicted_change_fraction": predicted_fraction,
        "prediction_to_truth_ratio": safe_divide(
            predicted_fraction,
            prevalence,
        ),
        "prediction_bias": prediction_bias,
        "precision_recall_gap": (
            float(
                region[
                    "precision"
                ]
            )
            - float(
                region[
                    "recall"
                ]
            )
        ),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "evaluated_pixels": evaluated_pixels,
        "false_positive_rate_all_pixels": safe_divide(
            false_positive,
            evaluated_pixels,
        ),
        "false_negative_rate_all_pixels": safe_divide(
            false_negative,
            evaluated_pixels,
        ),
        "total_error_rate": safe_divide(
            false_positive
            + false_negative,
            evaluated_pixels,
        ),
        "false_discovery_fraction": safe_divide(
            false_positive,
            true_positive
            + false_positive,
        ),
        "miss_rate": safe_divide(
            false_negative,
            true_positive
            + false_negative,
        ),
        "dominant_error": determine_error_direction(
            false_positive,
            false_negative,
        ),
    }


def build_generalization_comparison(
    test_micro: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    metric_names = (
        "precision",
        "recall",
        "f1",
        "iou",
        "accuracy",
    )

    validation = {
        metric_name: float(
            validation_metrics[
                metric_name
            ]
        )
        for metric_name in metric_names
    }

    test = {
        metric_name: float(
            test_micro[
                metric_name
            ]
        )
        for metric_name in metric_names
    }

    delta = {
        metric_name: (
            test[
                metric_name
            ]
            - validation[
                metric_name
            ]
        )
        for metric_name in metric_names
    }

    return {
        "validation": validation,
        "official_test_micro": test,
        "delta_test_minus_validation": delta,
    }


def select_regions(
    regions: Sequence[Mapping[str, Any]],
    metric: str,
    count: int,
    reverse: bool,
) -> list[str]:
    ranked = sorted(
        regions,
        key=lambda region: float(
            region[
                metric
            ]
        ),
        reverse=reverse,
    )

    return [
        str(
            region[
                "region"
            ]
        )
        for region in ranked[
            :count
        ]
    ]


def build_analysis(
    test_payload: Mapping[str, Any],
    validation_payload: Mapping[str, Any],
    test_result_sha256: str,
    validation_summary_sha256: str,
    reported_invocations: int,
    identical_console_metrics_reported: bool,
) -> dict[str, Any]:
    protocol = require_mapping(
        test_payload.get(
            "protocol"
        ),
        "protocol",
    )

    access = require_mapping(
        test_payload.get(
            "access"
        ),
        "access",
    )

    evaluation = require_mapping(
        test_payload.get(
            "evaluation"
        ),
        "evaluation",
    )

    micro = require_mapping(
        evaluation.get(
            "micro"
        ),
        "evaluation.micro",
    )

    macro = require_mapping(
        evaluation.get(
            "macro"
        ),
        "evaluation.macro",
    )

    validation_metrics = require_mapping(
        validation_payload.get(
            "best_threshold_metrics"
        ),
        "best_threshold_metrics",
    )

    if float(
        protocol[
            "threshold"
        ]
    ) != FROZEN_THRESHOLD:
        raise RuntimeError(
            "Official test threshold is not frozen."
        )

    if int(
        protocol[
            "checkpoint_epoch"
        ]
    ) != FROZEN_CHECKPOINT_EPOCH:
        raise RuntimeError(
            "Official test checkpoint epoch is not frozen."
        )

    if str(
        protocol[
            "checkpoint_sha256"
        ]
    ) != FROZEN_CHECKPOINT_SHA256:
        raise RuntimeError(
            "Official test checkpoint SHA-256 is not frozen."
        )

    if access.get(
        "test_data_used_for_tuning"
    ) is not False:
        raise RuntimeError(
            "Official test data must not be used for tuning."
        )

    raw_regions = evaluation.get(
        "per_region"
    )

    if not isinstance(
        raw_regions,
        list,
    ):
        raise TypeError(
            "evaluation.per_region must be a list."
        )

    if len(
        raw_regions
    ) != 10:
        raise RuntimeError(
            "Expected ten official test regions."
        )

    ranked_raw = sorted(
        (
            require_mapping(
                region,
                "per-region record",
            )
            for region in raw_regions
        ),
        key=lambda region: float(
            region[
                "f1"
            ]
        ),
        reverse=True,
    )

    regions = [
        analyze_region(
            region=region,
            rank=rank,
            total_regions=len(
                ranked_raw
            ),
        )
        for rank, region in enumerate(
            ranked_raw,
            start=1,
        )
    ]

    strongest = regions[
        0
    ]

    weakest = regions[
        -1
    ]

    macro_mean = require_mapping(
        macro.get(
            "mean"
        ),
        "macro.mean",
    )

    macro_median = require_mapping(
        macro.get(
            "median"
        ),
        "macro.median",
    )

    return {
        "integrity": {
            "test_result_sha256": test_result_sha256,
            "validation_summary_sha256": validation_summary_sha256,
            "checkpoint_sha256": FROZEN_CHECKPOINT_SHA256,
            "checkpoint_epoch": FROZEN_CHECKPOINT_EPOCH,
            "threshold": FROZEN_THRESHOLD,
            "reported_full_evaluation_invocations": (
                reported_invocations
            ),
            "identical_console_metrics_reported": (
                identical_console_metrics_reported
            ),
            "first_result_file_was_overwritten": (
                reported_invocations
                > 1
            ),
            "test_data_used_for_tuning": False,
        },
        "generalization": build_generalization_comparison(
            test_micro=micro,
            validation_metrics=validation_metrics,
        ),
        "distribution": {
            "micro_f1": float(
                micro[
                    "f1"
                ]
            ),
            "macro_mean_f1": float(
                macro_mean[
                    "f1"
                ]
            ),
            "macro_median_f1": float(
                macro_median[
                    "f1"
                ]
            ),
            "micro_iou": float(
                micro[
                    "iou"
                ]
            ),
            "macro_mean_iou": float(
                macro_mean[
                    "iou"
                ]
            ),
            "macro_median_iou": float(
                macro_median[
                    "iou"
                ]
            ),
            "micro_minus_macro_mean_f1": (
                float(
                    micro[
                        "f1"
                    ]
                )
                - float(
                    macro_mean[
                        "f1"
                    ]
                )
            ),
        },
        "summary": {
            "strongest_region": strongest[
                "region"
            ],
            "strongest_region_f1": strongest[
                "f1"
            ],
            "weakest_region": weakest[
                "region"
            ],
            "weakest_region_f1": weakest[
                "f1"
            ],
            "false_positive_focus_regions": select_regions(
                regions=regions,
                metric="false_positive_rate_all_pixels",
                count=3,
                reverse=True,
            ),
            "false_negative_focus_regions": select_regions(
                regions=regions,
                metric="false_negative_rate_all_pixels",
                count=3,
                reverse=True,
            ),
            "highest_prevalence_regions": select_regions(
                regions=regions,
                metric="change_prevalence",
                count=3,
                reverse=True,
            ),
            "lowest_prevalence_regions": select_regions(
                regions=regions,
                metric="change_prevalence",
                count=3,
                reverse=False,
            ),
            "qualitative_reference_regions": [
                region[
                    "region"
                ]
                for region in regions[
                    :2
                ]
            ],
            "qualitative_failure_regions": [
                region[
                    "region"
                ]
                for region in regions[
                    -3:
                ]
            ],
        },
        "regions": regions,
    }


def build_argument_parser(
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test-results",
        type=Path,
        default=Path(
            "reports/week6/test_results.json"
        ),
    )

    parser.add_argument(
        "--validation-summary",
        type=Path,
        default=Path(
            "experiments/run_full/threshold_search/"
            "threshold_search_summary.json"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/week6/failure_analysis.json"
        ),
    )

    parser.add_argument(
        "--reported-invocations",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--identical-console-metrics-reported",
        action="store_true",
    )

    return parser


def main(
) -> int:
    arguments = build_argument_parser().parse_args()

    if arguments.reported_invocations <= 0:
        raise ValueError(
            "Reported invocation count must be positive."
        )

    if arguments.output.exists():
        raise FileExistsError(
            f"Failure-analysis output already exists: {arguments.output}"
        )

    test_payload = load_json(
        arguments.test_results
    )

    validation_payload = load_json(
        arguments.validation_summary
    )

    analysis = build_analysis(
        test_payload=test_payload,
        validation_payload=validation_payload,
        test_result_sha256=calculate_sha256(
            arguments.test_results
        ),
        validation_summary_sha256=calculate_sha256(
            arguments.validation_summary
        ),
        reported_invocations=arguments.reported_invocations,
        identical_console_metrics_reported=(
            arguments.identical_console_metrics_reported
        ),
    )

    arguments.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_output = arguments.output.with_suffix(
        arguments.output.suffix
        + ".tmp"
    )

    temporary_output.write_text(
        json.dumps(
            analysis,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary_output.replace(
        arguments.output
    )

    generalization = analysis[
        "generalization"
    ][
        "delta_test_minus_validation"
    ]

    summary = analysis[
        "summary"
    ]

    distribution = analysis[
        "distribution"
    ]

    print("GeoWatch Week 6 failure analysis completed")
    print("  Output:", arguments.output)
    print(
        "  Test-result SHA-256:",
        analysis["integrity"]["test_result_sha256"],
    )
    print(
        "  Test minus validation F1:",
        generalization["f1"],
    )
    print(
        "  Test minus validation IoU:",
        generalization["iou"],
    )
    print(
        "  Test minus validation precision:",
        generalization["precision"],
    )
    print(
        "  Test minus validation recall:",
        generalization["recall"],
    )
    print(
        "  Micro minus macro-mean F1:",
        distribution["micro_minus_macro_mean_f1"],
    )
    print(
        "  Strongest region:",
        summary["strongest_region"],
    )
    print(
        "  Weakest region:",
        summary["weakest_region"],
    )
    print(
        "  False-positive focus:",
        ", ".join(
            summary[
                "false_positive_focus_regions"
            ]
        ),
    )
    print(
        "  False-negative focus:",
        ", ".join(
            summary[
                "false_negative_focus_regions"
            ]
        ),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
