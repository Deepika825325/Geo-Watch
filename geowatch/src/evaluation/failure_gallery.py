from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.evaluation.evaluate_test import (
    FROZEN_CHECKPOINT_EPOCH,
    FROZEN_THRESHOLD,
    OSCDTestDataset,
    calculate_patch_counts,
    counts_to_metrics,
    load_frozen_protocol,
)
from src.training.train import (
    build_model,
    load_training_config,
)


matplotlib.use(
    "Agg"
)

import matplotlib.pyplot as plt


def calculate_sha256(
    path: Path,
) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


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


def read_expected_sha256(
    path: Path,
) -> str:
    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    fields = path.read_text(
        encoding="utf-8"
    ).strip().split()

    if not fields:
        raise ValueError(
            f"SHA-256 file is empty: {path}"
        )

    expected = fields[
        0
    ].strip().lower()

    if len(
        expected
    ) != 64:
        raise ValueError(
            "Expected SHA-256 must contain 64 hexadecimal characters."
        )

    return expected


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


def build_error_masks(
    probabilities: Tensor,
    targets: Tensor,
    valid_mask: Tensor,
    threshold: float,
) -> dict[str, np.ndarray]:
    if probabilities.shape != targets.shape:
        raise ValueError(
            "Probability and target shapes do not match."
        )

    if probabilities.shape != valid_mask.shape:
        raise ValueError(
            "Probability and valid-mask shapes do not match."
        )

    predictions = (
        probabilities
        >= threshold
    )

    truth = (
        targets
        >= 0.5
    )

    valid = (
        valid_mask
        >= 0.5
    )

    true_positive = (
        valid
        & predictions
        & truth
    )

    false_positive = (
        valid
        & predictions
        & ~truth
    )

    false_negative = (
        valid
        & ~predictions
        & truth
    )

    def convert(
        value: Tensor,
    ) -> np.ndarray:
        return value.squeeze(
            0
        ).detach().cpu().numpy().astype(
            bool,
            copy=False,
        )

    return {
        "ground_truth": convert(
            truth
            & valid
        ),
        "prediction": convert(
            predictions
            & valid
        ),
        "true_positive": convert(
            true_positive
        ),
        "false_positive": convert(
            false_positive
        ),
        "false_negative": convert(
            false_negative
        ),
        "valid": convert(
            valid
        ),
    }


def normalize_rgb_pair(
    before: Tensor,
    after: Tensor,
    valid_mask: Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    if before.ndim != 3:
        raise ValueError(
            "Before tensor must have three dimensions."
        )

    if after.shape != before.shape:
        raise ValueError(
            "Before and after image shapes do not match."
        )

    if before.shape[
        0
    ] < 3:
        raise ValueError(
            "At least three image bands are required."
        )

    valid = valid_mask.squeeze(
        0
    ).detach().cpu().numpy().astype(
        bool,
        copy=False,
    )

    before_array = before.detach().cpu().numpy()
    after_array = after.detach().cpu().numpy()

    rgb_indices = (
        2,
        1,
        0,
    )

    before_rgb = np.transpose(
        before_array[
            list(
                rgb_indices
            )
        ],
        (
            1,
            2,
            0,
        ),
    ).astype(
        np.float32,
        copy=True,
    )

    after_rgb = np.transpose(
        after_array[
            list(
                rgb_indices
            )
        ],
        (
            1,
            2,
            0,
        ),
    ).astype(
        np.float32,
        copy=True,
    )

    if not np.any(
        valid
    ):
        raise ValueError(
            "RGB normalization requires valid pixels."
        )

    for channel in range(
        3
    ):
        values = np.concatenate(
            (
                before_rgb[
                    ...,
                    channel,
                ][
                    valid
                ],
                after_rgb[
                    ...,
                    channel,
                ][
                    valid
                ],
            )
        )

        lower, upper = np.percentile(
            values,
            (
                2.0,
                98.0,
            ),
        )

        if upper <= lower:
            upper = lower + 1.0e-6

        before_rgb[
            ...,
            channel,
        ] = np.clip(
            (
                before_rgb[
                    ...,
                    channel,
                ]
                - lower
            )
            / (
                upper
                - lower
            ),
            0.0,
            1.0,
        )

        after_rgb[
            ...,
            channel,
        ] = np.clip(
            (
                after_rgb[
                    ...,
                    channel,
                ]
                - lower
            )
            / (
                upper
                - lower
            ),
            0.0,
            1.0,
        )

    before_rgb[
        ~valid
    ] = 0.0

    after_rgb[
        ~valid
    ] = 0.0

    return (
        before_rgb,
        after_rgb,
    )


def select_top_records(
    records: Sequence[Mapping[str, Any]],
    regions: Sequence[str],
    error_type: str,
    count_per_region: int,
) -> list[dict[str, Any]]:
    if count_per_region <= 0:
        raise ValueError(
            "count_per_region must be positive."
        )

    metric_name = {
        "false_positive": "false_positive_rate_all_pixels",
        "false_negative": "false_negative_rate_all_pixels",
    }.get(
        error_type
    )

    if metric_name is None:
        raise ValueError(
            f"Unsupported error type: {error_type}"
        )

    selected: list[
        dict[str, Any]
    ] = []

    for region in regions:
        region_records = [
            record
            for record in records
            if str(
                record[
                    "region"
                ]
            )
            == region
        ]

        if len(
            region_records
        ) < count_per_region:
            raise RuntimeError(
                f"Not enough candidate patches for {region}."
            )

        ranked = sorted(
            region_records,
            key=lambda record: (
                -float(
                    record[
                        metric_name
                    ]
                ),
                -float(
                    record[
                        "total_error_rate"
                    ]
                ),
                str(
                    record[
                        "patch_id"
                    ]
                ),
            ),
        )

        for rank, record in enumerate(
            ranked[
                :count_per_region
            ],
            start=1,
        ):
            selected_record = dict(
                record
            )

            selected_record[
                "error_type"
            ] = error_type

            selected_record[
                "selection_rank"
            ] = rank

            selected.append(
                selected_record
            )

    return selected


def collect_candidates(
    dataset: OSCDTestDataset,
    model: torch.nn.Module,
    threshold: float,
    device: torch.device,
    focus_regions: Sequence[str],
    batch_size: int,
    num_workers: int,
) -> tuple[list[dict[str, Any]], int]:
    focus = set(
        focus_regions
    )

    loader_arguments: dict[
        str,
        Any,
    ] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": (
            device.type
            == "cuda"
        ),
        "drop_last": False,
    }

    if num_workers > 0:
        loader_arguments[
            "persistent_workers"
        ] = True

        loader_arguments[
            "prefetch_factor"
        ] = 2

    loader = DataLoader(
        **loader_arguments
    )

    candidates: list[
        dict[str, Any]
    ] = []

    inferred_patches = 0

    model.eval()

    with torch.inference_mode():
        for batch in loader:
            before_device = batch[
                "before"
            ].to(
                device,
                non_blocking=(
                    device.type
                    == "cuda"
                ),
            )

            after_device = batch[
                "after"
            ].to(
                device,
                non_blocking=(
                    device.type
                    == "cuda"
                ),
            )

            with torch.amp.autocast(
                device_type=device.type,
                enabled=(
                    device.type
                    == "cuda"
                ),
            ):
                logits = model(
                    before_device,
                    after_device,
                )

            probabilities = torch.sigmoid(
                logits.float()
            ).cpu()

            batch_size_actual = int(
                probabilities.shape[
                    0
                ]
            )

            inferred_patches += batch_size_actual

            for index in range(
                batch_size_actual
            ):
                region = str(
                    batch[
                        "region"
                    ][
                        index
                    ]
                )

                if region not in focus:
                    continue

                target = batch[
                    "mask"
                ][
                    index
                ].cpu()

                valid_mask = batch[
                    "valid_mask"
                ][
                    index
                ].cpu()

                probability = probabilities[
                    index
                ]

                counts = calculate_patch_counts(
                    probabilities=probability,
                    targets=target,
                    valid_mask=valid_mask,
                    threshold=threshold,
                )

                metrics = counts_to_metrics(
                    counts
                )

                evaluated_pixels = int(
                    metrics[
                        "evaluated_pixels"
                    ]
                )

                candidates.append(
                    {
                        "region": region,
                        "patch_id": str(
                            batch[
                                "patch_id"
                            ][
                                index
                            ]
                        ),
                        "row": int(
                            batch[
                                "row"
                            ][
                                index
                            ].item()
                        ),
                        "column": int(
                            batch[
                                "column"
                            ][
                                index
                            ].item()
                        ),
                        "valid_height": int(
                            batch[
                                "valid_height"
                            ][
                                index
                            ].item()
                        ),
                        "valid_width": int(
                            batch[
                                "valid_width"
                            ][
                                index
                            ].item()
                        ),
                        **metrics,
                        "false_positive_rate_all_pixels": safe_divide(
                            int(
                                metrics[
                                    "false_positive"
                                ]
                            ),
                            evaluated_pixels,
                        ),
                        "false_negative_rate_all_pixels": safe_divide(
                            int(
                                metrics[
                                    "false_negative"
                                ]
                            ),
                            evaluated_pixels,
                        ),
                        "total_error_rate": safe_divide(
                            int(
                                metrics[
                                    "false_positive"
                                ]
                            )
                            + int(
                                metrics[
                                    "false_negative"
                                ]
                            ),
                            evaluated_pixels,
                        ),
                        "_before": batch[
                            "before"
                        ][
                            index
                        ].cpu(),
                        "_after": batch[
                            "after"
                        ][
                            index
                        ].cpu(),
                        "_target": target,
                        "_valid_mask": valid_mask,
                        "_probability": probability,
                    }
                )

    if inferred_patches != len(
        dataset
    ):
        raise RuntimeError(
            "Failure-gallery inference did not cover every test patch."
        )

    return (
        candidates,
        inferred_patches,
    )


def render_entry(
    entry: Mapping[str, Any],
    output_path: Path,
    threshold: float,
) -> None:
    before_rgb, after_rgb = normalize_rgb_pair(
        before=entry[
            "_before"
        ],
        after=entry[
            "_after"
        ],
        valid_mask=entry[
            "_valid_mask"
        ],
    )

    masks = build_error_masks(
        probabilities=entry[
            "_probability"
        ],
        targets=entry[
            "_target"
        ],
        valid_mask=entry[
            "_valid_mask"
        ],
        threshold=threshold,
    )

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(
            15,
            9,
        ),
    )

    flattened = axes.ravel()

    flattened[
        0
    ].imshow(
        before_rgb
    )

    flattened[
        0
    ].set_title(
        "Before"
    )

    flattened[
        1
    ].imshow(
        after_rgb
    )

    flattened[
        1
    ].set_title(
        "After"
    )

    flattened[
        2
    ].imshow(
        masks[
            "ground_truth"
        ],
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    flattened[
        2
    ].set_title(
        "Ground truth"
    )

    flattened[
        3
    ].imshow(
        masks[
            "prediction"
        ],
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    flattened[
        3
    ].set_title(
        "Prediction"
    )

    flattened[
        4
    ].imshow(
        masks[
            "false_positive"
        ],
        cmap="Reds",
        vmin=0,
        vmax=1,
    )

    flattened[
        4
    ].set_title(
        "False positives"
    )

    flattened[
        5
    ].imshow(
        masks[
            "false_negative"
        ],
        cmap="Blues",
        vmin=0,
        vmax=1,
    )

    flattened[
        5
    ].set_title(
        "False negatives"
    )

    for axis in flattened:
        axis.axis(
            "off"
        )

    figure.suptitle(
        (
            f"{entry['region']} | "
            f"{entry['error_type']} rank "
            f"{entry['selection_rank']} | "
            f"F1={float(entry['f1']):.4f} | "
            f"IoU={float(entry['iou']):.4f}"
        )
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def build_argument_parser(
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "experiments/run_full/train_config.yaml"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "experiments/run_full/checkpoints/"
            "best_model_epoch24.pt"
        ),
    )

    parser.add_argument(
        "--threshold-summary",
        type=Path,
        default=Path(
            "experiments/run_full/threshold_search/"
            "threshold_search_summary.json"
        ),
    )

    parser.add_argument(
        "--official-root",
        type=Path,
        default=Path(
            "data/benchmark/oscd/week6_official"
        ),
    )

    parser.add_argument(
        "--test-results",
        type=Path,
        default=Path(
            "reports/week6/test_results.json"
        ),
    )

    parser.add_argument(
        "--test-results-sha256",
        type=Path,
        default=Path(
            "reports/week6/test_results.sha256"
        ),
    )

    parser.add_argument(
        "--failure-analysis",
        type=Path,
        default=Path(
            "reports/week6/failure_analysis.json"
        ),
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(
            "reports/week6/failure_gallery"
        ),
    )

    parser.add_argument(
        "--count-per-region",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--device",
        choices=(
            "auto",
            "cpu",
            "cuda",
        ),
        default="auto",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    return parser


def main(
) -> int:
    arguments = build_argument_parser().parse_args()

    expected_test_sha256 = read_expected_sha256(
        arguments.test_results_sha256
    )

    original_test_sha256 = calculate_sha256(
        arguments.test_results
    )

    if original_test_sha256 != expected_test_sha256:
        raise RuntimeError(
            "Frozen official test-result SHA-256 does not match."
        )

    analysis = load_json(
        arguments.failure_analysis
    )

    integrity = require_mapping(
        analysis.get(
            "integrity"
        ),
        "integrity",
    )

    if str(
        integrity[
            "test_result_sha256"
        ]
    ) != original_test_sha256:
        raise RuntimeError(
            "Failure analysis references a different test result."
        )

    summary = require_mapping(
        analysis.get(
            "summary"
        ),
        "summary",
    )

    false_positive_regions = tuple(
        str(
            region
        )
        for region in summary[
            "false_positive_focus_regions"
        ]
    )

    false_negative_regions = tuple(
        str(
            region
        )
        for region in summary[
            "false_negative_focus_regions"
        ]
    )

    focus_regions = tuple(
        dict.fromkeys(
            (
                *false_positive_regions,
                *false_negative_regions,
            )
        )
    )

    if arguments.output_directory.exists() and any(
        arguments.output_directory.iterdir()
    ):
        raise FileExistsError(
            "Failure-gallery directory is not empty."
        )

    if arguments.device == "auto":
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = torch.device(
            arguments.device
        )

    if (
        device.type
        == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but is unavailable."
        )

    frozen = load_frozen_protocol(
        config_path=arguments.config,
        threshold_summary_path=arguments.threshold_summary,
        checkpoint_path=arguments.checkpoint,
    )

    config = load_training_config(
        arguments.config
    )

    dataset = OSCDTestDataset(
        official_root=arguments.official_root,
        region_names=frozen.test_regions,
        band_names=frozen.bands,
        patch_size=frozen.patch_size,
        stride=frozen.stride,
        reflectance_scale=frozen.reflectance_scale,
        clip_minimum=frozen.clip_minimum,
        clip_maximum=frozen.clip_maximum,
    )

    model = build_model(
        config=config,
        device=device,
        disable_pretrained=True,
    )

    checkpoint = torch.load(
        arguments.checkpoint,
        map_location=device,
        weights_only=False,
    )

    if int(
        checkpoint[
            "epoch"
        ]
    ) != FROZEN_CHECKPOINT_EPOCH:
        raise RuntimeError(
            "Failure-gallery checkpoint epoch is not frozen."
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    candidates, inferred_patches = collect_candidates(
        dataset=dataset,
        model=model,
        threshold=FROZEN_THRESHOLD,
        device=device,
        focus_regions=focus_regions,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
    )

    selected = [
        *select_top_records(
            records=candidates,
            regions=false_positive_regions,
            error_type="false_positive",
            count_per_region=arguments.count_per_region,
        ),
        *select_top_records(
            records=candidates,
            regions=false_negative_regions,
            error_type="false_negative",
            count_per_region=arguments.count_per_region,
        ),
    ]

    arguments.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_entries: list[
        dict[str, Any]
    ] = []

    for entry in selected:
        filename = (
            f"{entry['error_type']}_"
            f"{entry['region']}_"
            f"{int(entry['selection_rank']):02d}_"
            f"{entry['patch_id']}.png"
        )

        output_path = (
            arguments.output_directory
            / filename
        )

        render_entry(
            entry=entry,
            output_path=output_path,
            threshold=FROZEN_THRESHOLD,
        )

        serializable = {
            key: value
            for key, value in entry.items()
            if not key.startswith(
                "_"
            )
        }

        serializable[
            "file"
        ] = filename

        serializable[
            "file_sha256"
        ] = calculate_sha256(
            output_path
        )

        manifest_entries.append(
            serializable
        )

    final_test_sha256 = calculate_sha256(
        arguments.test_results
    )

    if final_test_sha256 != original_test_sha256:
        raise RuntimeError(
            "Official test results changed during gallery generation."
        )

    manifest = {
        "protocol": {
            "checkpoint_epoch": FROZEN_CHECKPOINT_EPOCH,
            "checkpoint_sha256": frozen.checkpoint_sha256,
            "threshold": FROZEN_THRESHOLD,
            "test_result_sha256": original_test_sha256,
            "test_data_used_for_tuning": False,
            "selection_role": "post_hoc_failure_visualization_only",
        },
        "generation": {
            "device": str(
                device
            ),
            "inferred_patches": inferred_patches,
            "focus_regions": list(
                focus_regions
            ),
            "count_per_region": arguments.count_per_region,
            "gallery_entry_count": len(
                manifest_entries
            ),
        },
        "entries": manifest_entries,
    }

    manifest_path = (
        arguments.output_directory
        / "manifest.json"
    )

    temporary_manifest = manifest_path.with_suffix(
        ".json.tmp"
    )

    temporary_manifest.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary_manifest.replace(
        manifest_path
    )

    print("GeoWatch failure gallery completed")
    print("  Directory:", arguments.output_directory)
    print("  Device:", device)
    print("  Patches inferred:", inferred_patches)
    print("  Focus regions:", ", ".join(focus_regions))
    print("  Gallery images:", len(manifest_entries))
    print("  Test-result SHA-256:", final_test_sha256)
    print("  Test data used for tuning:", False)

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
