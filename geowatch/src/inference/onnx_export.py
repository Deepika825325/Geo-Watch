from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import Tensor

from src.evaluation.evaluate_test import (
    OSCDTestDataset,
    OSCDTestSample,
    load_frozen_protocol,
)
from src.training.train import (
    build_model,
    load_training_config,
)


def load_frozen_model(
    config_path: Path,
    checkpoint_path: Path,
    expected_epoch: int,
    device: torch.device,
) -> torch.nn.Module:
    config = load_training_config(
        config_path
    )

    model = build_model(
        config=config,
        device=device,
        disable_pretrained=True,
    )

    checkpoint: Mapping[
        str,
        Any,
    ] = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    epoch = int(
        checkpoint[
            "epoch"
        ]
    )

    if epoch != expected_epoch:
        raise RuntimeError(
            f"Expected checkpoint epoch {expected_epoch}; received {epoch}."
        )

    state_dict = checkpoint.get(
        "model_state_dict"
    )

    if not isinstance(
        state_dict,
        Mapping,
    ):
        raise RuntimeError(
            "Checkpoint model_state_dict was not found."
        )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    return model


def select_real_test_patch(
    dataset: OSCDTestDataset,
    minimum_changed_pixels: int,
) -> tuple[
    int,
    OSCDTestSample,
    int,
]:
    if minimum_changed_pixels < 0:
        raise ValueError(
            "minimum_changed_pixels cannot be negative."
        )

    best_index = -1
    best_changed_pixels = -1
    best_sample: OSCDTestSample | None = None

    for index in range(
        len(
            dataset
        )
    ):
        sample = dataset[
            index
        ]

        changed_pixels = int(
            torch.count_nonzero(
                (
                    sample[
                        "mask"
                    ]
                    >= 0.5
                )
                & (
                    sample[
                        "valid_mask"
                    ]
                    >= 0.5
                )
            ).item()
        )

        if changed_pixels > best_changed_pixels:
            best_index = index
            best_changed_pixels = changed_pixels
            best_sample = sample

        if changed_pixels >= minimum_changed_pixels:
            return (
                index,
                sample,
                changed_pixels,
            )

    if (
        best_sample is None
        or best_index < 0
    ):
        raise RuntimeError(
            "OSCD test dataset contains no patches."
        )

    return (
        best_index,
        best_sample,
        best_changed_pixels,
    )


def validate_input_pair(
    before: Tensor,
    after: Tensor,
    band_count: int,
    patch_size: int,
) -> None:
    expected_shape = (
        band_count,
        patch_size,
        patch_size,
    )

    if tuple(
        before.shape
    ) != expected_shape:
        raise RuntimeError(
            f"Unexpected before tensor shape: {tuple(before.shape)}."
        )

    if tuple(
        after.shape
    ) != expected_shape:
        raise RuntimeError(
            f"Unexpected after tensor shape: {tuple(after.shape)}."
        )

    if before.dtype != torch.float32:
        raise RuntimeError(
            f"Unexpected before dtype: {before.dtype}."
        )

    if after.dtype != torch.float32:
        raise RuntimeError(
            f"Unexpected after dtype: {after.dtype}."
        )

    if not torch.isfinite(
        before
    ).all():
        raise RuntimeError(
            "Before tensor contains non-finite values."
        )

    if not torch.isfinite(
        after
    ).all():
        raise RuntimeError(
            "After tensor contains non-finite values."
        )


def export_to_onnx(
    model: torch.nn.Module,
    before: Tensor,
    after: Tensor,
    output_path: Path,
    opset_version: int,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    before_batch = (
        before
        .unsqueeze(
            0
        )
        .contiguous()
    )

    after_batch = (
        after
        .unsqueeze(
            0
        )
        .contiguous()
    )

    torch.onnx.export(
        model,
        (
            before_batch,
            after_batch,
        ),
        output_path,
        input_names=[
            "before",
            "after",
        ],
        output_names=[
            "logits",
        ],
        dynamic_axes={
            "before": {
                0: "batch",
            },
            "after": {
                0: "batch",
            },
            "logits": {
                0: "batch",
            },
        },
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,
    )

    if (
        not output_path.is_file()
        or output_path.stat().st_size <= 0
    ):
        raise RuntimeError(
            "ONNX export did not create a valid file."
        )

    exported_model = onnx.load(
        output_path
    )

    onnx.checker.check_model(
        exported_model
    )


def sigmoid_numpy(
    logits: np.ndarray,
) -> np.ndarray:
    clipped = np.clip(
        logits,
        -80.0,
        80.0,
    )

    return (
        np.float32(
            1.0
        )
        / (
            np.float32(
                1.0
            )
            + np.exp(
                -clipped
            )
        )
    ).astype(
        np.float32,
        copy=False,
    )


def compare_cuda_vs_onnx_cpu(
    cuda_model: torch.nn.Module,
    onnx_path: Path,
    before: Tensor,
    after: Tensor,
    threshold: float,
) -> dict[
    str,
    float | int,
]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for the frozen PyTorch reference."
        )

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision(
        "highest"
    )

    before_cpu = (
        before
        .unsqueeze(
            0
        )
        .contiguous()
    )

    after_cpu = (
        after
        .unsqueeze(
            0
        )
        .contiguous()
    )

    before_cuda = before_cpu.to(
        device="cuda",
        non_blocking=False,
    )

    after_cuda = after_cpu.to(
        device="cuda",
        non_blocking=False,
    )

    with torch.inference_mode():
        cuda_logits = (
            cuda_model(
                before_cuda,
                after_cuda,
            )
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

    session_options = (
        ort.SessionOptions()
    )

    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    session = ort.InferenceSession(
        str(
            onnx_path
        ),
        sess_options=session_options,
        providers=[
            "CPUExecutionProvider",
        ],
    )

    onnx_logits = session.run(
        [
            "logits",
        ],
        {
            "before": np.ascontiguousarray(
                before_cpu.numpy()
            ),
            "after": np.ascontiguousarray(
                after_cpu.numpy()
            ),
        },
    )[
        0
    ].astype(
        np.float32,
        copy=False,
    )

    if cuda_logits.shape != onnx_logits.shape:
        raise RuntimeError(
            "CUDA and ONNX output shapes differ."
        )

    absolute_logit_difference = np.abs(
        cuda_logits
        - onnx_logits
    )

    cuda_probabilities = sigmoid_numpy(
        cuda_logits
    )

    onnx_probabilities = sigmoid_numpy(
        onnx_logits
    )

    absolute_probability_difference = np.abs(
        cuda_probabilities
        - onnx_probabilities
    )

    cuda_mask = (
        cuda_probabilities
        >= threshold
    )

    onnx_mask = (
        onnx_probabilities
        >= threshold
    )

    return {
        "max_absolute_logit_difference": float(
            absolute_logit_difference.max()
        ),
        "mean_absolute_logit_difference": float(
            absolute_logit_difference.mean()
        ),
        "max_absolute_probability_difference": float(
            absolute_probability_difference.max()
        ),
        "mean_absolute_probability_difference": float(
            absolute_probability_difference.mean()
        ),
        "binary_mask_pixel_agreement": float(
            np.mean(
                cuda_mask
                == onnx_mask
            )
        ),
        "cuda_changed_pixels": int(
            np.count_nonzero(
                cuda_mask
            )
        ),
        "onnx_changed_pixels": int(
            np.count_nonzero(
                onnx_mask
            )
        ),
        "evaluated_pixels": int(
            cuda_mask.size
        ),
    }


def build_argument_parser(
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export the frozen GeoWatch model to ONNX and compare "
            "CUDA PyTorch output with CPU ONNX Runtime output."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "experiments/run_full/train_config.yaml"
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
        "--checkpoint",
        type=Path,
        default=Path(
            "experiments/run_full/checkpoints/"
            "best_model_epoch24.pt"
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
        "--onnx-output",
        type=Path,
        default=Path(
            "deploy/model.onnx"
        ),
    )

    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path(
            "reports/deployment/onnx_parity.json"
        ),
    )

    parser.add_argument(
        "--minimum-changed-pixels",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--opset-version",
        type=int,
        default=17,
    )

    return parser


def main(
) -> int:
    arguments = (
        build_argument_parser()
        .parse_args()
    )

    frozen = load_frozen_protocol(
        config_path=arguments.config,
        threshold_summary_path=arguments.threshold_summary,
        checkpoint_path=arguments.checkpoint,
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

    (
        sample_index,
        sample,
        ground_truth_changed_pixels,
    ) = select_real_test_patch(
        dataset=dataset,
        minimum_changed_pixels=(
            arguments.minimum_changed_pixels
        ),
    )

    before = sample[
        "before"
    ]

    after = sample[
        "after"
    ]

    validate_input_pair(
        before=before,
        after=after,
        band_count=len(
            frozen.bands
        ),
        patch_size=frozen.patch_size,
    )

    cpu_model = load_frozen_model(
        config_path=arguments.config,
        checkpoint_path=arguments.checkpoint,
        expected_epoch=(
            frozen.checkpoint_epoch
        ),
        device=torch.device(
            "cpu"
        ),
    )

    export_to_onnx(
        model=cpu_model,
        before=before,
        after=after,
        output_path=arguments.onnx_output,
        opset_version=(
            arguments.opset_version
        ),
    )

    cuda_model = load_frozen_model(
        config_path=arguments.config,
        checkpoint_path=arguments.checkpoint,
        expected_epoch=(
            frozen.checkpoint_epoch
        ),
        device=torch.device(
            "cuda"
        ),
    )

    parity = compare_cuda_vs_onnx_cpu(
        cuda_model=cuda_model,
        onnx_path=arguments.onnx_output,
        before=before,
        after=after,
        threshold=frozen.threshold,
    )

    report: dict[
        str,
        Any,
    ] = {
        "checkpoint": {
            "path": str(
                arguments.checkpoint
            ),
            "sha256": (
                frozen.checkpoint_sha256
            ),
            "epoch": (
                frozen.checkpoint_epoch
            ),
        },
        "protocol": {
            "bands": list(
                frozen.bands
            ),
            "patch_size": (
                frozen.patch_size
            ),
            "threshold": (
                frozen.threshold
            ),
        },
        "sample": {
            "dataset_index": (
                sample_index
            ),
            "patch_id": sample[
                "patch_id"
            ],
            "region": sample[
                "region"
            ],
            "row": int(
                sample[
                    "row"
                ]
            ),
            "column": int(
                sample[
                    "column"
                ]
            ),
            "ground_truth_changed_pixels": (
                ground_truth_changed_pixels
            ),
            "valid_pixels": int(
                torch.count_nonzero(
                    sample[
                        "valid_mask"
                    ]
                    >= 0.5
                ).item()
            ),
        },
        "runtime": {
            "pytorch_reference": (
                "CUDA"
            ),
            "onnx_provider": (
                "CPUExecutionProvider"
            ),
            "onnx_opset": (
                arguments.opset_version
            ),
        },
        "parity": parity,
    }

    arguments.report_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    arguments.report_output.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
