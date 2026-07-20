from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import rasterio
from numpy.typing import NDArray

matplotlib.use(
    "Agg"
)

from matplotlib import pyplot as plt

from src.inference.predictor import (
    FROZEN_BANDS,
    FROZEN_CHECKPOINT_EPOCH,
    FROZEN_CHECKPOINT_SHA256,
    FROZEN_PATCH_SIZE,
    FROZEN_STRIDE,
    FROZEN_THRESHOLD,
    FrozenChangePredictor,
    PredictionResult,
    write_mask_geotiff,
    write_probability_geotiff,
)


QUALITATIVE_LABEL = (
    "qualitative_only_no_ground_truth_metrics"
)

RGB_BANDS = (
    "B04",
    "B03",
    "B02",
)

FORBIDDEN_METRIC_KEYS = {
    "f1",
    "iou",
    "precision",
    "recall",
}


class QualitativeInferenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class OutputRecord:
    role: str
    path: str
    sha256: str
    size_bytes: int
    media_type: str
    qualitative: bool


def calculate_sha256(
    path: Path,
) -> str:
    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as source:
        while True:
            block = source.read(
                1024
                * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def stretch_channel(
    values: NDArray[np.float32],
) -> NDArray[np.float32]:
    finite = values[
        np.isfinite(
            values
        )
    ]

    if finite.size == 0:
        raise QualitativeInferenceError(
            "RGB band contains no finite pixels."
        )

    lower = float(
        np.percentile(
            finite,
            2.0,
        )
    )

    upper = float(
        np.percentile(
            finite,
            98.0,
        )
    )

    if upper <= lower:
        return np.zeros_like(
            values,
            dtype=np.float32,
        )

    stretched = (
        values
        - lower
    ) / (
        upper
        - lower
    )

    return np.clip(
        stretched,
        0.0,
        1.0,
    ).astype(
        np.float32,
        copy=False,
    )


def read_rgb(
    directory: Path,
) -> NDArray[np.float32]:
    channels: list[
        NDArray[np.float32]
    ] = []

    reference: tuple[
        int,
        int,
        str,
        str,
    ] | None = None

    for band in RGB_BANDS:
        path = (
            directory
            / f"{band}.tif"
        )

        if not path.is_file():
            raise FileNotFoundError(
                path
            )

        with rasterio.open(
            path
        ) as dataset:
            values = dataset.read(
                1,
                out_dtype="float32",
            )

            current = (
                int(
                    dataset.height
                ),
                int(
                    dataset.width
                ),
                str(
                    dataset.crs
                ),
                str(
                    dataset.transform
                ),
            )

        if reference is None:
            reference = current
        elif current != reference:
            raise QualitativeInferenceError(
                "RGB source bands are not aligned."
            )

        channels.append(
            stretch_channel(
                values
            )
        )

    rgb = np.stack(
        channels,
        axis=-1,
    )

    if not np.isfinite(
        rgb
    ).all():
        raise QualitativeInferenceError(
            "RGB visualization contains non-finite pixels."
        )

    return rgb.astype(
        np.float32,
        copy=False,
    )


def validate_prediction(
    result: PredictionResult,
) -> None:
    if result.qualitative is not True:
        raise QualitativeInferenceError(
            "Hyderabad prediction must be labelled qualitative."
        )

    if result.threshold != FROZEN_THRESHOLD:
        raise QualitativeInferenceError(
            "Frozen threshold was modified."
        )

    if result.checkpoint_epoch != FROZEN_CHECKPOINT_EPOCH:
        raise QualitativeInferenceError(
            "Frozen checkpoint epoch was modified."
        )

    if result.checkpoint_sha256 != FROZEN_CHECKPOINT_SHA256:
        raise QualitativeInferenceError(
            "Frozen checkpoint hash was modified."
        )

    if result.bands != FROZEN_BANDS:
        raise QualitativeInferenceError(
            "Frozen input-band order was modified."
        )

    if result.patch_size != FROZEN_PATCH_SIZE:
        raise QualitativeInferenceError(
            "Frozen patch size was modified."
        )

    if result.stride != FROZEN_STRIDE:
        raise QualitativeInferenceError(
            "Frozen stride was modified."
        )

    expected_shape = (
        result.metadata.height,
        result.metadata.width,
    )

    if result.probability.shape != expected_shape:
        raise QualitativeInferenceError(
            "Probability raster shape is invalid."
        )

    if result.mask.shape != expected_shape:
        raise QualitativeInferenceError(
            "Mask raster shape is invalid."
        )

    if not np.isfinite(
        result.probability
    ).all():
        raise QualitativeInferenceError(
            "Probability raster contains non-finite values."
        )

    if float(
        result.probability.min()
    ) < 0.0:
        raise QualitativeInferenceError(
            "Probability raster contains values below zero."
        )

    if float(
        result.probability.max()
    ) > 1.0:
        raise QualitativeInferenceError(
            "Probability raster contains values above one."
        )

    unique_values = set(
        int(
            value
        )
        for value in np.unique(
            result.mask
        )
    )

    if not unique_values.issubset(
        {
            0,
            1,
        }
    ):
        raise QualitativeInferenceError(
            "Prediction mask must contain only zero and one."
        )


def create_visualization(
    before_rgb: NDArray[np.float32],
    after_rgb: NDArray[np.float32],
    result: PredictionResult,
    path: Path,
) -> None:
    if before_rgb.shape != after_rgb.shape:
        raise QualitativeInferenceError(
            "Before and after RGB arrays do not match."
        )

    if before_rgb.shape[
        :2
    ] != result.mask.shape:
        raise QualitativeInferenceError(
            "RGB and prediction dimensions do not match."
        )

    overlay = np.zeros(
        (
            result.metadata.height,
            result.metadata.width,
            4,
        ),
        dtype=np.float32,
    )

    overlay[
        :,
        :,
        0,
    ] = 1.0

    overlay[
        :,
        :,
        3,
    ] = (
        result.mask.astype(
            np.float32
        )
        * 0.55
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(
            14,
            12,
        ),
        constrained_layout=True,
    )

    axes[
        0,
        0,
    ].imshow(
        before_rgb
    )

    axes[
        0,
        0,
    ].set_title(
        "Before Sentinel-2 RGB\nQUALITATIVE ONLY"
    )

    axes[
        0,
        1,
    ].imshow(
        after_rgb
    )

    axes[
        0,
        1,
    ].set_title(
        "After Sentinel-2 RGB\nQUALITATIVE ONLY"
    )

    axes[
        1,
        0,
    ].imshow(
        result.probability,
        vmin=0.0,
        vmax=1.0,
    )

    axes[
        1,
        0,
    ].set_title(
        "Frozen-model prediction intensity\nQUALITATIVE ONLY"
    )

    axes[
        1,
        1,
    ].imshow(
        after_rgb
    )

    axes[
        1,
        1,
    ].imshow(
        overlay
    )

    axes[
        1,
        1,
    ].set_title(
        "Predicted-change overlay\nQUALITATIVE ONLY"
    )

    for axis in axes.flat:
        axis.set_axis_off()

    figure.suptitle(
        "GeoWatch Hyderabad qualitative demonstration\n"
        "No ground truth available — no performance metrics reported",
        fontsize=16,
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    if not path.is_file():
        raise QualitativeInferenceError(
            "Visualization was not created."
        )

    if path.stat().st_size == 0:
        raise QualitativeInferenceError(
            "Visualization is empty."
        )


def validate_manifest_keys(
    value: Any,
) -> None:
    if isinstance(
        value,
        dict,
    ):
        for key, child in value.items():
            normalized = str(
                key
            ).strip().lower()

            if normalized in FORBIDDEN_METRIC_KEYS:
                raise QualitativeInferenceError(
                    f"Forbidden performance-metric field: {key}"
                )

            validate_manifest_keys(
                child
            )

    elif isinstance(
        value,
        list,
    ):
        for child in value:
            validate_manifest_keys(
                child
            )


def atomic_write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():
        raise FileExistsError(
            path
        )

    temporary_path = path.with_suffix(
        path.suffix
        + ".tmp"
    )

    if temporary_path.exists():
        raise FileExistsError(
            temporary_path
        )

    validate_manifest_keys(
        payload
    )

    temporary_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(
        path
    )


def build_argument_parser(
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(
            "data/qualitative/hyderabad"
        ),
    )

    parser.add_argument(
        "--aligned-manifest",
        type=Path,
        default=Path(
            "reports/week7/hyderabad_qualitative/"
            "aligned_pair_manifest.json"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "reports/week7/hyderabad_qualitative"
        ),
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    return parser


def main(
) -> int:
    arguments = build_argument_parser().parse_args()

    if not arguments.aligned_manifest.is_file():
        raise FileNotFoundError(
            arguments.aligned_manifest
        )

    aligned_manifest = json.loads(
        arguments.aligned_manifest.read_text(
            encoding="utf-8"
        )
    )

    if aligned_manifest.get(
        "label"
    ) != QUALITATIVE_LABEL:
        raise QualitativeInferenceError(
            "Aligned input is not labelled qualitative."
        )

    if aligned_manifest.get(
        "ground_truth_available"
    ) is not False:
        raise QualitativeInferenceError(
            "Ground-truth status must remain false."
        )

    if aligned_manifest.get(
        "metrics_allowed"
    ) is not False:
        raise QualitativeInferenceError(
            "Performance-metric reporting must remain disabled."
        )

    targets = {
        "probability": (
            arguments.output_root
            / "hyderabad_qualitative_probability.tif"
        ),
        "mask": (
            arguments.output_root
            / "hyderabad_qualitative_mask.tif"
        ),
        "visualization": (
            arguments.output_root
            / "hyderabad_qualitative_visualization.png"
        ),
        "manifest": (
            arguments.output_root
            / "qualitative_inference_manifest.json"
        ),
    }

    existing = tuple(
        path
        for path in targets.values()
        if path.exists()
    )

    if existing:
        raise FileExistsError(
            "Qualitative outputs already exist: "
            + ", ".join(
                str(path)
                for path in existing
            )
        )

    predictor = FrozenChangePredictor(
        device=arguments.device,
        batch_size=arguments.batch_size,
    )

    result = predictor.predict_pair(
        before_directory=(
            arguments.input_root
            / "before"
        ),
        after_directory=(
            arguments.input_root
            / "after"
        ),
        qualitative=True,
    )

    validate_prediction(
        result
    )

    before_rgb = read_rgb(
        arguments.input_root
        / "before"
    )

    after_rgb = read_rgb(
        arguments.input_root
        / "after"
    )

    arguments.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory(
        prefix="geowatch_hyderabad_inference_",
        dir=arguments.output_root,
    ) as temporary_directory:
        staging_root = Path(
            temporary_directory
        )

        staged_probability = (
            staging_root
            / targets[
                "probability"
            ].name
        )

        staged_mask = (
            staging_root
            / targets[
                "mask"
            ].name
        )

        staged_visualization = (
            staging_root
            / targets[
                "visualization"
            ].name
        )

        write_probability_geotiff(
            result,
            staged_probability,
        )

        write_mask_geotiff(
            result,
            staged_mask,
        )

        create_visualization(
            before_rgb=before_rgb,
            after_rgb=after_rgb,
            result=result,
            path=staged_visualization,
        )

        output_records = [
            OutputRecord(
                role="qualitative_probability_raster",
                path=str(
                    targets[
                        "probability"
                    ]
                ),
                sha256=calculate_sha256(
                    staged_probability
                ),
                size_bytes=staged_probability.stat().st_size,
                media_type="image/tiff; application=geotiff",
                qualitative=True,
            ),
            OutputRecord(
                role="qualitative_binary_mask",
                path=str(
                    targets[
                        "mask"
                    ]
                ),
                sha256=calculate_sha256(
                    staged_mask
                ),
                size_bytes=staged_mask.stat().st_size,
                media_type="image/tiff; application=geotiff",
                qualitative=True,
            ),
            OutputRecord(
                role="qualitative_visualization",
                path=str(
                    targets[
                        "visualization"
                    ]
                ),
                sha256=calculate_sha256(
                    staged_visualization
                ),
                size_bytes=staged_visualization.stat().st_size,
                media_type="image/png",
                qualitative=True,
            ),
        ]

        staged_probability.replace(
            targets[
                "probability"
            ]
        )

        staged_mask.replace(
            targets[
                "mask"
            ]
        )

        staged_visualization.replace(
            targets[
                "visualization"
            ]
        )

    manifest: dict[str, Any] = {
        "label": QUALITATIVE_LABEL,
        "evaluation_role": "unlabelled_qualitative_demonstration",
        "ground_truth_available": False,
        "performance_metrics_allowed": False,
        "performance_metrics_reported": False,
        "source": {
            "input_root": str(
                arguments.input_root
            ),
            "aligned_pair_manifest": str(
                arguments.aligned_manifest
            ),
            "aligned_pair_manifest_sha256": calculate_sha256(
                arguments.aligned_manifest
            ),
        },
        "protocol": {
            "checkpoint_epoch": result.checkpoint_epoch,
            "checkpoint_sha256": result.checkpoint_sha256,
            "threshold": result.threshold,
            "bands": list(
                result.bands
            ),
            "patch_size": result.patch_size,
            "stride": result.stride,
            "patch_count": result.patch_count,
            "device": str(
                predictor.device
            ),
        },
        "raster": {
            "height": result.metadata.height,
            "width": result.metadata.width,
            "crs": str(
                result.metadata.crs
            ),
            "transform": [
                float(
                    value
                )
                for value in result.metadata.transform
            ],
        },
        "outputs": [
            asdict(
                record
            )
            for record in output_records
        ],
        "access": {
            "image_pixels_accessed": True,
            "model_inference_executed": True,
            "model_retrained": False,
            "threshold_retuned": False,
            "official_test_artifacts_modified": False,
        },
    }

    atomic_write_json(
        targets[
            "manifest"
        ],
        manifest,
    )

    print("Hyderabad qualitative inference completed")
    print("  Label:", QUALITATIVE_LABEL)
    print("  Ground truth available:", False)
    print("  Performance metrics reported:", False)
    print("  Device:", predictor.device)
    print("  Checkpoint epoch:", result.checkpoint_epoch)
    print("  Checkpoint SHA-256:", result.checkpoint_sha256)
    print("  Frozen threshold:", result.threshold)
    print("  Patch count:", result.patch_count)
    print("  Probability raster:", targets["probability"])
    print("  Binary mask:", targets["mask"])
    print("  Visualization:", targets["visualization"])
    print("  Manifest:", targets["manifest"])

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
