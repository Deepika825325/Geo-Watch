"""Binary change-detection metrics for GeoWatch.

This module evaluates the positive change class only. It supports OSCD-style
ground-truth masks encoded as 0 for unchanged and 255 for changed, while also
supporting predictions encoded as either 0/1 or 0/255.

Primary metrics:

* precision;
* recall;
* F1 score;
* intersection over union.

Overall pixel accuracy is included only as a secondary diagnostic because
change-detection datasets are usually dominated by unchanged pixels.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import rasterio
from PIL import Image
from rasterio.errors import RasterioIOError


LOGGER = logging.getLogger("geowatch.metrics")


class MetricsError(RuntimeError):
    """Raised when change-detection metrics cannot be computed safely."""


@dataclass(frozen=True)
class ConfusionCounts:
    """Binary confusion-matrix counts for the positive change class."""

    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    ignored_pixels: int
    evaluated_pixels: int

    @property
    def total_pixels(self) -> int:
        """Return evaluated plus ignored pixels."""
        return self.evaluated_pixels + self.ignored_pixels


@dataclass(frozen=True)
class ChangeMetrics:
    """Metrics calculated for the positive change class."""

    precision: float
    recall: float
    f1_score: float
    iou: float
    accuracy: float
    change_prevalence: float
    predicted_change_fraction: float
    counts: ConfusionCounts

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable metric dictionary."""
        payload = asdict(self)

        payload["metric_scope"] = "positive_change_class"
        payload["primary_metrics"] = [
            "precision",
            "recall",
            "f1_score",
            "iou",
        ]
        payload["accuracy_role"] = "secondary_diagnostic_only"

        return payload


def safe_divide(
    numerator: int | float,
    denominator: int | float,
    zero_division: float,
) -> float:
    """Divide safely using a configured value when the denominator is zero."""
    if denominator == 0:
        return float(zero_division)

    return float(numerator / denominator)


def normalize_values(
    values: Iterable[int],
    argument_name: str,
) -> tuple[int, ...]:
    """Validate and deduplicate integer pixel values."""
    normalized = tuple(
        sorted(
            {
                int(value)
                for value in values
            }
        )
    )

    if not normalized:
        raise MetricsError(
            f"{argument_name} must contain at least one value."
        )

    return normalized


def binarize_mask(
    mask: np.ndarray,
    change_values: Sequence[int] = (),
    ignore_values: Sequence[int] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a source mask to binary change and valid-pixel arrays.

    When ``change_values`` is empty, every positive pixel value is
    interpreted as change. This matches the official TorchGeo OSCD
    loading behaviour, where raw grayscale masks are clamped to the
    binary range 0/1.

    Args:
        mask: Two-dimensional numeric source mask.
        change_values: Optional explicit values representing change.
            When empty, the rule ``mask > 0`` is used.
        ignore_values: Pixel values excluded from evaluation.

    Returns:
        A tuple containing the binary change mask and valid-pixel mask.

    Raises:
        MetricsError: If the mask or value configuration is invalid.
    """
    if mask.ndim != 2:
        raise MetricsError(
            f"Expected a two-dimensional mask, received shape "
            f"{mask.shape}."
        )

    if not np.issubdtype(
        mask.dtype,
        np.number,
    ):
        raise MetricsError(
            f"Mask must use a numeric data type; received {mask.dtype}."
        )

    normalized_ignore = (
        normalize_values(
            ignore_values,
            "ignore_values",
        )
        if ignore_values
        else ()
    )

    if change_values:
        normalized_change = normalize_values(
            change_values,
            "change_values",
        )

        overlap = set(
            normalized_change
        ).intersection(
            normalized_ignore
        )

        if overlap:
            raise MetricsError(
                "Change and ignore values must not overlap. "
                f"Overlapping values: {sorted(overlap)}"
            )

        change_mask = np.isin(
            mask,
            np.asarray(
                normalized_change,
                dtype=mask.dtype,
            ),
        )
    else:
        change_mask = mask > 0

    if normalized_ignore:
        valid_mask = ~np.isin(
            mask,
            np.asarray(
                normalized_ignore,
                dtype=mask.dtype,
            ),
        )
    else:
        valid_mask = np.ones(
            mask.shape,
            dtype=bool,
        )

    return change_mask, valid_mask


def calculate_confusion_counts(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    ground_truth_change_values: Sequence[int] = (),
    prediction_change_values: Sequence[int] = (),
    ground_truth_ignore_values: Sequence[int] = (),
    prediction_ignore_values: Sequence[int] = (),
) -> ConfusionCounts:
    """Calculate binary confusion counts for the change class.

    Pixels ignored by either the ground truth or prediction are excluded.
    """
    if ground_truth.shape != prediction.shape:
        raise MetricsError(
            "Ground truth and prediction shapes do not match: "
            f"{ground_truth.shape} versus {prediction.shape}."
        )

    ground_truth_change, ground_truth_valid = binarize_mask(
        mask=ground_truth,
        change_values=ground_truth_change_values,
        ignore_values=ground_truth_ignore_values,
    )
    prediction_change, prediction_valid = binarize_mask(
        mask=prediction,
        change_values=prediction_change_values,
        ignore_values=prediction_ignore_values,
    )

    valid_pixels = (
        ground_truth_valid
        & prediction_valid
    )

    ignored_pixels = int(
        np.count_nonzero(~valid_pixels)
    )
    evaluated_pixels = int(
        np.count_nonzero(valid_pixels)
    )

    if evaluated_pixels == 0:
        raise MetricsError(
            "No valid pixels remain after applying ignore-value rules."
        )

    true_positive = int(
        np.count_nonzero(
            valid_pixels
            & ground_truth_change
            & prediction_change
        )
    )
    false_positive = int(
        np.count_nonzero(
            valid_pixels
            & ~ground_truth_change
            & prediction_change
        )
    )
    false_negative = int(
        np.count_nonzero(
            valid_pixels
            & ground_truth_change
            & ~prediction_change
        )
    )
    true_negative = int(
        np.count_nonzero(
            valid_pixels
            & ~ground_truth_change
            & ~prediction_change
        )
    )

    count_total = (
        true_positive
        + false_positive
        + false_negative
        + true_negative
    )

    if count_total != evaluated_pixels:
        raise MetricsError(
            "Confusion counts do not sum to the number of evaluated pixels."
        )

    return ConfusionCounts(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        true_negative=true_negative,
        ignored_pixels=ignored_pixels,
        evaluated_pixels=evaluated_pixels,
    )


def calculate_change_metrics(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    ground_truth_change_values: Sequence[int] = (),
    prediction_change_values: Sequence[int] = (),
    ground_truth_ignore_values: Sequence[int] = (),
    prediction_ignore_values: Sequence[int] = (),
    zero_division: float = 0.0,
) -> ChangeMetrics:
    """Calculate positive-class change-detection metrics."""
    if zero_division not in {0.0, 1.0}:
        raise MetricsError(
            "zero_division must be either 0.0 or 1.0."
        )

    counts = calculate_confusion_counts(
        ground_truth=ground_truth,
        prediction=prediction,
        ground_truth_change_values=ground_truth_change_values,
        prediction_change_values=prediction_change_values,
        ground_truth_ignore_values=ground_truth_ignore_values,
        prediction_ignore_values=prediction_ignore_values,
    )

    tp = counts.true_positive
    fp = counts.false_positive
    fn = counts.false_negative
    tn = counts.true_negative

    precision = safe_divide(
        tp,
        tp + fp,
        zero_division,
    )
    recall = safe_divide(
        tp,
        tp + fn,
        zero_division,
    )
    f1_score = safe_divide(
        2 * tp,
        2 * tp + fp + fn,
        zero_division,
    )
    iou = safe_divide(
        tp,
        tp + fp + fn,
        zero_division,
    )
    accuracy = safe_divide(
        tp + tn,
        counts.evaluated_pixels,
        zero_division,
    )
    change_prevalence = safe_divide(
        tp + fn,
        counts.evaluated_pixels,
        zero_division,
    )
    predicted_change_fraction = safe_divide(
        tp + fp,
        counts.evaluated_pixels,
        zero_division,
    )

    return ChangeMetrics(
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        iou=iou,
        accuracy=accuracy,
        change_prevalence=change_prevalence,
        predicted_change_fraction=predicted_change_fraction,
        counts=counts,
    )


def load_mask(path: Path) -> np.ndarray:
    """Load a two-dimensional mask from PNG, TIFF, or NumPy format."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Mask file does not exist: {path}"
        )

    suffix = path.suffix.lower()

    if suffix in {".tif", ".tiff"}:
        try:
            with rasterio.open(path) as dataset:
                if dataset.count != 1:
                    raise MetricsError(
                        f"Mask must contain exactly one band: {path}"
                    )

                return dataset.read(1)

        except RasterioIOError as error:
            raise MetricsError(
                f"Rasterio could not read mask: {path}"
            ) from error

    if suffix in {".png", ".jpg", ".jpeg", ".bmp"}:
        try:
            with Image.open(path) as image:
                array = np.asarray(
                    image.convert("L")
                )
        except OSError as error:
            raise MetricsError(
                f"Pillow could not read mask: {path}"
            ) from error

        return array

    if suffix == ".npy":
        try:
            array = np.load(
                path,
                allow_pickle=False,
            )
        except (OSError, ValueError) as error:
            raise MetricsError(
                f"NumPy could not read mask: {path}"
            ) from error

        if not isinstance(array, np.ndarray):
            raise MetricsError(
                f"NumPy mask is not an array: {path}"
            )

        return array

    raise MetricsError(
        "Unsupported mask format. Supported extensions: "
        ".png, .jpg, .jpeg, .bmp, .tif, .tiff, .npy"
    )


def write_json_atomic(
    payload: dict[str, Any],
    output_path: Path,
) -> None:
    """Write metric results atomically."""
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

    temporary_path.replace(output_path)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a binary change-detection prediction using "
            "positive-class metrics."
        )
    )

    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="Ground-truth change-mask path.",
    )
    parser.add_argument(
        "--prediction",
        type=Path,
        required=True,
        help="Predicted change-mask path.",
    )
    parser.add_argument(
        "--ground-truth-change-values",
        type=int,
        nargs="+",
        default=[],
        help=(
            "Explicit ground-truth change values. Default: every value greater than 0"
        ),
    )
    parser.add_argument(
        "--prediction-change-values",
        type=int,
        nargs="+",
        default=[],
        help=(
            "Explicit prediction change values. Default: every value greater than 0"
        ),
    )
    parser.add_argument(
        "--ground-truth-ignore-values",
        type=int,
        nargs="*",
        default=[],
        help="Ground-truth values excluded from evaluation.",
    )
    parser.add_argument(
        "--prediction-ignore-values",
        type=int,
        nargs="*",
        default=[],
        help="Prediction values excluded from evaluation.",
    )
    parser.add_argument(
        "--zero-division",
        type=float,
        choices=(0.0, 1.0),
        default=0.0,
        help=(
            "Metric value used when its denominator is zero. Default: 0"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for a JSON metrics report.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )

    return parser


def main() -> int:
    """Run change-detection metric evaluation."""
    parser = build_argument_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    try:
        ground_truth = load_mask(
            args.ground_truth
        )
        prediction = load_mask(
            args.prediction
        )

        metrics = calculate_change_metrics(
            ground_truth=ground_truth,
            prediction=prediction,
            ground_truth_change_values=(
                args.ground_truth_change_values
            ),
            prediction_change_values=(
                args.prediction_change_values
            ),
            ground_truth_ignore_values=(
                args.ground_truth_ignore_values
            ),
            prediction_ignore_values=(
                args.prediction_ignore_values
            ),
            zero_division=args.zero_division,
        )

        payload = {
            "ground_truth": str(args.ground_truth),
            "prediction": str(args.prediction),
            **metrics.to_dict(),
        }

        print("Change-class evaluation completed")
        print(
            f"  Precision: {metrics.precision:.6f}"
        )
        print(
            f"  Recall:    {metrics.recall:.6f}"
        )
        print(
            f"  F1 score: {metrics.f1_score:.6f}"
        )
        print(
            f"  IoU:       {metrics.iou:.6f}"
        )
        print(
            f"  Accuracy:  {metrics.accuracy:.6f} "
            "(secondary diagnostic)"
        )
        print(
            f"  Change prevalence: "
            f"{metrics.change_prevalence:.6f}"
        )
        print(
            f"  Evaluated pixels: "
            f"{metrics.counts.evaluated_pixels}"
        )

        if args.output_json is not None:
            write_json_atomic(
                payload=payload,
                output_path=args.output_json,
            )
            print(
                f"  JSON report: {args.output_json}"
            )

        return 0

    except (
        FileNotFoundError,
        PermissionError,
        MetricsError,
        ValueError,
        TypeError,
    ) as error:
        LOGGER.error("%s", error)
        return 1

    except Exception:
        LOGGER.exception(
            "Unexpected metrics evaluation failure."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
