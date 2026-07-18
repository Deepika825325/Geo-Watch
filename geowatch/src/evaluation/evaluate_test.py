from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import fmean, median
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import rasterio
import torch
from rasterio.errors import NotGeoreferencedWarning
from rasterio.windows import Window
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.data.oscd_dataset import (
    PatchRecord,
    build_region_record,
    normalize_band_names,
    read_mask_window,
)
from src.training.train import (
    build_model,
    load_training_config,
    require_mapping,
)


FROZEN_THRESHOLD = 0.76
FROZEN_CHECKPOINT_EPOCH = 24
FROZEN_CHECKPOINT_SHA256 = (
    "61e53ba86bc108d6ccbbb636c8da88c967e07da41471504878c74df6a903ea94"
)

EXPECTED_TEST_REGIONS = (
    "brasilia",
    "chongqing",
    "dubai",
    "lasvegas",
    "milano",
    "montpellier",
    "norcia",
    "rio",
    "saclay_w",
    "valencia",
)

IMAGES_DIRECTORY_NAME = (
    "Onera Satellite Change Detection dataset - Images"
)

TEST_LABELS_DIRECTORY_NAME = (
    "Onera Satellite Change Detection dataset - Test Labels"
)


class OSCDTestSample(TypedDict):
    before: Tensor
    after: Tensor
    mask: Tensor
    valid_mask: Tensor
    region: str
    patch_id: str
    row: int
    column: int
    valid_height: int
    valid_width: int


@dataclass(frozen=True)
class RegionAudit:
    region: str
    height: int
    width: int
    patch_count: int
    padded_patch_count: int
    bottom_padding: int
    right_padding: int
    requires_padding: bool
    valid_pixels: int
    covered_valid_pixels: int


@dataclass(frozen=True)
class FrozenProtocol:
    threshold: float
    checkpoint_epoch: int
    checkpoint_path: str
    checkpoint_sha256: str
    bands: tuple[str, ...]
    patch_size: int
    stride: int
    reflectance_scale: float
    clip_minimum: float
    clip_maximum: float
    development_regions: tuple[str, ...]
    test_regions: tuple[str, ...]


def calculate_sha256(
    path: Path,
) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def evaluation_starts(
    length: int,
    patch_size: int,
    stride: int,
) -> tuple[int, ...]:
    if length <= 0:
        raise ValueError(
            "Region dimension must be positive."
        )

    if patch_size <= 0:
        raise ValueError(
            "Patch size must be positive."
        )

    if stride <= 0:
        raise ValueError(
            "Stride must be positive."
        )

    if stride != patch_size:
        raise ValueError(
            "Exact-coverage evaluation requires stride "
            "to equal patch size."
        )

    return tuple(
        range(
            0,
            length,
            stride,
        )
    )


def calculate_valid_shape(
    region_height: int,
    region_width: int,
    patch: PatchRecord,
) -> tuple[int, int]:
    valid_height = max(
        0,
        min(
            patch.height,
            region_height - patch.row,
        ),
    )

    valid_width = max(
        0,
        min(
            patch.width,
            region_width - patch.column,
        ),
    )

    if valid_height <= 0 or valid_width <= 0:
        raise RuntimeError(
            "Patch contains no valid original image pixels."
        )

    return (
        valid_height,
        valid_width,
    )

def read_test_band_window(
    path: Path,
    patch: PatchRecord,
    reflectance_scale: float,
    clip_minimum: float | None,
    clip_maximum: float | None,
) -> np.ndarray:
    window = Window(
        col_off=patch.column,
        row_off=patch.row,
        width=patch.width,
        height=patch.height,
    )

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            NotGeoreferencedWarning,
        )

        with rasterio.open(
            path
        ) as dataset:
            array = dataset.read(
                1,
                window=window,
                boundless=True,
                fill_value=0,
                out_dtype="float32",
            )

    expected_shape = (
        patch.height,
        patch.width,
    )

    if array.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected raster shape from {path}: "
            f"{array.shape} versus {expected_shape}"
        )

    if not np.isfinite(
        array
    ).all():
        raise RuntimeError(
            f"Non-finite values found in {path}"
        )

    array /= np.float32(
        reflectance_scale
    )

    if clip_minimum is not None:
        np.maximum(
            array,
            np.float32(
                clip_minimum
            ),
            out=array,
        )

    if clip_maximum is not None:
        np.minimum(
            array,
            np.float32(
                clip_maximum
            ),
            out=array,
        )

    return array


class OSCDTestDataset(
    Dataset[OSCDTestSample]
):
    def __init__(
        self,
        official_root: Path | str,
        region_names: Sequence[str],
        band_names: Sequence[str],
        patch_size: int,
        stride: int,
        reflectance_scale: float,
        clip_minimum: float | None,
        clip_maximum: float | None,
    ) -> None:
        super().__init__()

        self.official_root = Path(
            official_root
        )

        self.band_names = normalize_band_names(
            band_names
        )

        self.patch_size = int(
            patch_size
        )

        self.stride = int(
            stride
        )

        self.reflectance_scale = float(
            reflectance_scale
        )

        self.clip_minimum = clip_minimum
        self.clip_maximum = clip_maximum

        selected_regions = tuple(
            str(region).strip().lower()
            for region in region_names
        )

        if selected_regions != EXPECTED_TEST_REGIONS:
            raise ValueError(
                "Official test regions do not match the frozen split."
            )

        if self.patch_size <= 0:
            raise ValueError(
                "patch_size must be positive."
            )

        if self.stride <= 0:
            raise ValueError(
                "stride must be positive."
            )

        if self.reflectance_scale <= 0.0:
            raise ValueError(
                "reflectance_scale must be positive."
            )

        images_root = (
            self.official_root
            / IMAGES_DIRECTORY_NAME
        )

        labels_root = (
            self.official_root
            / TEST_LABELS_DIRECTORY_NAME
        )

        if not images_root.is_dir():
            raise FileNotFoundError(
                images_root
            )

        if not labels_root.is_dir():
            raise FileNotFoundError(
                labels_root
            )

        available_regions = tuple(
            sorted(
                path.name.strip().lower()
                for path in labels_root.iterdir()
                if path.is_dir()
            )
        )

        if available_regions != EXPECTED_TEST_REGIONS:
            raise ValueError(
                "Extracted test-label regions do not match "
                "the frozen Week 6 split."
            )

        self.region_names = selected_regions

        self.regions = tuple(
            build_region_record(
                images_root=images_root,
                labels_root=labels_root,
                region_name=region,
                band_names=self.band_names,
            )
            for region in self.region_names
        )

        patch_records: list[
            PatchRecord
        ] = []

        for region_index, region in enumerate(
            self.regions
        ):
            row_starts = evaluation_starts(
                length=region.height,
                patch_size=self.patch_size,
                stride=self.stride,
            )

            column_starts = evaluation_starts(
                length=region.width,
                patch_size=self.patch_size,
                stride=self.stride,
            )

            for row in row_starts:
                for column in column_starts:
                    patch_records.append(
                        PatchRecord(
                            region_index=region_index,
                            row=row,
                            column=column,
                            height=self.patch_size,
                            width=self.patch_size,
                        )
                    )

        if not patch_records:
            raise RuntimeError(
                "Official test patch index is empty."
            )

        self.patches = tuple(
            patch_records
        )

    def __len__(
        self,
    ) -> int:
        return len(
            self.patches
        )

    def __getitem__(
        self,
        index: int,
    ) -> OSCDTestSample:
        if not isinstance(
            index,
            int,
        ):
            raise TypeError(
                f"Dataset index must be int; received {type(index)}."
            )

        if index < 0:
            index += len(
                self
            )

        if index < 0 or index >= len(
            self
        ):
            raise IndexError(
                index
            )

        patch = self.patches[
            index
        ]

        region = self.regions[
            patch.region_index
        ]

        before_arrays = [
            read_test_band_window(
                path=path,
                patch=patch,
                reflectance_scale=self.reflectance_scale,
                clip_minimum=self.clip_minimum,
                clip_maximum=self.clip_maximum,
            )
            for path in region.before_band_paths
        ]

        after_arrays = [
            read_test_band_window(
                path=path,
                patch=patch,
                reflectance_scale=self.reflectance_scale,
                clip_minimum=self.clip_minimum,
                clip_maximum=self.clip_maximum,
            )
            for path in region.after_band_paths
        ]

        mask_array = read_mask_window(
            path=region.label_path,
            patch=patch,
        )

        valid_height, valid_width = calculate_valid_shape(
            region_height=region.height,
            region_width=region.width,
            patch=patch,
        )

        valid_array = np.zeros(
            (
                patch.height,
                patch.width,
            ),
            dtype=np.float32,
        )

        valid_array[
            :valid_height,
            :valid_width,
        ] = 1.0

        before = torch.from_numpy(
            np.stack(
                before_arrays,
                axis=0,
            )
        ).to(
            dtype=torch.float32
        )

        after = torch.from_numpy(
            np.stack(
                after_arrays,
                axis=0,
            )
        ).to(
            dtype=torch.float32
        )

        mask = torch.from_numpy(
            mask_array[
                None,
                ...,
            ]
        ).to(
            dtype=torch.float32
        )

        valid_mask = torch.from_numpy(
            valid_array[
                None,
                ...,
            ]
        ).to(
            dtype=torch.float32
        )

        expected_image_shape = (
            len(
                self.band_names
            ),
            self.patch_size,
            self.patch_size,
        )

        expected_mask_shape = (
            1,
            self.patch_size,
            self.patch_size,
        )

        if tuple(
            before.shape
        ) != expected_image_shape:
            raise RuntimeError(
                f"Unexpected before shape: {tuple(before.shape)}"
            )

        if tuple(
            after.shape
        ) != expected_image_shape:
            raise RuntimeError(
                f"Unexpected after shape: {tuple(after.shape)}"
            )

        if tuple(
            mask.shape
        ) != expected_mask_shape:
            raise RuntimeError(
                f"Unexpected mask shape: {tuple(mask.shape)}"
            )

        if tuple(
            valid_mask.shape
        ) != expected_mask_shape:
            raise RuntimeError(
                f"Unexpected valid-mask shape: {tuple(valid_mask.shape)}"
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

        return OSCDTestSample(
            before=before,
            after=after,
            mask=mask,
            valid_mask=valid_mask,
            region=region.name,
            patch_id=(
                f"{region.name}_{patch.patch_id}"
            ),
            row=patch.row,
            column=patch.column,
            valid_height=valid_height,
            valid_width=valid_width,
        )

    def region_audits(
        self,
    ) -> tuple[RegionAudit, ...]:
        patch_counts = {
            region.name: 0
            for region in self.regions
        }

        padded_patch_counts = {
            region.name: 0
            for region in self.regions
        }

        covered_valid_pixels = {
            region.name: 0
            for region in self.regions
        }

        for patch in self.patches:
            region = self.regions[
                patch.region_index
            ]

            valid_height, valid_width = calculate_valid_shape(
                region_height=region.height,
                region_width=region.width,
                patch=patch,
            )

            patch_counts[
                region.name
            ] += 1

            covered_valid_pixels[
                region.name
            ] += (
                valid_height
                * valid_width
            )

            if (
                valid_height < patch.height
                or valid_width < patch.width
            ):
                padded_patch_counts[
                    region.name
                ] += 1

        audits: list[
            RegionAudit
        ] = []

        for region in self.regions:
            row_starts = evaluation_starts(
                length=region.height,
                patch_size=self.patch_size,
                stride=self.stride,
            )

            column_starts = evaluation_starts(
                length=region.width,
                patch_size=self.patch_size,
                stride=self.stride,
            )

            valid_pixels = (
                region.height
                * region.width
            )

            covered_pixels = covered_valid_pixels[
                region.name
            ]

            if covered_pixels != valid_pixels:
                raise RuntimeError(
                    f"Valid-pixel coverage mismatch for {region.name}: "
                    f"{covered_pixels} versus {valid_pixels}"
                )

            bottom_padding = (
                len(
                    row_starts
                )
                * self.patch_size
                - region.height
            )

            right_padding = (
                len(
                    column_starts
                )
                * self.patch_size
                - region.width
            )

            audits.append(
                RegionAudit(
                    region=region.name,
                    height=region.height,
                    width=region.width,
                    patch_count=patch_counts[
                        region.name
                    ],
                    padded_patch_count=padded_patch_counts[
                        region.name
                    ],
                    bottom_padding=bottom_padding,
                    right_padding=right_padding,
                    requires_padding=(
                        bottom_padding > 0
                        or right_padding > 0
                    ),
                    valid_pixels=valid_pixels,
                    covered_valid_pixels=covered_pixels,
                )
            )

        return tuple(
            audits
        )


def load_frozen_protocol(
    config_path: Path,
    threshold_summary_path: Path,
    checkpoint_path: Path,
) -> FrozenProtocol:
    config = load_training_config(
        config_path
    )

    dataset_config = require_mapping(
        config.get("dataset"),
        "dataset",
    )

    protocol_config = require_mapping(
        config.get("protocol"),
        "protocol",
    )

    threshold_summary: Mapping[str, Any] = json.loads(
        threshold_summary_path.read_text(
            encoding="utf-8"
        )
    )

    threshold_metrics = require_mapping(
        threshold_summary.get(
            "best_threshold_metrics"
        ),
        "best_threshold_metrics",
    )

    checkpoint_summary = require_mapping(
        threshold_summary.get(
            "checkpoint"
        ),
        "checkpoint",
    )

    checkpoint_sha256 = calculate_sha256(
        checkpoint_path
    )

    checkpoint: Mapping[str, Any] = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    threshold = float(
        threshold_metrics["threshold"]
    )

    checkpoint_epoch = int(
        checkpoint["epoch"]
    )

    if threshold != FROZEN_THRESHOLD:
        raise RuntimeError(
            "The selected threshold has changed."
        )

    if int(
        checkpoint_summary["epoch"]
    ) != FROZEN_CHECKPOINT_EPOCH:
        raise RuntimeError(
            "The summary checkpoint epoch has changed."
        )

    if checkpoint_epoch != FROZEN_CHECKPOINT_EPOCH:
        raise RuntimeError(
            "The loaded checkpoint epoch has changed."
        )

    if checkpoint_sha256 != FROZEN_CHECKPOINT_SHA256:
        raise RuntimeError(
            "The checkpoint SHA-256 does not match."
        )

    if str(
        checkpoint_summary["sha256"]
    ) != FROZEN_CHECKPOINT_SHA256:
        raise RuntimeError(
            "The summary checkpoint SHA-256 does not match."
        )

    if protocol_config.get(
        "official_test_regions_sealed"
    ) is not True:
        raise RuntimeError(
            "The sealed-test protocol is not recorded."
        )

    development_regions = tuple(
        str(region).strip().lower()
        for region in (
            tuple(
                dataset_config["train_regions"]
            )
            + tuple(
                dataset_config["validation_regions"]
            )
        )
    )

    if set(
        development_regions
    ).intersection(
        EXPECTED_TEST_REGIONS
    ):
        raise RuntimeError(
            "Development and official test regions overlap."
        )

    return FrozenProtocol(
        threshold=threshold,
        checkpoint_epoch=checkpoint_epoch,
        checkpoint_path=str(
            checkpoint_path
        ),
        checkpoint_sha256=checkpoint_sha256,
        bands=tuple(
            str(band)
            for band in dataset_config["bands"]
        ),
        patch_size=int(
            dataset_config["patch_size"]
        ),
        stride=int(
            dataset_config["stride"]
        ),
        reflectance_scale=float(
            dataset_config["reflectance_scale"]
        ),
        clip_minimum=float(
            dataset_config["clip_minimum"]
        ),
        clip_maximum=float(
            dataset_config["clip_maximum"]
        ),
        development_regions=development_regions,
        test_regions=EXPECTED_TEST_REGIONS,
    )


@dataclass(frozen=True)
class BinaryCounts:
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    ignored_pixels: int
    evaluated_pixels: int


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


def merge_counts(
    first: BinaryCounts,
    second: BinaryCounts,
) -> BinaryCounts:
    return BinaryCounts(
        true_positive=(
            first.true_positive
            + second.true_positive
        ),
        false_positive=(
            first.false_positive
            + second.false_positive
        ),
        false_negative=(
            first.false_negative
            + second.false_negative
        ),
        true_negative=(
            first.true_negative
            + second.true_negative
        ),
        ignored_pixels=(
            first.ignored_pixels
            + second.ignored_pixels
        ),
        evaluated_pixels=(
            first.evaluated_pixels
            + second.evaluated_pixels
        ),
    )


def calculate_patch_counts(
    probabilities: Tensor,
    targets: Tensor,
    valid_mask: Tensor,
    threshold: float,
) -> BinaryCounts:
    if probabilities.shape != targets.shape:
        raise ValueError(
            "Probability and target shapes do not match."
        )

    if probabilities.shape != valid_mask.shape:
        raise ValueError(
            "Probability and valid-mask shapes do not match."
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "Threshold must be between zero and one."
        )

    if not torch.isfinite(
        probabilities
    ).all():
        raise ValueError(
            "Probabilities contain non-finite values."
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

    evaluated_pixels = int(
        torch.count_nonzero(
            valid
        ).item()
    )

    ignored_pixels = int(
        valid.numel()
        - evaluated_pixels
    )

    if evaluated_pixels <= 0:
        raise ValueError(
            "No valid pixels are available."
        )

    true_positive = int(
        torch.count_nonzero(
            valid
            & predictions
            & truth
        ).item()
    )

    false_positive = int(
        torch.count_nonzero(
            valid
            & predictions
            & ~truth
        ).item()
    )

    false_negative = int(
        torch.count_nonzero(
            valid
            & ~predictions
            & truth
        ).item()
    )

    true_negative = int(
        torch.count_nonzero(
            valid
            & ~predictions
            & ~truth
        ).item()
    )

    counted_pixels = (
        true_positive
        + false_positive
        + false_negative
        + true_negative
    )

    if counted_pixels != evaluated_pixels:
        raise RuntimeError(
            "Confusion counts do not match valid pixels."
        )

    return BinaryCounts(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        true_negative=true_negative,
        ignored_pixels=ignored_pixels,
        evaluated_pixels=evaluated_pixels,
    )


def counts_to_metrics(
    counts: BinaryCounts,
) -> dict[str, float | int]:
    precision = safe_divide(
        counts.true_positive,
        (
            counts.true_positive
            + counts.false_positive
        ),
    )

    recall = safe_divide(
        counts.true_positive,
        (
            counts.true_positive
            + counts.false_negative
        ),
    )

    f1 = safe_divide(
        2
        * counts.true_positive,
        (
            2
            * counts.true_positive
            + counts.false_positive
            + counts.false_negative
        ),
    )

    iou = safe_divide(
        counts.true_positive,
        (
            counts.true_positive
            + counts.false_positive
            + counts.false_negative
        ),
    )

    accuracy = safe_divide(
        (
            counts.true_positive
            + counts.true_negative
        ),
        counts.evaluated_pixels,
    )

    change_prevalence = safe_divide(
        (
            counts.true_positive
            + counts.false_negative
        ),
        counts.evaluated_pixels,
    )

    predicted_change_fraction = safe_divide(
        (
            counts.true_positive
            + counts.false_positive
        ),
        counts.evaluated_pixels,
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "accuracy": accuracy,
        "change_prevalence": change_prevalence,
        "predicted_change_fraction": predicted_change_fraction,
        "true_positive": counts.true_positive,
        "false_positive": counts.false_positive,
        "false_negative": counts.false_negative,
        "true_negative": counts.true_negative,
        "ignored_pixels": counts.ignored_pixels,
        "evaluated_pixels": counts.evaluated_pixels,
    }


def calculate_macro_metrics(
    per_region: Sequence[
        Mapping[str, float | int | str]
    ],
) -> dict[str, dict[str, float]]:
    if not per_region:
        raise ValueError(
            "Per-region metrics cannot be empty."
        )

    metric_names = (
        "precision",
        "recall",
        "f1",
        "iou",
        "accuracy",
        "change_prevalence",
        "predicted_change_fraction",
    )

    mean_metrics: dict[
        str,
        float,
    ] = {}

    median_metrics: dict[
        str,
        float,
    ] = {}

    for metric_name in metric_names:
        values = [
            float(
                region[
                    metric_name
                ]
            )
            for region in per_region
        ]

        mean_metrics[
            metric_name
        ] = float(
            fmean(
                values
            )
        )

        median_metrics[
            metric_name
        ] = float(
            median(
                values
            )
        )

    return {
        "mean": mean_metrics,
        "median": median_metrics,
    }


def evaluate_official_test(
    dataset: OSCDTestDataset,
    model: torch.nn.Module,
    threshold: float,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    if threshold != FROZEN_THRESHOLD:
        raise RuntimeError(
            "Official evaluation threshold must remain frozen."
        )

    if batch_size <= 0:
        raise ValueError(
            "Batch size must be positive."
        )

    if num_workers < 0:
        raise ValueError(
            "Number of workers cannot be negative."
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

    zero_counts = BinaryCounts(
        true_positive=0,
        false_positive=0,
        false_negative=0,
        true_negative=0,
        ignored_pixels=0,
        evaluated_pixels=0,
    )

    region_counts = {
        region: zero_counts
        for region in EXPECTED_TEST_REGIONS
    }

    total_counts = zero_counts
    inferred_patches = 0

    model.eval()

    with torch.inference_mode():
        for batch in loader:
            before = batch[
                "before"
            ].to(
                device,
                non_blocking=(
                    device.type
                    == "cuda"
                ),
            )

            after = batch[
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
                    before,
                    after,
                )

            probabilities = torch.sigmoid(
                logits.float()
            ).cpu()

            targets = batch[
                "mask"
            ].cpu()

            valid_masks = batch[
                "valid_mask"
            ].cpu()

            regions = tuple(
                str(region)
                for region in batch[
                    "region"
                ]
            )

            if probabilities.shape != targets.shape:
                raise RuntimeError(
                    "Model output and target shapes do not match."
                )

            if probabilities.shape != valid_masks.shape:
                raise RuntimeError(
                    "Model output and valid-mask shapes do not match."
                )

            if probabilities.shape[
                0
            ] != len(
                regions
            ):
                raise RuntimeError(
                    "Batch region metadata is inconsistent."
                )

            for index, region in enumerate(
                regions
            ):
                if region not in region_counts:
                    raise RuntimeError(
                        f"Unexpected official test region: {region}"
                    )

                patch_counts = calculate_patch_counts(
                    probabilities=probabilities[
                        index
                    ],
                    targets=targets[
                        index
                    ],
                    valid_mask=valid_masks[
                        index
                    ],
                    threshold=threshold,
                )

                region_counts[
                    region
                ] = merge_counts(
                    region_counts[
                        region
                    ],
                    patch_counts,
                )

                total_counts = merge_counts(
                    total_counts,
                    patch_counts,
                )

                inferred_patches += 1

    if inferred_patches != len(
        dataset
    ):
        raise RuntimeError(
            "Not every official test patch was inferred."
        )

    audits = {
        audit.region: audit
        for audit in dataset.region_audits()
    }

    per_region: list[
        dict[str, float | int | str]
    ] = []

    for region in EXPECTED_TEST_REGIONS:
        metrics = counts_to_metrics(
            region_counts[
                region
            ]
        )

        audit = audits[
            region
        ]

        if int(
            metrics[
                "evaluated_pixels"
            ]
        ) != audit.valid_pixels:
            raise RuntimeError(
                f"Evaluated-pixel mismatch for {region}."
            )

        per_region.append(
            {
                "region": region,
                "height": audit.height,
                "width": audit.width,
                "patch_count": audit.patch_count,
                **metrics,
            }
        )

    if total_counts.evaluated_pixels != 3_077_936:
        raise RuntimeError(
            "Official test evaluated-pixel count is incorrect."
        )

    return {
        "threshold": threshold,
        "device": str(
            device
        ),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "inferred_patches": inferred_patches,
        "micro": counts_to_metrics(
            total_counts
        ),
        "macro": calculate_macro_metrics(
            per_region
        ),
        "per_region": per_region,
    }


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

    mode = parser.add_mutually_exclusive_group(
        required=True
    )

    mode.add_argument(
        "--audit-only",
        action="store_true",
    )

    mode.add_argument(
        "--run-evaluation",
        action="store_true",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/week6/test_results.json"
        ),
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

    audits = dataset.region_audits()

    dataset_payload = {
        "region_count": len(
            dataset.regions
        ),
        "patch_count": len(
            dataset
        ),
        "valid_pixel_count": sum(
            audit.valid_pixels
            for audit in audits
        ),
        "covered_valid_pixel_count": sum(
            audit.covered_valid_pixels
            for audit in audits
        ),
        "padded_patch_count": sum(
            audit.padded_patch_count
            for audit in audits
        ),
        "regions_requiring_padding": sum(
            audit.requires_padding
            for audit in audits
        ),
        "regions": [
            asdict(
                audit
            )
            for audit in audits
        ],
    }

    if arguments.audit_only:
        payload = {
            "protocol": asdict(
                frozen
            ),
            "dataset": dataset_payload,
            "access": {
                "test_image_pixels_accessed": False,
                "test_label_pixels_accessed": False,
                "model_inference_executed": False,
                "test_metrics_calculated": False,
                "threshold_modified": False,
                "checkpoint_modified": False,
            },
        }

        print(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
            )
        )

        return 0

    if arguments.output.exists():
        raise FileExistsError(
            "Official test results already exist. "
            "The frozen evaluation cannot be overwritten."
        )

    pending_output = arguments.output.with_suffix(
        arguments.output.suffix
        + ".tmp"
    )

    if pending_output.exists():
        raise FileExistsError(
            f"Pending evaluation output already exists: {pending_output}"
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

    config = load_training_config(
        arguments.config
    )

    model = build_model(
        config=config,
        device=device,
        disable_pretrained=True,
    )

    checkpoint: Mapping[str, Any] = torch.load(
        arguments.checkpoint,
        map_location=device,
        weights_only=False,
    )

    if int(
        checkpoint["epoch"]
    ) != FROZEN_CHECKPOINT_EPOCH:
        raise RuntimeError(
            "Loaded checkpoint epoch is not frozen epoch 24."
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    evaluation = evaluate_official_test(
        dataset=dataset,
        model=model,
        threshold=frozen.threshold,
        device=device,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
    )

    payload = {
        "protocol": asdict(
            frozen
        ),
        "dataset": dataset_payload,
        "evaluation": evaluation,
        "access": {
            "test_image_pixels_accessed": True,
            "test_label_pixels_accessed": True,
            "model_inference_executed": True,
            "test_metrics_calculated": True,
            "threshold_applied": True,
            "threshold_modified": False,
            "checkpoint_modified": False,
            "test_data_used_for_tuning": False,
        },
    }

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
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary_output.replace(
        arguments.output
    )

    micro = evaluation[
        "micro"
    ]

    macro = evaluation[
        "macro"
    ]

    print("GeoWatch official test evaluation completed")
    print("  Output:", arguments.output)
    print("  Device:", evaluation["device"])
    print("  Regions:", dataset_payload["region_count"])
    print("  Patches:", evaluation["inferred_patches"])
    print("  Evaluated pixels:", micro["evaluated_pixels"])
    print("  Frozen threshold:", frozen.threshold)
    print("  Micro precision:", micro["precision"])
    print("  Micro recall:", micro["recall"])
    print("  Micro F1:", micro["f1"])
    print("  Micro IoU:", micro["iou"])
    print("  Macro mean F1:", macro["mean"]["f1"])
    print("  Macro median F1:", macro["median"]["f1"])
    print("  Macro mean IoU:", macro["mean"]["iou"])
    print("  Macro median IoU:", macro["median"]["iou"])

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
