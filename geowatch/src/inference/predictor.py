from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from affine import Affine
from numpy.typing import NDArray
from rasterio.crs import CRS
from rasterio.windows import Window
from torch import Tensor, nn

from src.training.train import (
    build_model,
    load_training_config,
)


FROZEN_BANDS = (
    "B02",
    "B03",
    "B04",
    "B08",
)

FROZEN_THRESHOLD = 0.76
FROZEN_CHECKPOINT_EPOCH = 24
FROZEN_CHECKPOINT_SHA256 = (
    "61e53ba86bc108d6ccbbb636c8da88c967e07da41471504878c74df6a903ea94"
)

FROZEN_PATCH_SIZE = 256
FROZEN_STRIDE = 256
FROZEN_REFLECTANCE_SCALE = 10_000.0
FROZEN_CLIP_MINIMUM = 0.0
FROZEN_CLIP_MAXIMUM = 1.0


class FrozenProtocolError(RuntimeError):
    pass


class RasterPairError(RuntimeError):
    pass


@dataclass(frozen=True)
class RasterMetadata:
    height: int
    width: int
    dtype: str
    crs: CRS | None
    transform: Affine
    nodata: float | int | None


@dataclass(frozen=True)
class RasterPair:
    before: NDArray[np.float32]
    after: NDArray[np.float32]
    metadata: RasterMetadata
    before_files: tuple[str, ...]
    after_files: tuple[str, ...]


@dataclass(frozen=True)
class PredictionResult:
    probability: NDArray[np.float32]
    mask: NDArray[np.uint8]
    metadata: RasterMetadata
    threshold: float
    checkpoint_epoch: int
    checkpoint_sha256: str
    bands: tuple[str, ...]
    patch_size: int
    stride: int
    patch_count: int
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


def calculate_starts(
    length: int,
    patch_size: int,
    stride: int,
) -> tuple[int, ...]:
    if length <= 0:
        raise ValueError(
            "Raster dimension must be positive."
        )

    if patch_size <= 0:
        raise ValueError(
            "Patch size must be positive."
        )

    if stride != patch_size:
        raise ValueError(
            "Frozen inference requires stride equal to patch size."
        )

    return tuple(
        range(
            0,
            length,
            stride,
        )
    )


def resolve_band_paths(
    directory: Path,
    bands: tuple[str, ...] = FROZEN_BANDS,
) -> tuple[Path, ...]:
    if not directory.is_dir():
        raise FileNotFoundError(
            directory
        )

    paths = tuple(
        directory
        / f"{band}.tif"
        for band in bands
    )

    missing = tuple(
        path
        for path in paths
        if not path.is_file()
    )

    if missing:
        raise FileNotFoundError(
            "Missing required raster bands: "
            + ", ".join(
                str(path)
                for path in missing
            )
        )

    empty = tuple(
        path
        for path in paths
        if path.stat().st_size == 0
    )

    if empty:
        raise RasterPairError(
            "Empty raster bands: "
            + ", ".join(
                str(path)
                for path in empty
            )
        )

    return paths


def read_reference_metadata(
    path: Path,
) -> RasterMetadata:
    with rasterio.open(
        path
    ) as dataset:
        if dataset.count != 1:
            raise RasterPairError(
                f"Expected one raster band in {path}; "
                f"found {dataset.count}."
            )

        if dataset.height <= 0 or dataset.width <= 0:
            raise RasterPairError(
                f"Invalid raster dimensions in {path}."
            )

        return RasterMetadata(
            height=int(
                dataset.height
            ),
            width=int(
                dataset.width
            ),
            dtype=str(
                dataset.dtypes[
                    0
                ]
            ),
            crs=dataset.crs,
            transform=dataset.transform,
            nodata=dataset.nodata,
        )


def validate_band_metadata(
    path: Path,
    reference: RasterMetadata,
) -> None:
    metadata = read_reference_metadata(
        path
    )

    if metadata.height != reference.height:
        raise RasterPairError(
            f"Raster height mismatch: {path}"
        )

    if metadata.width != reference.width:
        raise RasterPairError(
            f"Raster width mismatch: {path}"
        )

    if metadata.crs != reference.crs:
        raise RasterPairError(
            f"Raster CRS mismatch: {path}"
        )

    if metadata.transform != reference.transform:
        raise RasterPairError(
            f"Raster transform mismatch: {path}"
        )

    if metadata.dtype != reference.dtype:
        raise RasterPairError(
            f"Raster data-type mismatch: {path}"
        )


def read_band_stack(
    paths: tuple[Path, ...],
    reference: RasterMetadata,
) -> NDArray[np.float32]:
    stack = np.empty(
        (
            len(
                paths
            ),
            reference.height,
            reference.width,
        ),
        dtype=np.float32,
    )

    for index, path in enumerate(
        paths
    ):
        validate_band_metadata(
            path,
            reference,
        )

        with rasterio.open(
            path
        ) as dataset:
            values = dataset.read(
                1,
                out_dtype="float32",
            )

        normalized = np.clip(
            values
            / FROZEN_REFLECTANCE_SCALE,
            FROZEN_CLIP_MINIMUM,
            FROZEN_CLIP_MAXIMUM,
        )

        stack[
            index
        ] = normalized.astype(
            np.float32,
            copy=False,
        )

    if not np.isfinite(
        stack
    ).all():
        raise RasterPairError(
            "Raster stack contains non-finite values."
        )

    return stack


def read_aligned_pair(
    before_directory: Path,
    after_directory: Path,
) -> RasterPair:
    before_paths = resolve_band_paths(
        before_directory
    )

    after_paths = resolve_band_paths(
        after_directory
    )

    reference = read_reference_metadata(
        before_paths[
            0
        ]
    )

    all_paths = (
        *before_paths,
        *after_paths,
    )

    for path in all_paths:
        validate_band_metadata(
            path,
            reference,
        )

    before = read_band_stack(
        before_paths,
        reference,
    )

    after = read_band_stack(
        after_paths,
        reference,
    )

    if before.shape != after.shape:
        raise RasterPairError(
            "Before and after raster stacks do not match."
        )

    return RasterPair(
        before=before,
        after=after,
        metadata=reference,
        before_files=tuple(
            str(path)
            for path in before_paths
        ),
        after_files=tuple(
            str(path)
            for path in after_paths
        ),
    )


def extract_padded_patch(
    array: NDArray[np.float32],
    row: int,
    column: int,
    patch_size: int,
) -> tuple[NDArray[np.float32], int, int]:
    if array.ndim != 3:
        raise ValueError(
            "Input array must have shape bands, height, width."
        )

    height = int(
        array.shape[
            1
        ]
    )

    width = int(
        array.shape[
            2
        ]
    )

    valid_height = min(
        patch_size,
        height
        - row,
    )

    valid_width = min(
        patch_size,
        width
        - column,
    )

    if valid_height <= 0 or valid_width <= 0:
        raise ValueError(
            "Patch origin is outside the raster."
        )

    patch = np.zeros(
        (
            int(
                array.shape[
                    0
                ]
            ),
            patch_size,
            patch_size,
        ),
        dtype=np.float32,
    )

    patch[
        :,
        :valid_height,
        :valid_width,
    ] = array[
        :,
        row:row
        + valid_height,
        column:column
        + valid_width,
    ]

    return (
        patch,
        valid_height,
        valid_width,
    )


def extract_logits(
    output: Any,
) -> Tensor:
    if isinstance(
        output,
        Tensor,
    ):
        logits = output
    elif isinstance(
        output,
        dict,
    ):
        candidate = output.get(
            "logits"
        )

        if not isinstance(
            candidate,
            Tensor,
        ):
            raise TypeError(
                "Model dictionary output does not contain tensor logits."
            )

        logits = candidate
    elif isinstance(
        output,
        (
            tuple,
            list,
        ),
    ):
        if not output:
            raise TypeError(
                "Model output sequence is empty."
            )

        candidate = output[
            0
        ]

        if not isinstance(
            candidate,
            Tensor,
        ):
            raise TypeError(
                "Model output sequence does not start with tensor logits."
            )

        logits = candidate
    else:
        raise TypeError(
            "Unsupported model output type."
        )

    if logits.ndim != 4:
        raise ValueError(
            "Model logits must have four dimensions."
        )

    if logits.shape[
        1
    ] != 1:
        raise ValueError(
            "Model logits must contain one output channel."
        )

    return logits


def run_tiled_inference(
    model: nn.Module,
    before: NDArray[np.float32],
    after: NDArray[np.float32],
    device: torch.device,
    batch_size: int,
    threshold: float = FROZEN_THRESHOLD,
    patch_size: int = FROZEN_PATCH_SIZE,
    stride: int = FROZEN_STRIDE,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.uint8],
    int,
]:
    if before.shape != after.shape:
        raise ValueError(
            "Before and after arrays must have identical shapes."
        )

    if before.ndim != 3:
        raise ValueError(
            "Input arrays must have shape bands, height, width."
        )

    if before.shape[
        0
    ] != len(
        FROZEN_BANDS
    ):
        raise ValueError(
            "Frozen model requires four input bands."
        )

    if batch_size <= 0:
        raise ValueError(
            "Batch size must be positive."
        )

    if threshold != FROZEN_THRESHOLD:
        raise FrozenProtocolError(
            "Inference threshold must remain frozen at 0.76."
        )

    height = int(
        before.shape[
            1
        ]
    )

    width = int(
        before.shape[
            2
        ]
    )

    row_starts = calculate_starts(
        height,
        patch_size,
        stride,
    )

    column_starts = calculate_starts(
        width,
        patch_size,
        stride,
    )

    coordinates = tuple(
        (
            row,
            column,
        )
        for row in row_starts
        for column in column_starts
    )

    probability = np.zeros(
        (
            height,
            width,
        ),
        dtype=np.float32,
    )

    model.eval()

    with torch.inference_mode():
        for offset in range(
            0,
            len(
                coordinates
            ),
            batch_size,
        ):
            batch_coordinates = coordinates[
                offset:offset
                + batch_size
            ]

            before_patches: list[
                NDArray[np.float32]
            ] = []

            after_patches: list[
                NDArray[np.float32]
            ] = []

            valid_shapes: list[
                tuple[int, int]
            ] = []

            for row, column in batch_coordinates:
                before_patch, valid_height, valid_width = (
                    extract_padded_patch(
                        before,
                        row,
                        column,
                        patch_size,
                    )
                )

                after_patch, after_height, after_width = (
                    extract_padded_patch(
                        after,
                        row,
                        column,
                        patch_size,
                    )
                )

                if after_height != valid_height:
                    raise RuntimeError(
                        "Before and after patch heights differ."
                    )

                if after_width != valid_width:
                    raise RuntimeError(
                        "Before and after patch widths differ."
                    )

                before_patches.append(
                    before_patch
                )

                after_patches.append(
                    after_patch
                )

                valid_shapes.append(
                    (
                        valid_height,
                        valid_width,
                    )
                )

            before_tensor = torch.from_numpy(
                np.stack(
                    before_patches
                )
            ).to(
                device,
                non_blocking=(
                    device.type
                    == "cuda"
                ),
            )

            after_tensor = torch.from_numpy(
                np.stack(
                    after_patches
                )
            ).to(
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
                raw_output = model(
                    before_tensor,
                    after_tensor,
                )

            logits = extract_logits(
                raw_output
            )

            expected_shape = (
                len(
                    batch_coordinates
                ),
                1,
                patch_size,
                patch_size,
            )

            if tuple(
                logits.shape
            ) != expected_shape:
                raise RuntimeError(
                    "Unexpected model-output shape: "
                    f"{tuple(logits.shape)}"
                )

            batch_probabilities = torch.sigmoid(
                logits.float()
            ).cpu().numpy()

            for index, (
                row,
                column,
            ) in enumerate(
                batch_coordinates
            ):
                valid_height, valid_width = valid_shapes[
                    index
                ]

                probability[
                    row:row
                    + valid_height,
                    column:column
                    + valid_width,
                ] = batch_probabilities[
                    index,
                    0,
                    :valid_height,
                    :valid_width,
                ]

    mask = (
        probability
        >= threshold
    ).astype(
        np.uint8,
        copy=False,
    )

    return (
        probability,
        mask,
        len(
            coordinates
        ),
    )


class FrozenChangePredictor:
    def __init__(
        self,
        config_path: Path = Path(
            "experiments/run_full/train_config.yaml"
        ),
        checkpoint_path: Path = Path(
            "experiments/run_full/checkpoints/"
            "best_model_epoch24.pt"
        ),
        device: str = "auto",
        batch_size: int = 8,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(
                "Batch size must be positive."
            )

        checkpoint_sha256 = calculate_sha256(
            checkpoint_path
        )

        if checkpoint_sha256 != FROZEN_CHECKPOINT_SHA256:
            raise FrozenProtocolError(
                "Frozen checkpoint SHA-256 mismatch."
            )

        if device == "auto":
            selected_device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            selected_device = torch.device(
                device
            )

        if (
            selected_device.type
            == "cuda"
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "CUDA was requested but is unavailable."
            )

        config = load_training_config(
            config_path
        )

        model = build_model(
            config=config,
            device=selected_device,
            disable_pretrained=True,
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=selected_device,
            weights_only=False,
        )

        if not isinstance(
            checkpoint,
            dict,
        ):
            raise FrozenProtocolError(
                "Frozen checkpoint must be a dictionary."
            )

        checkpoint_epoch = int(
            checkpoint.get(
                "epoch",
                -1,
            )
        )

        if checkpoint_epoch != FROZEN_CHECKPOINT_EPOCH:
            raise FrozenProtocolError(
                "Frozen checkpoint epoch mismatch."
            )

        state_dict = checkpoint.get(
            "model_state_dict"
        )

        if not isinstance(
            state_dict,
            dict,
        ):
            raise FrozenProtocolError(
                "Frozen checkpoint has no model state dictionary."
            )

        model.load_state_dict(
            state_dict,
            strict=True,
        )

        model.eval()

        self._model = model
        self._device = selected_device
        self._batch_size = batch_size
        self._checkpoint_path = checkpoint_path
        self._checkpoint_sha256 = checkpoint_sha256

    @property
    def device(
        self,
    ) -> torch.device:
        return self._device

    @property
    def threshold(
        self,
    ) -> float:
        return FROZEN_THRESHOLD

    @property
    def checkpoint_sha256(
        self,
    ) -> str:
        return self._checkpoint_sha256

    def predict_pair(
        self,
        before_directory: Path,
        after_directory: Path,
        qualitative: bool,
    ) -> PredictionResult:
        pair = read_aligned_pair(
            before_directory,
            after_directory,
        )

        probability, mask, patch_count = run_tiled_inference(
            model=self._model,
            before=pair.before,
            after=pair.after,
            device=self._device,
            batch_size=self._batch_size,
            threshold=FROZEN_THRESHOLD,
            patch_size=FROZEN_PATCH_SIZE,
            stride=FROZEN_STRIDE,
        )

        return PredictionResult(
            probability=probability,
            mask=mask,
            metadata=pair.metadata,
            threshold=FROZEN_THRESHOLD,
            checkpoint_epoch=FROZEN_CHECKPOINT_EPOCH,
            checkpoint_sha256=self._checkpoint_sha256,
            bands=FROZEN_BANDS,
            patch_size=FROZEN_PATCH_SIZE,
            stride=FROZEN_STRIDE,
            patch_count=patch_count,
            qualitative=qualitative,
        )


def write_probability_geotiff(
    result: PredictionResult,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    profile: dict[str, Any] = {
        "driver": "GTiff",
        "height": result.metadata.height,
        "width": result.metadata.width,
        "count": 1,
        "dtype": "float32",
        "crs": result.metadata.crs,
        "transform": result.metadata.transform,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
    }

    with rasterio.open(
        path,
        "w",
        **profile,
    ) as dataset:
        dataset.write(
            result.probability,
            1,
        )

        dataset.update_tags(
            qualitative=str(
                result.qualitative
            ).lower(),
            threshold=str(
                result.threshold
            ),
            checkpoint_epoch=str(
                result.checkpoint_epoch
            ),
            checkpoint_sha256=result.checkpoint_sha256,
            bands=",".join(
                result.bands
            ),
        )


def write_mask_geotiff(
    result: PredictionResult,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    profile: dict[str, Any] = {
        "driver": "GTiff",
        "height": result.metadata.height,
        "width": result.metadata.width,
        "count": 1,
        "dtype": "uint8",
        "crs": result.metadata.crs,
        "transform": result.metadata.transform,
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
    }

    with rasterio.open(
        path,
        "w",
        **profile,
    ) as dataset:
        dataset.write(
            result.mask,
            1,
        )

        dataset.update_tags(
            qualitative=str(
                result.qualitative
            ).lower(),
            threshold=str(
                result.threshold
            ),
            checkpoint_epoch=str(
                result.checkpoint_epoch
            ),
            checkpoint_sha256=result.checkpoint_sha256,
            bands=",".join(
                result.bands
            ),
        )
