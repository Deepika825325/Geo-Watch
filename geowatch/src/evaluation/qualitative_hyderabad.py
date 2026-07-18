from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import rasterio
from rasterio.errors import NotGeoreferencedWarning

from src.evaluation.evaluate_test import evaluation_starts


FROZEN_BANDS = (
    "B02",
    "B03",
    "B04",
    "B08",
)

FROZEN_PATCH_SIZE = 256
FROZEN_STRIDE = 256
FROZEN_THRESHOLD = 0.76
FROZEN_CHECKPOINT_EPOCH = 24
FROZEN_CHECKPOINT_SHA256 = (
    "61e53ba86bc108d6ccbbb636c8da88c967e07da41471504878c74df6a903ea94"
)


class HyderabadInputError(RuntimeError):
    pass


@dataclass(frozen=True)
class RasterMetadata:
    path: str
    height: int
    width: int
    band_count: int
    dtype: str
    crs: str | None
    transform: tuple[float, ...]


@dataclass(frozen=True)
class HyderabadInputAudit:
    input_root: str
    bands: tuple[str, ...]
    height: int
    width: int
    dtype: str
    crs: str | None
    transform: tuple[float, ...]
    patch_size: int
    stride: int
    row_starts: tuple[int, ...]
    column_starts: tuple[int, ...]
    patch_count: int
    bottom_padding: int
    right_padding: int
    original_pixel_count: int
    padded_pixel_count: int
    before_files: tuple[str, ...]
    after_files: tuple[str, ...]


def normalize_transform(
    values: tuple[float, ...],
) -> tuple[float, ...]:
    return tuple(
        round(
            float(value),
            12,
        )
        for value in values
    )


def read_raster_metadata(
    path: Path,
) -> RasterMetadata:
    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            NotGeoreferencedWarning,
        )

        with rasterio.open(
            path
        ) as dataset:
            if dataset.count != 1:
                raise HyderabadInputError(
                    f"Expected one raster band in {path}; "
                    f"found {dataset.count}."
                )

            if dataset.height <= 0 or dataset.width <= 0:
                raise HyderabadInputError(
                    f"Invalid raster dimensions in {path}."
                )

            dtype = str(
                dataset.dtypes[
                    0
                ]
            )

            crs = (
                str(
                    dataset.crs
                )
                if dataset.crs is not None
                else None
            )

            transform = normalize_transform(
                tuple(
                    float(value)
                    for value in dataset.transform
                )
            )

            return RasterMetadata(
                path=str(
                    path
                ),
                height=int(
                    dataset.height
                ),
                width=int(
                    dataset.width
                ),
                band_count=int(
                    dataset.count
                ),
                dtype=dtype,
                crs=crs,
                transform=transform,
            )


def build_band_paths(
    directory: Path,
) -> tuple[Path, ...]:
    if not directory.is_dir():
        raise FileNotFoundError(
            directory
        )

    paths = tuple(
        directory
        / f"{band}.tif"
        for band in FROZEN_BANDS
    )

    missing = tuple(
        path
        for path in paths
        if not path.is_file()
    )

    if missing:
        raise FileNotFoundError(
            "Missing required Hyderabad bands: "
            + ", ".join(
                str(path)
                for path in missing
            )
        )

    return paths


def validate_metadata_alignment(
    metadata: tuple[RasterMetadata, ...],
) -> RasterMetadata:
    if not metadata:
        raise HyderabadInputError(
            "Raster metadata cannot be empty."
        )

    shapes = {
        (
            item.height,
            item.width,
        )
        for item in metadata
    }

    if len(
        shapes
    ) != 1:
        raise HyderabadInputError(
            "All Hyderabad before and after bands must "
            "have identical dimensions."
        )

    data_types = {
        item.dtype
        for item in metadata
    }

    if len(
        data_types
    ) != 1:
        raise HyderabadInputError(
            "All Hyderabad bands must use the same data type."
        )

    coordinate_systems = {
        item.crs
        for item in metadata
    }

    if len(
        coordinate_systems
    ) != 1:
        raise HyderabadInputError(
            "All Hyderabad bands must use consistent CRS metadata."
        )

    transforms = {
        item.transform
        for item in metadata
    }

    if len(
        transforms
    ) != 1:
        raise HyderabadInputError(
            "All Hyderabad bands must use identical raster transforms."
        )

    return metadata[
        0
    ]


def audit_hyderabad_inputs(
    input_root: Path,
    patch_size: int = FROZEN_PATCH_SIZE,
    stride: int = FROZEN_STRIDE,
) -> HyderabadInputAudit:
    if patch_size != FROZEN_PATCH_SIZE:
        raise HyderabadInputError(
            "Hyderabad patch size must remain frozen at 256."
        )

    if stride != FROZEN_STRIDE:
        raise HyderabadInputError(
            "Hyderabad stride must remain frozen at 256."
        )

    before_paths = build_band_paths(
        input_root
        / "before"
    )

    after_paths = build_band_paths(
        input_root
        / "after"
    )

    all_paths = (
        *before_paths,
        *after_paths,
    )

    metadata = tuple(
        read_raster_metadata(
            path
        )
        for path in all_paths
    )

    reference = validate_metadata_alignment(
        metadata
    )

    height = reference.height
    width = reference.width

    row_starts = evaluation_starts(
        length=height,
        patch_size=patch_size,
        stride=stride,
    )

    column_starts = evaluation_starts(
        length=width,
        patch_size=patch_size,
        stride=stride,
    )

    patch_count = (
        len(
            row_starts
        )
        * len(
            column_starts
        )
    )

    padded_height = (
        len(
            row_starts
        )
        * patch_size
    )

    padded_width = (
        len(
            column_starts
        )
        * patch_size
    )

    bottom_padding = (
        padded_height
        - height
    )

    right_padding = (
        padded_width
        - width
    )

    return HyderabadInputAudit(
        input_root=str(
            input_root
        ),
        bands=FROZEN_BANDS,
        height=height,
        width=width,
        dtype=reference.dtype,
        crs=reference.crs,
        transform=reference.transform,
        patch_size=patch_size,
        stride=stride,
        row_starts=row_starts,
        column_starts=column_starts,
        patch_count=patch_count,
        bottom_padding=bottom_padding,
        right_padding=right_padding,
        original_pixel_count=(
            height
            * width
        ),
        padded_pixel_count=(
            padded_height
            * padded_width
        ),
        before_files=tuple(
            str(path)
            for path in before_paths
        ),
        after_files=tuple(
            str(path)
            for path in after_paths
        ),
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
        "--patch-size",
        type=int,
        default=FROZEN_PATCH_SIZE,
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=FROZEN_STRIDE,
    )

    parser.add_argument(
        "--audit-only",
        action="store_true",
    )

    return parser


def main(
) -> int:
    arguments = build_argument_parser().parse_args()

    if not arguments.audit_only:
        raise RuntimeError(
            "Only Hyderabad input auditing is enabled in this step."
        )

    audit = audit_hyderabad_inputs(
        input_root=arguments.input_root,
        patch_size=arguments.patch_size,
        stride=arguments.stride,
    )

    payload = {
        "protocol": {
            "checkpoint_epoch": FROZEN_CHECKPOINT_EPOCH,
            "checkpoint_sha256": FROZEN_CHECKPOINT_SHA256,
            "threshold": FROZEN_THRESHOLD,
            "bands": list(
                FROZEN_BANDS
            ),
            "patch_size": FROZEN_PATCH_SIZE,
            "stride": FROZEN_STRIDE,
            "evaluation_role": "unlabelled_qualitative_only",
        },
        "input": asdict(
            audit
        ),
        "access": {
            "image_pixels_accessed": False,
            "model_inference_executed": False,
            "metrics_calculated": False,
            "official_test_results_modified": False,
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


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
