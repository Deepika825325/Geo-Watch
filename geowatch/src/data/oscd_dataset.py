"""PyTorch dataset for OSCD training-region patches.

This Week 3 implementation intentionally supports only the official OSCD
training regions. Test-region images and labels are not enumerated or opened
during architecture and dataset debugging.

Rectified OSCD bands are read from ``imgs_1_rect`` and ``imgs_2_rect``.
The authoritative binary label is ``cm/cm.png``. Its alpha channel is
excluded before applying the OSCD rule:

    0 = unchanged
    >0 = changed

The ``*-cm.tif`` files are deliberately not used because they contain class
codes 1 and 2 rather than a zero-background binary mask.
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
import rasterio
import torch
from PIL import Image
from rasterio.errors import NotGeoreferencedWarning
from rasterio.windows import Window
from torch import Tensor
from torch.utils.data import Dataset


LOGGER = logging.getLogger(
    "geowatch.oscd_dataset"
)

DEFAULT_BANDS = (
    "B02",
    "B03",
    "B04",
    "B08",
)

SUPPORTED_BANDS = (
    "B02",
    "B03",
    "B04",
    "B08",
    "B11",
    "B12",
)

IMAGES_DIRECTORY_NAME = (
    "Onera Satellite Change Detection dataset - Images"
)

TRAIN_LABELS_DIRECTORY_NAME = (
    "Onera Satellite Change Detection dataset - Train Labels"
)


class OSCDDatasetError(RuntimeError):
    """Raised when OSCD data violates the expected training contract."""


class OSCDSample(TypedDict):
    """One training patch returned by :class:`OSCDTrainingDataset`."""

    before: Tensor
    after: Tensor
    mask: Tensor
    region: str
    patch_id: str
    row: int
    column: int


PairedTransform = Callable[
    [Tensor, Tensor, Tensor],
    tuple[Tensor, Tensor, Tensor],
]


@dataclass(frozen=True)
class RegionRecord:
    """Validated file paths and dimensions for one training region."""

    name: str
    before_band_paths: tuple[Path, ...]
    after_band_paths: tuple[Path, ...]
    label_path: Path
    height: int
    width: int


@dataclass(frozen=True)
class PatchRecord:
    """Deterministic spatial window belonging to one training region."""

    region_index: int
    row: int
    column: int
    height: int
    width: int

    @property
    def patch_id(self) -> str:
        """Return a stable identifier independent of dataset ordering."""
        return (
            f"r{self.region_index:02d}"
            f"_y{self.row:05d}"
            f"_x{self.column:05d}"
            f"_h{self.height}"
            f"_w{self.width}"
        )


def normalize_band_names(
    band_names: Sequence[str],
) -> tuple[str, ...]:
    """Validate and normalize an ordered Sentinel-2 band list."""
    normalized = tuple(
        str(band).strip().upper()
        for band in band_names
    )

    if not normalized:
        raise OSCDDatasetError(
            "At least one input band is required."
        )

    if len(set(normalized)) != len(normalized):
        raise OSCDDatasetError(
            f"Band names must be unique; received {normalized}."
        )

    unsupported = sorted(
        set(normalized).difference(
            SUPPORTED_BANDS
        )
    )

    if unsupported:
        raise OSCDDatasetError(
            f"Unsupported OSCD bands: {unsupported}. "
            f"Supported bands are {SUPPORTED_BANDS}."
        )

    if len(normalized) not in (
        4,
        6,
    ):
        raise OSCDDatasetError(
            "GeoWatch currently expects either four or six bands; "
            f"received {len(normalized)}."
        )

    required_visible_bands = {
        "B02",
        "B03",
        "B04",
    }

    missing_visible = sorted(
        required_visible_bands.difference(
            normalized
        )
    )

    if missing_visible:
        raise OSCDDatasetError(
            "ImageNet encoder adaptation requires B02, B03 and B04. "
            f"Missing bands: {missing_visible}."
        )

    return normalized


def generate_patch_starts(
    length: int,
    patch_size: int,
    stride: int,
) -> tuple[int, ...]:
    """Generate deterministic starts while covering the final image edge.

    The final start is anchored at ``length - patch_size`` when the normal
    stride sequence does not exactly reach the image boundary. This avoids
    dropping rightmost or bottommost pixels.
    """
    if length <= 0:
        raise OSCDDatasetError(
            f"Spatial length must be positive; received {length}."
        )

    if patch_size <= 0:
        raise OSCDDatasetError(
            "patch_size must be greater than zero."
        )

    if stride <= 0:
        raise OSCDDatasetError(
            "stride must be greater than zero."
        )

    if length < patch_size:
        raise OSCDDatasetError(
            f"Region dimension {length} is smaller than "
            f"patch size {patch_size}."
        )

    starts = list(
        range(
            0,
            length - patch_size + 1,
            stride,
        )
    )

    final_start = length - patch_size

    if not starts:
        starts.append(
            final_start
        )
    elif starts[-1] != final_start:
        starts.append(
            final_start
        )

    return tuple(
        starts
    )


def read_raster_shape(
    path: Path,
) -> tuple[int, int]:
    """Read one raster's height and width without loading all pixels."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Required OSCD raster does not exist: {path}"
        )

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            NotGeoreferencedWarning,
        )

        with rasterio.open(path) as dataset:
            if dataset.count != 1:
                raise OSCDDatasetError(
                    f"Expected one raster band in {path}; "
                    f"found {dataset.count}."
                )

            return (
                int(dataset.height),
                int(dataset.width),
            )


def read_label_shape(
    path: Path,
) -> tuple[int, int]:
    """Read the height and width of the authoritative PNG label."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Required OSCD PNG label does not exist: {path}"
        )

    with Image.open(path) as image:
        width, height = image.size

    return (
        int(height),
        int(width),
    )


def build_region_record(
    images_root: Path,
    labels_root: Path,
    region_name: str,
    band_names: tuple[str, ...],
) -> RegionRecord:
    """Validate and index one official OSCD training region."""
    region_images = (
        images_root
        / region_name
    )
    region_labels = (
        labels_root
        / region_name
    )

    if not region_images.is_dir():
        raise OSCDDatasetError(
            f"Missing OSCD image region: {region_images}"
        )

    if not region_labels.is_dir():
        raise OSCDDatasetError(
            f"Missing OSCD training-label region: {region_labels}"
        )

    before_directory = (
        region_images
        / "imgs_1_rect"
    )
    after_directory = (
        region_images
        / "imgs_2_rect"
    )

    if not before_directory.is_dir():
        raise OSCDDatasetError(
            f"Missing rectified first-date directory: {before_directory}"
        )

    if not after_directory.is_dir():
        raise OSCDDatasetError(
            f"Missing rectified second-date directory: {after_directory}"
        )

    before_paths = tuple(
        before_directory
        / f"{band}.tif"
        for band in band_names
    )
    after_paths = tuple(
        after_directory
        / f"{band}.tif"
        for band in band_names
    )

    label_path = (
        region_labels
        / "cm"
        / "cm.png"
    )

    raster_paths = (
        *before_paths,
        *after_paths,
    )

    shapes = {
        read_raster_shape(path)
        for path in raster_paths
    }

    if len(shapes) != 1:
        raise OSCDDatasetError(
            f"Rectified rasters are not shape-aligned in {region_name}: "
            f"{sorted(shapes)}"
        )

    height, width = next(
        iter(shapes)
    )

    label_shape = read_label_shape(
        label_path
    )

    if label_shape != (
        height,
        width,
    ):
        raise OSCDDatasetError(
            f"Label shape {label_shape} does not match imagery "
            f"shape {(height, width)} in {region_name}."
        )

    return RegionRecord(
        name=region_name,
        before_band_paths=before_paths,
        after_band_paths=after_paths,
        label_path=label_path,
        height=height,
        width=width,
    )


def read_band_window(
    path: Path,
    patch: PatchRecord,
    reflectance_scale: float,
    clip_minimum: float | None,
    clip_maximum: float | None,
) -> np.ndarray:
    """Read and normalize one rectified spectral patch."""
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

        with rasterio.open(path) as dataset:
            array = dataset.read(
                1,
                window=window,
                out_dtype="float32",
            )

    expected_shape = (
        patch.height,
        patch.width,
    )

    if array.shape != expected_shape:
        raise OSCDDatasetError(
            f"Unexpected raster window shape from {path}. "
            f"Expected {expected_shape}; found {array.shape}."
        )

    if not np.isfinite(
        array
    ).all():
        raise OSCDDatasetError(
            f"Non-finite raster values found in {path}."
        )

    array /= np.float32(
        reflectance_scale
    )

    if clip_minimum is not None:
        np.maximum(
            array,
            np.float32(clip_minimum),
            out=array,
        )

    if clip_maximum is not None:
        np.minimum(
            array,
            np.float32(clip_maximum),
            out=array,
        )

    return array


def read_mask_window(
    path: Path,
    patch: PatchRecord,
) -> np.ndarray:
    """Read one binary label patch while excluding any alpha channel."""
    crop_box = (
        patch.column,
        patch.row,
        patch.column + patch.width,
        patch.row + patch.height,
    )

    with Image.open(path) as image:
        cropped = image.crop(
            crop_box
        )

        if cropped.mode in (
            "RGBA",
            "LA",
        ):
            cropped = cropped.convert(
                "RGB"
            )

        array = np.asarray(
            cropped
        )

    expected_spatial_shape = (
        patch.height,
        patch.width,
    )

    if array.shape[:2] != expected_spatial_shape:
        raise OSCDDatasetError(
            f"Unexpected mask window shape from {path}. "
            f"Expected {expected_spatial_shape}; found {array.shape}."
        )

    if array.ndim == 2:
        binary = array > 0
    elif array.ndim == 3:
        colour_channels = array[
            ...,
            :3,
        ]

        binary = np.any(
            colour_channels > 0,
            axis=-1,
        )
    else:
        raise OSCDDatasetError(
            f"Unsupported label dimensions in {path}: {array.shape}"
        )

    return binary.astype(
        np.float32,
        copy=False,
    )


class OSCDTrainingDataset(
    Dataset[OSCDSample]
):
    """Patch-level PyTorch dataset over official OSCD training regions.

    The class deliberately discovers regions only from the OSCD training
    label directory. It cannot accidentally enumerate the test-label
    directory during Week 3.
    """

    def __init__(
        self,
        raw_root: Path | str,
        region_names: Sequence[str] | None = None,
        band_names: Sequence[str] = DEFAULT_BANDS,
        patch_size: int = 256,
        stride: int = 256,
        reflectance_scale: float = 10_000.0,
        clip_minimum: float | None = 0.0,
        clip_maximum: float | None = 1.0,
        transform: PairedTransform | None = None,
    ) -> None:
        """Initialize the deterministic training-patch index.

        Args:
            raw_root: Extracted OSCD raw directory.
            region_names: Explicit subset of official training regions.
                ``None`` uses all 14 training regions. A geographic
                train/validation split should be supplied explicitly rather
                than generated randomly inside this class.
            band_names: Ordered four-band or six-band input configuration.
            patch_size: Square patch height and width.
            stride: Spatial distance between patch starts.
            reflectance_scale: Divisor converting integer Sentinel values
                into approximate reflectance.
            clip_minimum: Optional minimum normalized value.
            clip_maximum: Optional maximum normalized value.
            transform: Optional paired transform applied jointly to before,
                after and mask tensors.
        """
        super().__init__()

        self.raw_root = Path(
            raw_root
        )
        self.band_names = normalize_band_names(
            band_names
        )

        if patch_size <= 0:
            raise OSCDDatasetError(
                "patch_size must be greater than zero."
            )

        if stride <= 0:
            raise OSCDDatasetError(
                "stride must be greater than zero."
            )

        if reflectance_scale <= 0:
            raise OSCDDatasetError(
                "reflectance_scale must be greater than zero."
            )

        if (
            clip_minimum is not None
            and clip_maximum is not None
            and clip_minimum >= clip_maximum
        ):
            raise OSCDDatasetError(
                "clip_minimum must be smaller than clip_maximum."
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
        self.transform = transform

        images_root = (
            self.raw_root
            / IMAGES_DIRECTORY_NAME
        )
        labels_root = (
            self.raw_root
            / TRAIN_LABELS_DIRECTORY_NAME
        )

        if not images_root.is_dir():
            raise FileNotFoundError(
                f"OSCD images directory does not exist: {images_root}"
            )

        if not labels_root.is_dir():
            raise FileNotFoundError(
                "OSCD training-label directory does not exist: "
                f"{labels_root}"
            )

        available_regions = tuple(
            sorted(
                path.name
                for path in labels_root.iterdir()
                if path.is_dir()
            )
        )

        if len(available_regions) != 14:
            raise OSCDDatasetError(
                "Expected exactly 14 official OSCD training regions; "
                f"found {len(available_regions)}."
            )

        if region_names is None:
            selected_regions = available_regions
        else:
            selected_regions = tuple(
                str(region).strip()
                for region in region_names
            )

            if not selected_regions:
                raise OSCDDatasetError(
                    "region_names must not be empty."
                )

            if len(
                set(selected_regions)
            ) != len(selected_regions):
                raise OSCDDatasetError(
                    "region_names must not contain duplicates."
                )

            unknown_regions = sorted(
                set(selected_regions).difference(
                    available_regions
                )
            )

            if unknown_regions:
                raise OSCDDatasetError(
                    "Requested regions are not official OSCD training "
                    f"regions: {unknown_regions}"
                )

        self.region_names = tuple(
            selected_regions
        )

        self.regions = tuple(
            build_region_record(
                images_root=images_root,
                labels_root=labels_root,
                region_name=region_name,
                band_names=self.band_names,
            )
            for region_name in self.region_names
        )

        patch_records: list[
            PatchRecord
        ] = []

        for region_index, region in enumerate(
            self.regions
        ):
            row_starts = generate_patch_starts(
                length=region.height,
                patch_size=self.patch_size,
                stride=self.stride,
            )
            column_starts = generate_patch_starts(
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
            raise OSCDDatasetError(
                "The OSCD patch index is empty."
            )

        self.patches = tuple(
            patch_records
        )

    def __len__(
        self,
    ) -> int:
        """Return the number of indexed training patches."""
        return len(
            self.patches
        )

    def __iter__(
        self,
    ) -> Iterator[OSCDSample]:
        """Iterate over samples using standard dataset indexing."""
        for index in range(
            len(self)
        ):
            yield self[
                index
            ]

    def __getitem__(
        self,
        index: int,
    ) -> OSCDSample:
        """Load one normalized bi-temporal patch and binary mask."""
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
                f"OSCD dataset index out of range: {index}"
            )

        patch = self.patches[
            index
        ]
        region = self.regions[
            patch.region_index
        ]

        before_arrays = [
            read_band_window(
                path=path,
                patch=patch,
                reflectance_scale=self.reflectance_scale,
                clip_minimum=self.clip_minimum,
                clip_maximum=self.clip_maximum,
            )
            for path in region.before_band_paths
        ]

        after_arrays = [
            read_band_window(
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
                np.newaxis,
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
            raise OSCDDatasetError(
                f"Unexpected before tensor shape: {tuple(before.shape)}"
            )

        if tuple(
            after.shape
        ) != expected_image_shape:
            raise OSCDDatasetError(
                f"Unexpected after tensor shape: {tuple(after.shape)}"
            )

        if tuple(
            mask.shape
        ) != expected_mask_shape:
            raise OSCDDatasetError(
                f"Unexpected mask tensor shape: {tuple(mask.shape)}"
            )

        if self.transform is not None:
            before, after, mask = self.transform(
                before,
                after,
                mask,
            )

        return OSCDSample(
            before=before,
            after=after,
            mask=mask,
            region=region.name,
            patch_id=(
                f"{region.name}_{patch.patch_id}"
            ),
            row=patch.row,
            column=patch.column,
        )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the training-only dataset audit CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Audit the GeoWatch OSCD training-only PyTorch dataset."
        )
    )

    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path(
            "data/benchmark/oscd/raw"
        ),
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=None,
        help=(
            "Optional explicit subset of official training regions."
        ),
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        default=list(
            DEFAULT_BANDS
        ),
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
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
    """Run a non-test OSCD training sample audit."""
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
        dataset = OSCDTrainingDataset(
            raw_root=args.raw_root,
            region_names=args.regions,
            band_names=args.bands,
            patch_size=args.patch_size,
            stride=args.stride,
        )

        sample = dataset[
            args.sample_index
        ]

        before = sample[
            "before"
        ]
        after = sample[
            "after"
        ]
        mask = sample[
            "mask"
        ]

        mask_values = torch.unique(
            mask
        ).tolist()

        if not set(
            mask_values
        ).issubset(
            {
                0.0,
                1.0,
            }
        ):
            raise OSCDDatasetError(
                f"Mask is not binary: {mask_values}"
            )

        if not torch.isfinite(
            before
        ).all():
            raise OSCDDatasetError(
                "Before tensor contains non-finite values."
            )

        if not torch.isfinite(
            after
        ).all():
            raise OSCDDatasetError(
                "After tensor contains non-finite values."
            )

        print(
            "GeoWatch OSCD training dataset audit passed"
        )
        print(
            "  Official training regions available:",
            14,
        )
        print(
            "  Selected training regions:",
            len(
                dataset.region_names
            ),
        )
        print(
            "  Training patches:",
            len(
                dataset
            ),
        )
        print(
            "  Bands:",
            dataset.band_names,
        )
        print(
            "  Patch size:",
            dataset.patch_size,
        )
        print(
            "  Stride:",
            dataset.stride,
        )
        print(
            "  Sample region:",
            sample["region"],
        )
        print(
            "  Sample patch ID:",
            sample["patch_id"],
        )
        print(
            "  Before shape:",
            tuple(
                before.shape
            ),
        )
        print(
            "  After shape:",
            tuple(
                after.shape
            ),
        )
        print(
            "  Mask shape:",
            tuple(
                mask.shape
            ),
        )
        print(
            "  Before range:",
            (
                float(
                    before.min().item()
                ),
                float(
                    before.max().item()
                ),
            ),
        )
        print(
            "  After range:",
            (
                float(
                    after.min().item()
                ),
                float(
                    after.max().item()
                ),
            ),
        )
        print(
            "  Mask values:",
            mask_values,
        )
        print(
            "  Patch change fraction:",
            float(
                mask.mean().item()
            ),
        )
        print(
            "  Test images accessed:",
            False,
        )
        print(
            "  Test labels accessed:",
            False,
        )

        return 0

    except (
        FileNotFoundError,
        PermissionError,
        OSCDDatasetError,
        IndexError,
        TypeError,
        ValueError,
        OSError,
    ) as error:
        LOGGER.error(
            "%s",
            error,
        )
        return 1

    except Exception:
        LOGGER.exception(
            "Unexpected OSCD dataset-audit failure."
        )
        return 1


if __name__ == "__main__":
    sys.exit(
        main()
    )
