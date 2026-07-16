"""Tests for the GeoWatch OSCD training dataset.

Tests create a temporary OSCD-style training structure. No OSCD test image
or test-label directory is created, enumerated or accessed.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from PIL import Image
from rasterio.transform import from_origin
from torch.utils.data import DataLoader

from src.data.oscd_dataset import (
    IMAGES_DIRECTORY_NAME,
    TRAIN_LABELS_DIRECTORY_NAME,
    OSCDDatasetError,
    OSCDTrainingDataset,
    PatchRecord,
    generate_patch_starts,
    normalize_band_names,
    read_mask_window,
)
from src.models.siamese_unet import SiameseUNet


SYNTHETIC_REGION_NAMES = tuple(
    f"region_{index:02d}"
    for index in range(14)
)

SYNTHETIC_BANDS = (
    "B02",
    "B03",
    "B04",
    "B08",
    "B11",
    "B12",
)

IMAGE_SIZE = 64
PATCH_SIZE = 32


def write_single_band_raster(
    path: Path,
    array: np.ndarray,
) -> None:
    """Write one georeferenced single-band uint16 raster."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if array.ndim != 2:
        raise ValueError(
            f"Expected a two-dimensional array; received {array.shape}."
        )

    with rasterio.open(
        path,
        mode="w",
        driver="GTiff",
        height=int(array.shape[0]),
        width=int(array.shape[1]),
        count=1,
        dtype=array.dtype,
        crs="EPSG:32644",
        transform=from_origin(
            500_000.0,
            2_000_000.0,
            10.0,
            10.0,
        ),
    ) as dataset:
        dataset.write(
            array,
            1,
        )


def create_rgba_change_mask(
    path: Path,
) -> None:
    """Create an RGBA label whose alpha channel is nonzero everywhere."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    mask = np.zeros(
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
            4,
        ),
        dtype=np.uint8,
    )

    # Fully opaque alpha must not be interpreted as change.
    mask[
        ...,
        3,
    ] = 255

    # A 16x16 changed block lies inside the first 32x32 patch.
    mask[
        8:24,
        8:24,
        0,
    ] = 255

    Image.fromarray(
        mask,
        mode="RGBA",
    ).save(
        path
    )


@pytest.fixture()
def synthetic_oscd_root(
    tmp_path: Path,
) -> Iterator[Path]:
    """Create a minimal 14-region OSCD training-only directory contract."""
    raw_root = (
        tmp_path
        / "oscd"
        / "raw"
    )

    images_root = (
        raw_root
        / IMAGES_DIRECTORY_NAME
    )
    labels_root = (
        raw_root
        / TRAIN_LABELS_DIRECTORY_NAME
    )

    # The production dataset validates that 14 training-region directories
    # exist. Only region_00 is selected and therefore needs complete imagery.
    for region_name in SYNTHETIC_REGION_NAMES:
        (
            labels_root
            / region_name
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    selected_region = SYNTHETIC_REGION_NAMES[0]

    before_directory = (
        images_root
        / selected_region
        / "imgs_1_rect"
    )
    after_directory = (
        images_root
        / selected_region
        / "imgs_2_rect"
    )

    for band_index, band_name in enumerate(
        SYNTHETIC_BANDS
    ):
        before_value = (
            1_000
            + band_index * 100
        )
        after_value = (
            2_000
            + band_index * 100
        )

        before_array = np.full(
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
            before_value,
            dtype=np.uint16,
        )
        after_array = np.full(
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
            after_value,
            dtype=np.uint16,
        )

        write_single_band_raster(
            before_directory
            / f"{band_name}.tif",
            before_array,
        )
        write_single_band_raster(
            after_directory
            / f"{band_name}.tif",
            after_array,
        )

    create_rgba_change_mask(
        labels_root
        / selected_region
        / "cm"
        / "cm.png"
    )

    yield raw_root


def test_patch_starts_cover_trailing_edge() -> None:
    """The patch index must not discard the final row or column."""
    starts = generate_patch_starts(
        length=65,
        patch_size=32,
        stride=32,
    )

    assert starts == (
        0,
        32,
        33,
    )
    assert starts[-1] + 32 == 65


def test_invalid_band_configuration_is_rejected() -> None:
    """Encoder-incompatible band selections must fail immediately."""
    with pytest.raises(
        OSCDDatasetError,
        match="requires B02, B03 and B04",
    ):
        normalize_band_names(
            (
                "B02",
                "B03",
                "B08",
                "B11",
            )
        )


def test_rgba_alpha_is_not_interpreted_as_change(
    synthetic_oscd_root: Path,
) -> None:
    """Opaque background alpha must not turn the entire mask positive."""
    label_path = (
        synthetic_oscd_root
        / TRAIN_LABELS_DIRECTORY_NAME
        / SYNTHETIC_REGION_NAMES[0]
        / "cm"
        / "cm.png"
    )

    patch = PatchRecord(
        region_index=0,
        row=0,
        column=0,
        height=PATCH_SIZE,
        width=PATCH_SIZE,
    )

    mask = read_mask_window(
        path=label_path,
        patch=patch,
    )

    assert mask.shape == (
        PATCH_SIZE,
        PATCH_SIZE,
    )
    assert set(
        np.unique(mask).tolist()
    ) == {
        0.0,
        1.0,
    }

    # The changed block is 16x16 inside a 32x32 patch.
    assert float(
        mask.mean()
    ) == pytest.approx(
        0.25
    )


def test_four_band_dataset_sample(
    synthetic_oscd_root: Path,
) -> None:
    """Four-band samples must match the Siamese model input contract."""
    dataset = OSCDTrainingDataset(
        raw_root=synthetic_oscd_root,
        region_names=(
            SYNTHETIC_REGION_NAMES[0],
        ),
        band_names=(
            "B02",
            "B03",
            "B04",
            "B08",
        ),
        patch_size=PATCH_SIZE,
        stride=PATCH_SIZE,
    )

    sample = dataset[0]

    before = sample["before"]
    after = sample["after"]
    mask = sample["mask"]

    assert len(dataset) == 4
    assert tuple(before.shape) == (
        4,
        PATCH_SIZE,
        PATCH_SIZE,
    )
    assert tuple(after.shape) == (
        4,
        PATCH_SIZE,
        PATCH_SIZE,
    )
    assert tuple(mask.shape) == (
        1,
        PATCH_SIZE,
        PATCH_SIZE,
    )

    assert before.dtype == torch.float32
    assert after.dtype == torch.float32
    assert mask.dtype == torch.float32

    assert torch.allclose(
        before[0],
        torch.full(
            (
                PATCH_SIZE,
                PATCH_SIZE,
            ),
            0.1,
        ),
    )
    assert torch.allclose(
        after[0],
        torch.full(
            (
                PATCH_SIZE,
                PATCH_SIZE,
            ),
            0.2,
        ),
    )

    assert set(
        torch.unique(
            mask
        ).tolist()
    ) == {
        0.0,
        1.0,
    }
    assert float(
        mask.mean().item()
    ) == pytest.approx(
        0.25
    )

    assert sample["region"] == (
        SYNTHETIC_REGION_NAMES[0]
    )
    assert sample["row"] == 0
    assert sample["column"] == 0


def test_six_band_dataset_sample(
    synthetic_oscd_root: Path,
) -> None:
    """Six-band samples must preserve the configured channel ordering."""
    dataset = OSCDTrainingDataset(
        raw_root=synthetic_oscd_root,
        region_names=(
            SYNTHETIC_REGION_NAMES[0],
        ),
        band_names=SYNTHETIC_BANDS,
        patch_size=PATCH_SIZE,
        stride=PATCH_SIZE,
    )

    sample = dataset[0]

    assert tuple(
        sample["before"].shape
    ) == (
        6,
        PATCH_SIZE,
        PATCH_SIZE,
    )
    assert tuple(
        sample["after"].shape
    ) == (
        6,
        PATCH_SIZE,
        PATCH_SIZE,
    )

    assert float(
        sample["before"][
            5,
            0,
            0,
        ].item()
    ) == pytest.approx(
        0.15
    )
    assert float(
        sample["after"][
            5,
            0,
            0,
        ].item()
    ) == pytest.approx(
        0.25
    )


def test_paired_transform_is_applied_consistently(
    synthetic_oscd_root: Path,
) -> None:
    """A transform must receive and return all three aligned tensors."""
    def paired_transform(
        before: torch.Tensor,
        after: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        return (
            before + 1.0,
            after + 2.0,
            1.0 - mask,
        )

    dataset = OSCDTrainingDataset(
        raw_root=synthetic_oscd_root,
        region_names=(
            SYNTHETIC_REGION_NAMES[0],
        ),
        patch_size=PATCH_SIZE,
        stride=PATCH_SIZE,
        transform=paired_transform,
    )

    sample = dataset[0]

    assert float(
        sample["before"][
            0,
            0,
            0,
        ].item()
    ) == pytest.approx(
        1.1
    )
    assert float(
        sample["after"][
            0,
            0,
            0,
        ].item()
    ) == pytest.approx(
        2.2
    )
    assert float(
        sample["mask"].mean().item()
    ) == pytest.approx(
        0.75
    )


def test_dataloader_batches_training_patches(
    synthetic_oscd_root: Path,
) -> None:
    """Default PyTorch collation must batch tensor and metadata fields."""
    dataset = OSCDTrainingDataset(
        raw_root=synthetic_oscd_root,
        region_names=(
            SYNTHETIC_REGION_NAMES[0],
        ),
        patch_size=PATCH_SIZE,
        stride=PATCH_SIZE,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    batch = next(
        iter(loader)
    )

    assert tuple(
        batch["before"].shape
    ) == (
        2,
        4,
        PATCH_SIZE,
        PATCH_SIZE,
    )
    assert tuple(
        batch["after"].shape
    ) == (
        2,
        4,
        PATCH_SIZE,
        PATCH_SIZE,
    )
    assert tuple(
        batch["mask"].shape
    ) == (
        2,
        1,
        PATCH_SIZE,
        PATCH_SIZE,
    )
    assert len(
        batch["patch_id"]
    ) == 2


def test_dataset_batch_matches_siamese_model_contract(
    synthetic_oscd_root: Path,
) -> None:
    """A dataset batch must pass directly through the Siamese U-Net."""
    dataset = OSCDTrainingDataset(
        raw_root=synthetic_oscd_root,
        region_names=(
            SYNTHETIC_REGION_NAMES[0],
        ),
        patch_size=PATCH_SIZE,
        stride=PATCH_SIZE,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    batch = next(
        iter(loader)
    )

    model = SiameseUNet(
        input_channels=4,
        pretrained_encoder=False,
        decoder_channels=(
            64,
            32,
            16,
            16,
        ),
        head_channels=8,
    )

    model.eval()

    with torch.inference_mode():
        logits = model(
            batch["before"],
            batch["after"],
        )

    assert tuple(
        logits.shape
    ) == (
        2,
        1,
        PATCH_SIZE,
        PATCH_SIZE,
    )
    assert torch.isfinite(
        logits
    ).all()


def test_unknown_training_region_is_rejected(
    synthetic_oscd_root: Path,
) -> None:
    """The dataset must not silently accept an unknown region."""
    with pytest.raises(
        OSCDDatasetError,
        match="not official OSCD training regions",
    ):
        OSCDTrainingDataset(
            raw_root=synthetic_oscd_root,
            region_names=(
                "unknown_region",
            ),
            patch_size=PATCH_SIZE,
            stride=PATCH_SIZE,
        )
