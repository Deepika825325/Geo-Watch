from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from affine import Affine
from rasterio.crs import CRS
from torch import Tensor, nn

from src.inference.predictor import (
    FROZEN_BANDS,
    FROZEN_THRESHOLD,
    RasterPairError,
    calculate_starts,
    read_aligned_pair,
    run_tiled_inference,
)


class DifferenceModel(nn.Module):
    def forward(
        self,
        before: Tensor,
        after: Tensor,
    ) -> Tensor:
        return (
            after[
                :,
                0:1,
            ]
            - before[
                :,
                0:1,
            ]
        ) * 10.0


def write_single_band(
    path: Path,
    value: int,
    height: int,
    width: int,
    transform: Affine,
    crs: CRS,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    array = np.full(
        (
            height,
            width,
        ),
        value,
        dtype=np.uint16,
    )

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint16",
        crs=crs,
        transform=transform,
    ) as dataset:
        dataset.write(
            array,
            1,
        )


def build_pair(
    root: Path,
    before_value: int = 0,
    after_value: int = 10_000,
    height: int = 300,
    width: int = 500,
) -> tuple[Path, Path]:
    transform = Affine.translation(
        78.0,
        18.0,
    ) * Affine.scale(
        10.0,
        -10.0,
    )

    crs = CRS.from_epsg(
        32644
    )

    before = (
        root
        / "before"
    )

    after = (
        root
        / "after"
    )

    for band in FROZEN_BANDS:
        write_single_band(
            before
            / f"{band}.tif",
            value=before_value,
            height=height,
            width=width,
            transform=transform,
            crs=crs,
        )

        write_single_band(
            after
            / f"{band}.tif",
            value=after_value,
            height=height,
            width=width,
            transform=transform,
            crs=crs,
        )

    return (
        before,
        after,
    )


def test_calculate_starts_exact_grid(
) -> None:
    assert calculate_starts(
        500,
        256,
        256,
    ) == (
        0,
        256,
    )

    assert calculate_starts(
        512,
        256,
        256,
    ) == (
        0,
        256,
    )

    with pytest.raises(
        ValueError,
        match="stride equal to patch size",
    ):
        calculate_starts(
            500,
            256,
            128,
        )


def test_read_aligned_pair_normalizes_reflectance(
    tmp_path: Path,
) -> None:
    before, after = build_pair(
        tmp_path
    )

    pair = read_aligned_pair(
        before,
        after,
    )

    assert pair.before.shape == (
        4,
        300,
        500,
    )

    assert pair.after.shape == (
        4,
        300,
        500,
    )

    assert float(
        pair.before.min()
    ) == pytest.approx(
        0.0
    )

    assert float(
        pair.after.max()
    ) == pytest.approx(
        1.0
    )

    assert pair.metadata.height == 300
    assert pair.metadata.width == 500
    assert pair.metadata.crs == CRS.from_epsg(
        32644
    )


def test_read_aligned_pair_rejects_transform_mismatch(
    tmp_path: Path,
) -> None:
    before, after = build_pair(
        tmp_path
    )

    mismatched_transform = Affine.translation(
        79.0,
        18.0,
    ) * Affine.scale(
        10.0,
        -10.0,
    )

    write_single_band(
        after
        / "B08.tif",
        value=10_000,
        height=300,
        width=500,
        transform=mismatched_transform,
        crs=CRS.from_epsg(
            32644
        ),
    )

    with pytest.raises(
        RasterPairError,
        match="transform mismatch",
    ):
        read_aligned_pair(
            before,
            after,
        )


def test_tiled_inference_restores_original_shape(
) -> None:
    before = np.zeros(
        (
            4,
            300,
            500,
        ),
        dtype=np.float32,
    )

    after = np.ones(
        (
            4,
            300,
            500,
        ),
        dtype=np.float32,
    )

    probability, mask, patch_count = run_tiled_inference(
        model=DifferenceModel(),
        before=before,
        after=after,
        device=torch.device(
            "cpu"
        ),
        batch_size=3,
        threshold=FROZEN_THRESHOLD,
    )

    assert probability.shape == (
        300,
        500,
    )

    assert mask.shape == (
        300,
        500,
    )

    assert patch_count == 4

    assert np.isfinite(
        probability
    ).all()

    assert float(
        probability.min()
    ) > FROZEN_THRESHOLD

    assert np.all(
        mask
        == 1
    )


def test_tiled_inference_rejects_threshold_change(
) -> None:
    before = np.zeros(
        (
            4,
            256,
            256,
        ),
        dtype=np.float32,
    )

    after = np.ones(
        (
            4,
            256,
            256,
        ),
        dtype=np.float32,
    )

    with pytest.raises(
        RuntimeError,
        match="threshold must remain frozen",
    ):
        run_tiled_inference(
            model=DifferenceModel(),
            before=before,
            after=after,
            device=torch.device(
                "cpu"
            ),
            batch_size=1,
            threshold=0.50,
        )
