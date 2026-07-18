from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.evaluation.qualitative_hyderabad import (
    FROZEN_BANDS,
    HyderabadInputError,
    audit_hyderabad_inputs,
)


TEST_CRS = "EPSG:32644"
TEST_TRANSFORM = from_origin(
    78.0,
    18.0,
    10.0,
    10.0,
)


def write_raster(
    path: Path,
    height: int,
    width: int,
    crs: str = TEST_CRS,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    values = np.zeros(
        (
            1,
            height,
            width,
        ),
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
        transform=TEST_TRANSFORM,
    ) as dataset:
        dataset.write(
            values
        )


def build_input_pair(
    root: Path,
    height: int = 300,
    width: int = 500,
) -> None:
    for timestamp in (
        "before",
        "after",
    ):
        for band in FROZEN_BANDS:
            write_raster(
                root
                / timestamp
                / f"{band}.tif",
                height=height,
                width=width,
            )


def test_hyderabad_audit_builds_exact_grid(
    tmp_path: Path,
) -> None:
    build_input_pair(
        tmp_path
    )

    audit = audit_hyderabad_inputs(
        input_root=tmp_path
    )

    assert audit.height == 300
    assert audit.width == 500

    assert audit.row_starts == (
        0,
        256,
    )

    assert audit.column_starts == (
        0,
        256,
    )

    assert audit.patch_count == 4
    assert audit.bottom_padding == 212
    assert audit.right_padding == 12
    assert audit.original_pixel_count == 150_000
    assert audit.padded_pixel_count == 262_144
    assert audit.dtype == "uint16"
    assert audit.crs == TEST_CRS


def test_hyderabad_audit_rejects_missing_band(
    tmp_path: Path,
) -> None:
    build_input_pair(
        tmp_path
    )

    (
        tmp_path
        / "after"
        / "B08.tif"
    ).unlink()

    with pytest.raises(
        FileNotFoundError,
        match="B08",
    ):
        audit_hyderabad_inputs(
            input_root=tmp_path
        )


def test_hyderabad_audit_rejects_shape_mismatch(
    tmp_path: Path,
) -> None:
    build_input_pair(
        tmp_path
    )

    write_raster(
        tmp_path
        / "after"
        / "B08.tif",
        height=299,
        width=500,
    )

    with pytest.raises(
        HyderabadInputError,
        match="identical dimensions",
    ):
        audit_hyderabad_inputs(
            input_root=tmp_path
        )
