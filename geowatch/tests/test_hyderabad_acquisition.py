from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS

from scripts.download_hyderabad_pair import (
    QUALITATIVE_LABEL,
    REQUIRED_BANDS,
    AcquisitionError,
    build_target_grid,
    reproject_asset,
    target_affine,
    validate_selection,
)


def build_selection(
) -> dict[str, object]:
    return {
        "label": QUALITATIVE_LABEL,
        "ground_truth_available": False,
        "metrics_allowed": False,
        "required_assets": list(
            REQUIRED_BANDS
        ),
        "grid": {
            "target_epsg": 32644,
            "width": 512,
            "height": 512,
            "pixel_size": 10.0,
            "left": 200000.0,
            "bottom": 1994880.0,
            "right": 205120.0,
            "top": 2000000.0,
        },
        "selected": {
            "before": {
                "item_id": "before",
                "datetime": "2020-03-29T00:00:00+00:00",
                "cloud_cover": 1.0,
                "mgrs_tile": "43QHV",
            },
            "after": {
                "item_id": "after",
                "datetime": "2025-03-28T00:00:00+00:00",
                "cloud_cover": 1.0,
                "mgrs_tile": "43QHV",
            },
        },
        "access": {
            "official_test_artifacts_modified": False,
        },
    }


def write_source_raster(
    path: Path,
    value: int,
) -> None:
    values = np.full(
        (
            512,
            512,
        ),
        value,
        dtype=np.uint16,
    )

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=512,
        width=512,
        count=1,
        dtype="uint16",
        crs=CRS.from_epsg(
            32644
        ),
        transform=Affine(
            10.0,
            0.0,
            200000.0,
            0.0,
            -10.0,
            2000000.0,
        ),
        nodata=0,
    ) as destination:
        destination.write(
            values,
            1,
        )


def test_selection_contract_is_accepted(
) -> None:
    selection = build_selection()

    validate_selection(
        selection
    )


def test_selection_rejects_metric_reporting(
) -> None:
    selection = build_selection()

    selection[
        "metrics_allowed"
    ] = True

    with pytest.raises(
        AcquisitionError,
        match="metric reporting",
    ):
        validate_selection(
            selection
        )


def test_target_grid_is_exact(
) -> None:
    grid = build_target_grid(
        build_selection()
    )

    assert grid.epsg == 32644
    assert grid.width == 512
    assert grid.height == 512
    assert grid.pixel_size == 10.0

    assert target_affine(
        grid
    ) == Affine(
        10.0,
        0.0,
        200000.0,
        0.0,
        -10.0,
        2000000.0,
    )


def test_local_raster_reprojection_preserves_grid(
    tmp_path: Path,
) -> None:
    source_path = (
        tmp_path
        / "source.tif"
    )

    write_source_raster(
        source_path,
        value=5000,
    )

    grid = build_target_grid(
        build_selection()
    )

    values, source = reproject_asset(
        str(
            source_path
        ),
        grid,
    )

    assert values.shape == (
        512,
        512,
    )

    assert values.dtype == np.uint16

    assert int(
        values.min()
    ) == 5000

    assert int(
        values.max()
    ) == 5000

    assert source.crs == "EPSG:32644"
