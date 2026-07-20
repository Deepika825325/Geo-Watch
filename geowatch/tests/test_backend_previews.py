from __future__ import annotations

from io import BytesIO
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.crs import CRS
from rasterio.transform import from_origin

from src.backend.previews import (
    RasterPreviewError,
    render_raster_preview,
    resolve_preview_source,
)


def write_raster(
    path: Path,
    values: np.ndarray,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with rasterio.open(
        path,
        mode="w",
        driver="GTiff",
        height=values.shape[
            0
        ],
        width=values.shape[
            1
        ],
        count=1,
        dtype=values.dtype,
        crs=CRS.from_epsg(
            32644
        ),
        transform=from_origin(
            213774.52715530345,
            1926749.7587847603,
            10.0,
            10.0,
        ),
    ) as dataset:
        dataset.write(
            values,
            1,
        )


def test_probability_preview_is_transparent_png(
    tmp_path: Path,
) -> None:
    source = (
        tmp_path
        / "probability.tif"
    )

    values = np.array(
        [
            [
                0.0,
                0.1,
                0.5,
                1.0,
            ],
            [
                0.2,
                0.4,
                0.8,
                0.9,
            ],
        ],
        dtype=np.float32,
    )

    write_raster(
        source,
        values,
    )

    preview = render_raster_preview(
        source,
        "probability",
    )

    assert preview.width == 4
    assert preview.height == 2
    assert preview.source_crs == "EPSG:32644"

    south, west, north, east = (
        preview.bounds
    )

    assert south < north
    assert west < east

    image = Image.open(
        BytesIO(
            preview.content
        )
    )

    assert image.format == "PNG"
    assert image.mode == "RGBA"
    assert image.size == (
        4,
        2,
    )

    rgba = np.asarray(
        image
    )

    assert rgba[
        0,
        0,
        3,
    ] == 0

    assert rgba[
        0,
        2,
        3,
    ] > rgba[
        0,
        1,
        3,
    ]

    assert rgba[
        0,
        3,
        3,
    ] > rgba[
        0,
        2,
        3,
    ]


def test_mask_preview_only_shows_changed_pixels(
    tmp_path: Path,
) -> None:
    source = (
        tmp_path
        / "mask.tif"
    )

    values = np.array(
        [
            [
                0,
                1,
                0,
            ],
            [
                1,
                1,
                0,
            ],
        ],
        dtype=np.uint8,
    )

    write_raster(
        source,
        values,
    )

    preview = render_raster_preview(
        source,
        "mask",
    )

    image = Image.open(
        BytesIO(
            preview.content
        )
    )

    rgba = np.asarray(
        image
    )

    alpha = rgba[
        :,
        :,
        3,
    ]

    assert int(
        np.count_nonzero(
            alpha
        )
    ) == 3

    assert set(
        np.unique(
            alpha
        ).tolist()
    ) == {
        0,
        190,
    }


def test_preview_source_resolves_known_artifact(
    tmp_path: Path,
) -> None:
    artifact_root = (
        tmp_path
        / "artifacts"
    )

    request_id = uuid4()

    source = (
        artifact_root
        / str(
            request_id
        )
        / "probability.tif"
    )

    source.parent.mkdir(
        parents=True
    )

    source.write_bytes(
        b"preview-source"
    )

    resolved = resolve_preview_source(
        artifact_root,
        request_id,
        "probability",
    )

    assert resolved == source.resolve()


def test_preview_source_rejects_missing_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError
    ):
        resolve_preview_source(
            tmp_path
            / "artifacts",
            uuid4(),
            "mask",
        )


def test_preview_source_rejects_symlink_escape(
    tmp_path: Path,
) -> None:
    artifact_root = (
        tmp_path
        / "artifacts"
    )

    artifact_root.mkdir()

    outside = (
        tmp_path
        / "outside"
    )

    outside.mkdir()

    (
        outside
        / "mask.tif"
    ).write_bytes(
        b"outside"
    )

    request_id = uuid4()

    (
        artifact_root
        / str(
            request_id
        )
    ).symlink_to(
        outside,
        target_is_directory=True,
    )

    with pytest.raises(
        FileNotFoundError
    ):
        resolve_preview_source(
            artifact_root,
            request_id,
            "mask",
        )


def test_preview_rejects_multiband_raster(
    tmp_path: Path,
) -> None:
    source = (
        tmp_path
        / "multiband.tif"
    )

    with rasterio.open(
        source,
        mode="w",
        driver="GTiff",
        height=2,
        width=2,
        count=2,
        dtype=np.uint8,
        crs=CRS.from_epsg(
            32644
        ),
        transform=from_origin(
            213774.0,
            1926749.0,
            10.0,
            10.0,
        ),
    ) as dataset:
        dataset.write(
            np.zeros(
                (
                    2,
                    2,
                ),
                dtype=np.uint8,
            ),
            1,
        )

        dataset.write(
            np.zeros(
                (
                    2,
                    2,
                ),
                dtype=np.uint8,
            ),
            2,
        )

    with pytest.raises(
        RasterPreviewError
    ):
        render_raster_preview(
            source,
            "mask",
        )
