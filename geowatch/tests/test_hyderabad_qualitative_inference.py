from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS

from scripts.run_hyderabad_qualitative import (
    create_visualization,
    stretch_channel,
    validate_prediction,
)
from src.inference.predictor import (
    FROZEN_BANDS,
    FROZEN_CHECKPOINT_EPOCH,
    FROZEN_CHECKPOINT_SHA256,
    FROZEN_PATCH_SIZE,
    FROZEN_STRIDE,
    FROZEN_THRESHOLD,
    PredictionResult,
    RasterMetadata,
    write_mask_geotiff,
)


def build_result(
    qualitative: bool = True,
) -> PredictionResult:
    metadata = RasterMetadata(
        height=32,
        width=32,
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
        nodata=None,
    )

    probability = np.zeros(
        (
            32,
            32,
        ),
        dtype=np.float32,
    )

    probability[
        8:24,
        8:24,
    ] = 0.9

    mask = (
        probability
        >= FROZEN_THRESHOLD
    ).astype(
        np.uint8
    )

    return PredictionResult(
        probability=probability,
        mask=mask,
        metadata=metadata,
        threshold=FROZEN_THRESHOLD,
        checkpoint_epoch=FROZEN_CHECKPOINT_EPOCH,
        checkpoint_sha256=FROZEN_CHECKPOINT_SHA256,
        bands=FROZEN_BANDS,
        patch_size=FROZEN_PATCH_SIZE,
        stride=FROZEN_STRIDE,
        patch_count=1,
        qualitative=qualitative,
    )


def test_stretch_channel_range(
) -> None:
    values = np.arange(
        100,
        dtype=np.float32,
    ).reshape(
        10,
        10,
    )

    stretched = stretch_channel(
        values
    )

    assert stretched.shape == values.shape
    assert stretched.dtype == np.float32
    assert float(
        stretched.min()
    ) >= 0.0
    assert float(
        stretched.max()
    ) <= 1.0


def test_prediction_contract_accepts_frozen_qualitative_result(
) -> None:
    validate_prediction(
        build_result()
    )


def test_mask_zero_is_not_nodata(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "mask.tif"
    )

    write_mask_geotiff(
        build_result(),
        path,
    )

    with rasterio.open(
        path
    ) as dataset:
        assert dataset.nodata is None

        values = dataset.read(
            1
        )

        assert set(
            int(
                value
            )
            for value in np.unique(
                values
            )
        ) == {
            0,
            1,
        }

        assert dataset.tags()[
            "qualitative"
        ] == "true"


def test_visualization_is_created(
    tmp_path: Path,
) -> None:
    before = np.zeros(
        (
            32,
            32,
            3,
        ),
        dtype=np.float32,
    )

    after = np.ones(
        (
            32,
            32,
            3,
        ),
        dtype=np.float32,
    )

    path = (
        tmp_path
        / "visualization.png"
    )

    create_visualization(
        before_rgb=before,
        after_rgb=after,
        result=build_result(),
        path=path,
    )

    assert path.is_file()
    assert path.stat().st_size > 0
