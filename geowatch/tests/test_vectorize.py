from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS

from src.inference.predictor import (
    FROZEN_CHECKPOINT_EPOCH,
    FROZEN_CHECKPOINT_SHA256,
    FROZEN_THRESHOLD,
)
from src.inference.vectorize import (
    VectorizationConfig,
    VectorizationError,
    vectorize_change_mask,
    vectorize_prediction_rasters,
    write_geojson,
)


TRANSFORM = Affine(
    10.0,
    0.0,
    200000.0,
    0.0,
    -10.0,
    2000000.0,
)

SOURCE_CRS = CRS.from_epsg(
    32644
)


def build_arrays(
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    mask = np.zeros(
        (
            10,
            10,
        ),
        dtype=np.uint8,
    )

    mask[
        2:6,
        1:6,
    ] = 1

    probability = np.full(
        (
            10,
            10,
        ),
        0.1,
        dtype=np.float32,
    )

    probability[
        mask
        == 1
    ] = 0.9

    return (
        mask,
        probability,
    )


def write_prediction_raster(
    path: Path,
    values: np.ndarray,
    dtype: str,
    transform: Affine = TRANSFORM,
) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=int(
            values.shape[
                0
            ]
        ),
        width=int(
            values.shape[
                1
            ]
        ),
        count=1,
        dtype=dtype,
        crs=SOURCE_CRS,
        transform=transform,
    ) as dataset:
        dataset.write(
            values,
            1,
        )

        dataset.update_tags(
            threshold=str(
                FROZEN_THRESHOLD
            ),
            checkpoint_epoch=str(
                FROZEN_CHECKPOINT_EPOCH
            ),
            checkpoint_sha256=FROZEN_CHECKPOINT_SHA256,
            qualitative="true",
        )


def test_vectorize_block_area_and_probability(
) -> None:
    mask, probability = build_arrays()

    collection = vectorize_change_mask(
        mask=mask,
        probability=probability,
        transform=TRANSFORM,
        source_crs=SOURCE_CRS,
        qualitative=True,
    )

    assert collection[
        "type"
    ] == "FeatureCollection"

    assert collection[
        "metadata"
    ][
        "feature_count"
    ] == 1

    feature = collection[
        "features"
    ][
        0
    ]

    assert feature[
        "geometry"
    ][
        "type"
    ] == "Polygon"

    assert feature[
        "properties"
    ][
        "area_m2"
    ] == pytest.approx(
        2000.0
    )

    assert feature[
        "properties"
    ][
        "pixel_count"
    ] == 20

    assert feature[
        "properties"
    ][
        "mean_probability"
    ] == pytest.approx(
        0.9
    )

    assert feature[
        "properties"
    ][
        "maximum_probability"
    ] == pytest.approx(
        0.9
    )

    assert feature[
        "properties"
    ][
        "qualitative"
    ] is True


def test_minimum_area_filter_removes_small_component(
) -> None:
    mask = np.zeros(
        (
            10,
            10,
        ),
        dtype=np.uint8,
    )

    mask[
        0,
        0,
    ] = 1

    mask[
        4:7,
        4:7,
    ] = 1

    probability = np.full(
        (
            10,
            10,
        ),
        0.9,
        dtype=np.float32,
    )

    collection = vectorize_change_mask(
        mask=mask,
        probability=probability,
        transform=TRANSFORM,
        source_crs=SOURCE_CRS,
        config=VectorizationConfig(
            minimum_area_m2=500.0
        ),
    )

    assert collection[
        "metadata"
    ][
        "feature_count"
    ] == 1

    assert collection[
        "features"
    ][
        0
    ][
        "properties"
    ][
        "area_m2"
    ] == pytest.approx(
        900.0
    )


def test_geographic_source_crs_is_rejected(
) -> None:
    mask, probability = build_arrays()

    with pytest.raises(
        VectorizationError,
        match="projected",
    ):
        vectorize_change_mask(
            mask=mask,
            probability=probability,
            transform=TRANSFORM,
            source_crs=CRS.from_epsg(
                4326
            ),
        )


def test_vectorization_rejects_threshold_change(
) -> None:
    mask, probability = build_arrays()

    with pytest.raises(
        VectorizationError,
        match="threshold must remain frozen",
    ):
        vectorize_change_mask(
            mask=mask,
            probability=probability,
            transform=TRANSFORM,
            source_crs=SOURCE_CRS,
            threshold=0.5,
        )


def test_prediction_raster_alignment_is_enforced(
    tmp_path: Path,
) -> None:
    mask, probability = build_arrays()

    mask_path = (
        tmp_path
        / "mask.tif"
    )

    probability_path = (
        tmp_path
        / "probability.tif"
    )

    write_prediction_raster(
        mask_path,
        mask,
        "uint8",
    )

    write_prediction_raster(
        probability_path,
        probability,
        "float32",
        transform=Affine(
            10.0,
            0.0,
            200010.0,
            0.0,
            -10.0,
            2000000.0,
        ),
    )

    with pytest.raises(
        VectorizationError,
        match="not aligned",
    ):
        vectorize_prediction_rasters(
            mask_path,
            probability_path,
        )


def test_prediction_rasters_create_qualitative_geojson(
    tmp_path: Path,
) -> None:
    mask, probability = build_arrays()

    mask_path = (
        tmp_path
        / "mask.tif"
    )

    probability_path = (
        tmp_path
        / "probability.tif"
    )

    output_path = (
        tmp_path
        / "changes.geojson"
    )

    write_prediction_raster(
        mask_path,
        mask,
        "uint8",
    )

    write_prediction_raster(
        probability_path,
        probability,
        "float32",
    )

    collection = vectorize_prediction_rasters(
        mask_path,
        probability_path,
    )

    assert collection[
        "metadata"
    ][
        "qualitative"
    ] is True

    assert collection[
        "metadata"
    ][
        "ground_truth_available"
    ] is False

    assert collection[
        "metadata"
    ][
        "performance_metrics_reported"
    ] is False

    assert collection[
        "metadata"
    ][
        "destination_crs"
    ] == "EPSG:4326"

    write_geojson(
        collection,
        output_path,
    )

    loaded = json.loads(
        output_path.read_text(
            encoding="utf-8"
        )
    )

    assert loaded[
        "type"
    ] == "FeatureCollection"

    assert loaded[
        "metadata"
    ][
        "feature_count"
    ] == 1


def test_uniform_float32_probability_preserves_order(
) -> None:
    mask = np.ones(
        (
            64,
            64,
        ),
        dtype=np.uint8,
    )

    probability = np.full(
        (
            64,
            64,
        ),
        0.9,
        dtype=np.float32,
    )

    collection = vectorize_change_mask(
        mask=mask,
        probability=probability,
        transform=TRANSFORM,
        source_crs=SOURCE_CRS,
    )

    properties = collection[
        "features"
    ][
        0
    ][
        "properties"
    ]

    mean_probability = float(
        properties[
            "mean_probability"
        ]
    )

    maximum_probability = float(
        properties[
            "maximum_probability"
        ]
    )

    assert maximum_probability >= mean_probability

    assert maximum_probability == pytest.approx(
        mean_probability
    )
