from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from src.backend.schemas import (
    ArtifactResponse,
    FrozenProtocolResponse,
    GeoJSONFeatureCollection,
    HealthResponse,
    InferenceRequest,
    InferenceResponse,
    RasterSummary,
    VectorizationOptions,
)


CHECKPOINT_SHA256 = (
    "61e53ba86bc108d6ccbbb636c8da88c967e07da41471504878c74df6a903ea94"
)


def build_collection(
) -> dict[str, object]:
    return {
        "type": "FeatureCollection",
        "name": "geowatch_predicted_changes",
        "metadata": {
            "source_crs": "EPSG:32644",
            "destination_crs": "EPSG:4326",
            "height": 10,
            "width": 10,
            "transform": (
                10.0,
                0.0,
                200000.0,
                0.0,
                -10.0,
                2000000.0,
                0.0,
                0.0,
                1.0,
            ),
            "threshold": 0.76,
            "checkpoint_epoch": 24,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "qualitative": True,
            "ground_truth_available": False,
            "performance_metrics_reported": False,
            "feature_count": 1,
            "total_area_m2": 100.0,
            "total_pixel_count": 1,
            "minimum_area_m2": 0.0,
            "simplify_tolerance_m": 0.0,
            "connectivity": 8,
        },
        "features": [
            {
                "type": "Feature",
                "id": "change-000001",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [
                                78.0,
                                17.0,
                            ],
                            [
                                78.001,
                                17.0,
                            ],
                            [
                                78.001,
                                17.001,
                            ],
                            [
                                78.0,
                                17.001,
                            ],
                            [
                                78.0,
                                17.0,
                            ],
                        ]
                    ],
                },
                "properties": {
                    "change_id": "change-000001",
                    "area_m2": 100.0,
                    "perimeter_m": 40.0,
                    "pixel_count": 1,
                    "mean_probability": 0.9,
                    "maximum_probability": 0.9,
                    "qualitative": True,
                },
            }
        ],
    }


def build_raster(
) -> RasterSummary:
    return RasterSummary(
        height=10,
        width=10,
        crs="EPSG:32644",
        transform=(
            10.0,
            0.0,
            200000.0,
            0.0,
            -10.0,
            2000000.0,
            0.0,
            0.0,
            1.0,
        ),
        patch_count=1,
    )


def build_artifact(
) -> ArtifactResponse:
    return ArtifactResponse(
        role="change_geojson",
        uri="reports/changes.geojson",
        sha256="a"
        * 64,
        size_bytes=100,
        media_type="application/geo+json",
        qualitative=True,
    )


def test_inference_request_defaults(
) -> None:
    request = InferenceRequest(
        before_directory="data/before",
        after_directory="data/after",
        qualitative=True,
    )

    assert request.persist is True
    assert request.vectorization.connectivity == 8
    assert request.vectorization.destination_crs == "EPSG:4326"


def test_inference_request_rejects_same_directory(
) -> None:
    with pytest.raises(
        ValidationError,
        match="must be different",
    ):
        InferenceRequest(
            before_directory="data/pair",
            after_directory="data/pair",
        )


def test_vectorization_options_reject_negative_area(
) -> None:
    with pytest.raises(
        ValidationError,
    ):
        VectorizationOptions(
            minimum_area_m2=-1.0
        )


def test_frozen_protocol_rejects_threshold_change(
) -> None:
    with pytest.raises(
        ValidationError,
    ):
        FrozenProtocolResponse(
            threshold=0.5
        )


def test_feature_collection_validates_totals(
) -> None:
    collection = GeoJSONFeatureCollection.model_validate(
        build_collection()
    )

    assert collection.metadata.feature_count == 1
    assert len(
        collection.features
    ) == 1


def test_feature_collection_rejects_count_mismatch(
) -> None:
    payload = build_collection()

    metadata = payload[
        "metadata"
    ]

    assert isinstance(
        metadata,
        dict,
    )

    metadata[
        "feature_count"
    ] = 2

    with pytest.raises(
        ValidationError,
        match="Feature count",
    ):
        GeoJSONFeatureCollection.model_validate(
            payload
        )


def test_completed_response_accepts_qualitative_result(
) -> None:
    response = InferenceResponse(
        request_id=uuid4(),
        qualitative=True,
        ground_truth_available=False,
        raster=build_raster(),
        changes=GeoJSONFeatureCollection.model_validate(
            build_collection()
        ),
        artifacts=[
            build_artifact(),
        ],
        persisted=True,
        stored_change_count=1,
    )

    assert response.status == "completed"
    assert response.protocol.threshold == 0.76
    assert response.performance_metrics_reported is False


def test_response_rejects_inconsistent_persistence(
) -> None:
    with pytest.raises(
        ValidationError,
        match="Stored change count",
    ):
        InferenceResponse(
            request_id=uuid4(),
            qualitative=True,
            ground_truth_available=False,
            raster=build_raster(),
            changes=GeoJSONFeatureCollection.model_validate(
                build_collection()
            ),
            artifacts=[
                build_artifact(),
            ],
            persisted=False,
            stored_change_count=1,
        )


def test_health_response_contains_frozen_protocol(
) -> None:
    response = HealthResponse(
        model_loaded=True,
        database_connected=False,
    )

    serialized = response.model_dump(
        mode="json"
    )

    assert serialized[
        "status"
    ] == "ok"

    assert serialized[
        "protocol"
    ][
        "checkpoint_epoch"
    ] == 24

    assert serialized[
        "protocol"
    ][
        "threshold"
    ] == 0.76


def test_unknown_fields_are_rejected(
) -> None:
    with pytest.raises(
        ValidationError,
        match="Extra inputs",
    ):
        HealthResponse(
            model_loaded=True,
            database_connected=True,
            unexpected=True,
        )


def test_float32_probability_boundary_is_normalized(
) -> None:
    payload = build_collection()

    features = payload[
        "features"
    ]

    assert isinstance(
        features,
        list,
    )

    feature = features[
        0
    ]

    assert isinstance(
        feature,
        dict,
    )

    properties = feature[
        "properties"
    ]

    assert isinstance(
        properties,
        dict,
    )

    properties[
        "mean_probability"
    ] = 0.9000000357627869

    properties[
        "maximum_probability"
    ] = 0.8999999761581421

    collection = GeoJSONFeatureCollection.model_validate(
        payload
    )

    normalized = collection.features[
        0
    ].properties

    assert normalized.maximum_probability == (
        normalized.mean_probability
    )
