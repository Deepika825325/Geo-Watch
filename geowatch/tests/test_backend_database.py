from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from geoalchemy2 import Geometry
from geoalchemy2.shape import from_shape, to_shape
from shapely.geometry import Polygon

from src.backend.database import (
    ChangeRecord,
    DatabaseConfig,
    DatabaseConfigurationError,
    build_change_records,
    build_request_record,
    change_record_to_response,
    validate_database_config,
)
from src.backend.schemas import GeoJSONFeatureCollection


CHECKPOINT_SHA256 = (
    "61e53ba86bc108d6ccbbb636c8da88c967e07da41471504878c74df6a903ea94"
)


def build_collection(
) -> GeoJSONFeatureCollection:
    return GeoJSONFeatureCollection.model_validate(
        {
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
    )


def test_database_config_accepts_psycopg_url(
) -> None:
    validate_database_config(
        DatabaseConfig()
    )


def test_database_config_rejects_non_postgresql_url(
) -> None:
    with pytest.raises(
        DatabaseConfigurationError,
        match="postgresql",
    ):
        validate_database_config(
            DatabaseConfig(
                url="sqlite:///geowatch.db"
            )
        )


def test_database_config_reads_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = (
        "postgresql+psycopg://"
        "user:password@database:5432/geowatch"
    )

    monkeypatch.setenv(
        "GEOWATCH_DATABASE_URL",
        url,
    )

    assert DatabaseConfig.from_environment().url == url


def test_geometry_column_uses_srid_4326(
) -> None:
    geometry_type = ChangeRecord.__table__.c.geometry.type

    assert isinstance(
        geometry_type,
        Geometry,
    )

    assert geometry_type.srid == 4326
    assert geometry_type.spatial_index is False


def test_build_request_record_preserves_frozen_protocol(
) -> None:
    request_id = uuid4()

    record = build_request_record(
        request_id=request_id,
        aoi_name="Hyderabad qualitative AOI",
        collection=build_collection(),
    )

    assert record.request_id == request_id
    assert record.qualitative is True
    assert record.ground_truth_available is False
    assert record.performance_metrics_reported is False
    assert record.checkpoint_epoch == 24
    assert record.checkpoint_sha256 == CHECKPOINT_SHA256
    assert record.threshold == 0.76
    assert record.feature_count == 1


def test_build_change_records_creates_spatial_record(
) -> None:
    request_id = uuid4()

    records = build_change_records(
        request_id=request_id,
        collection=build_collection(),
    )

    assert len(
        records
    ) == 1

    record = records[
        0
    ]

    geometry = to_shape(
        record.geometry
    )

    assert record.request_id == request_id
    assert record.change_id == "change-000001"
    assert record.area_m2 == 100.0
    assert record.pixel_count == 1
    assert record.qualitative is True
    assert geometry.geom_type == "Polygon"
    assert geometry.is_valid


def test_change_record_converts_to_response(
) -> None:
    polygon = Polygon(
        [
            (
                78.0,
                17.0,
            ),
            (
                78.001,
                17.0,
            ),
            (
                78.001,
                17.001,
            ),
            (
                78.0,
                17.001,
            ),
            (
                78.0,
                17.0,
            ),
        ]
    )

    record = ChangeRecord(
        id=uuid4(),
        request_id=uuid4(),
        change_id="change-000001",
        geometry=from_shape(
            polygon,
            srid=4326,
            extended=True,
        ),
        area_m2=100.0,
        perimeter_m=40.0,
        pixel_count=1,
        mean_probability=0.9,
        maximum_probability=0.9,
        qualitative=True,
        created_at=datetime.now(
            timezone.utc
        ),
    )

    response = change_record_to_response(
        record
    )

    assert response.geometry.type == "Polygon"
    assert response.change_id == "change-000001"
    assert response.qualitative is True


def test_migration_contains_postgis_constraints_and_index(
) -> None:
    path = Path(
        "migrations/001_create_changes_table.sql"
    )

    text = path.read_text(
        encoding="utf-8"
    )

    normalized = " ".join(
        text.split()
    ).lower()

    assert "create extension if not exists postgis" in normalized
    assert "geometry geometry(geometry, 4326)" in normalized
    assert "using gist (geometry)" in normalized
    assert "st_isvalid(geometry)" in normalized
    assert "st_srid(geometry) = 4326" in normalized
    assert "threshold = 0.76" in normalized
    assert CHECKPOINT_SHA256 in text
