from __future__ import annotations

import hashlib
from io import BytesIO
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import numpy as np
from PIL import Image
from affine import Affine
from fastapi.testclient import TestClient
from rasterio.crs import CRS

from src.backend.main import (
    create_app,
    get_database,
    get_predictor,
)
from src.backend.schemas import (
    GeoJSONFeatureCollection,
    StoredChangeResponse,
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
)


class FakePredictor:
    backend = "onnx_cpu"

    def predict_pair(
        self,
        before_directory: Path,
        after_directory: Path,
        qualitative: bool,
    ) -> PredictionResult:
        probability = np.full(
            (
                32,
                32,
            ),
            0.1,
            dtype=np.float32,
        )

        probability[
            8:16,
            8:16,
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
            metadata=RasterMetadata(
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
            ),
            threshold=FROZEN_THRESHOLD,
            checkpoint_epoch=FROZEN_CHECKPOINT_EPOCH,
            checkpoint_sha256=FROZEN_CHECKPOINT_SHA256,
            bands=FROZEN_BANDS,
            patch_size=FROZEN_PATCH_SIZE,
            stride=FROZEN_STRIDE,
            patch_count=1,
            qualitative=qualitative,
        )


class FakeDatabase:
    def __init__(
        self,
    ) -> None:
        self.collections: dict[
            UUID,
            GeoJSONFeatureCollection,
        ] = {}

    def ping(
        self,
    ) -> bool:
        return True

    def persist_feature_collection(
        self,
        request_id: UUID,
        aoi_name: str | None,
        collection: GeoJSONFeatureCollection,
    ) -> int:
        self.collections[
            request_id
        ] = collection

        return collection.metadata.feature_count

    def get_changes(
        self,
        request_id: UUID,
    ) -> list[StoredChangeResponse]:
        collection = self.collections.get(
            request_id
        )

        if collection is None:
            return []

        responses: list[
            StoredChangeResponse
        ] = []

        for feature in collection.features:
            properties = feature.properties

            responses.append(
                StoredChangeResponse(
                    id=uuid4(),
                    request_id=request_id,
                    change_id=properties.change_id,
                    geometry=feature.geometry,
                    area_m2=properties.area_m2,
                    perimeter_m=properties.perimeter_m,
                    pixel_count=properties.pixel_count,
                    mean_probability=(
                        properties.mean_probability
                    ),
                    maximum_probability=(
                        properties.maximum_probability
                    ),
                    qualitative=properties.qualitative,
                    created_at=datetime.now(
                        timezone.utc
                    ),
                )
            )

        return responses

    def dispose(
        self,
    ) -> None:
        return None


def create_test_client(
    artifact_root: Path,
    database: FakeDatabase | None,
) -> TestClient:
    app = create_app(
        load_resources=False,
        artifact_root=artifact_root,
    )

    predictor = FakePredictor()

    app.dependency_overrides[
        get_predictor
    ] = lambda: predictor

    app.dependency_overrides[
        get_database
    ] = lambda: database

    return TestClient(
        app
    )


def create_input_directories(
    root: Path,
) -> tuple[
    Path,
    Path,
]:
    before = (
        root
        / "before"
    )

    after = (
        root
        / "after"
    )

    before.mkdir(
        parents=True
    )

    after.mkdir(
        parents=True
    )

    return (
        before,
        after,
    )


def test_health_reports_loaded_resources(
    tmp_path: Path,
) -> None:
    database = FakeDatabase()

    app = create_app(
        load_resources=False,
        artifact_root=tmp_path,
    )

    app.state.predictor = FakePredictor()
    app.state.database = database

    with TestClient(
        app
    ) as client:
        response = client.get(
            "/health"
        )

    assert response.status_code == 200

    payload = response.json()

    assert payload[
        "model_loaded"
    ] is True

    assert payload[
        "model_backend"
    ] == "onnx_cpu"

    assert payload[
        "database_connected"
    ] is True

    assert payload[
        "protocol"
    ][
        "threshold"
    ] == 0.76


def test_inference_creates_artifacts_and_persists_changes(
    tmp_path: Path,
) -> None:
    database = FakeDatabase()

    artifact_root = (
        tmp_path
        / "artifacts"
    )

    before, after = create_input_directories(
        tmp_path
        / "input"
    )

    request_id = uuid4()

    with create_test_client(
        artifact_root,
        database,
    ) as client:
        response = client.post(
            "/v1/inference",
            json={
                "request_id": str(
                    request_id
                ),
                "before_directory": str(
                    before
                ),
                "after_directory": str(
                    after
                ),
                "aoi_name": "Test AOI",
                "qualitative": True,
                "persist": True,
            },
        )

        changes_response = client.get(
            f"/v1/requests/{request_id}/changes"
        )

    assert response.status_code == 201

    payload = response.json()

    assert payload[
        "status"
    ] == "completed"

    assert payload[
        "qualitative"
    ] is True

    assert payload[
        "ground_truth_available"
    ] is False

    assert payload[
        "performance_metrics_reported"
    ] is False

    assert payload[
        "persisted"
    ] is True

    assert payload[
        "stored_change_count"
    ] == 1

    assert payload[
        "changes"
    ][
        "metadata"
    ][
        "feature_count"
    ] == 1

    assert len(
        payload[
            "artifacts"
        ]
    ) == 3

    for artifact in payload[
        "artifacts"
    ]:
        path = Path(
            artifact[
                "uri"
            ]
        )

        assert path.is_file()
        assert path.stat().st_size > 0

        actual_sha256 = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()

        assert actual_sha256 == artifact[
            "sha256"
        ]

    assert changes_response.status_code == 200
    assert len(
        changes_response.json()
    ) == 1


def test_inference_without_persistence_does_not_require_database(
    tmp_path: Path,
) -> None:
    before, after = create_input_directories(
        tmp_path
        / "input"
    )

    with create_test_client(
        tmp_path
        / "artifacts",
        None,
    ) as client:
        response = client.post(
            "/v1/inference",
            json={
                "before_directory": str(
                    before
                ),
                "after_directory": str(
                    after
                ),
                "qualitative": False,
                "persist": False,
            },
        )

    assert response.status_code == 201

    payload = response.json()

    assert payload[
        "persisted"
    ] is False

    assert payload[
        "stored_change_count"
    ] == 0


def test_inference_rejects_missing_input_directory(
    tmp_path: Path,
) -> None:
    database = FakeDatabase()

    existing = (
        tmp_path
        / "after"
    )

    existing.mkdir()

    with create_test_client(
        tmp_path
        / "artifacts",
        database,
    ) as client:
        response = client.post(
            "/v1/inference",
            json={
                "before_directory": str(
                    tmp_path
                    / "missing"
                ),
                "after_directory": str(
                    existing
                ),
                "persist": True,
            },
        )

    assert response.status_code == 400
    assert "does not exist" in response.json()[
        "detail"
    ]


def test_persistence_requires_database(
    tmp_path: Path,
) -> None:
    before, after = create_input_directories(
        tmp_path
        / "input"
    )

    with create_test_client(
        tmp_path
        / "artifacts",
        None,
    ) as client:
        response = client.post(
            "/v1/inference",
            json={
                "before_directory": str(
                    before
                ),
                "after_directory": str(
                    after
                ),
                "persist": True,
            },
        )

    assert response.status_code == 503
    assert "database is unavailable" in response.json()[
        "detail"
    ]


def test_artifact_endpoint_serves_known_files(
    tmp_path: Path,
) -> None:
    artifact_root = (
        tmp_path
        / "artifacts"
    )

    request_id = uuid4()

    request_directory = (
        artifact_root
        / str(
            request_id
        )
    )

    request_directory.mkdir(
        parents=True
    )

    expected_artifacts = {
        "probability_raster": (
            "probability.tif",
            b"probability-raster",
            "image/tiff",
        ),
        "binary_mask": (
            "mask.tif",
            b"binary-mask",
            "image/tiff",
        ),
        "change_geojson": (
            "changes.geojson",
            b'{"type":"FeatureCollection","features":[]}',
            "application/geo+json",
        ),
    }

    for (
        filename,
        content,
        _,
    ) in expected_artifacts.values():
        (
            request_directory
            / filename
        ).write_bytes(
            content
        )

    with create_test_client(
        artifact_root,
        None,
    ) as client:
        for (
            role,
            (
                filename,
                expected_content,
                expected_media_type,
            ),
        ) in expected_artifacts.items():
            response = client.get(
                (
                    f"/v1/requests/{request_id}"
                    f"/artifacts/{role}"
                )
            )

            assert response.status_code == 200
            assert response.content == expected_content

            assert response.headers[
                "content-type"
            ].startswith(
                expected_media_type
            )

            assert filename in response.headers[
                "content-disposition"
            ]


def test_artifact_endpoint_returns_not_found(
    tmp_path: Path,
) -> None:
    request_id = uuid4()

    with create_test_client(
        tmp_path
        / "artifacts",
        None,
    ) as client:
        response = client.get(
            (
                f"/v1/requests/{request_id}"
                "/artifacts/probability_raster"
            )
        )

    assert response.status_code == 404

    assert (
        "Artifact was not found"
        in response.json()[
            "detail"
        ]
    )


def test_artifact_endpoint_rejects_unknown_role(
    tmp_path: Path,
) -> None:
    request_id = uuid4()

    with create_test_client(
        tmp_path
        / "artifacts",
        None,
    ) as client:
        response = client.get(
            (
                f"/v1/requests/{request_id}"
                "/artifacts/arbitrary_file"
            )
        )

    assert response.status_code == 422


def test_artifact_endpoint_rejects_symlink_escape(
    tmp_path: Path,
) -> None:
    artifact_root = (
        tmp_path
        / "artifacts"
    )

    artifact_root.mkdir()

    outside_directory = (
        tmp_path
        / "outside"
    )

    outside_directory.mkdir()

    (
        outside_directory
        / "probability.tif"
    ).write_bytes(
        b"outside-artifact"
    )

    request_id = uuid4()

    (
        artifact_root
        / str(
            request_id
        )
    ).symlink_to(
        outside_directory,
        target_is_directory=True,
    )

    with create_test_client(
        artifact_root,
        None,
    ) as client:
        response = client.get(
            (
                f"/v1/requests/{request_id}"
                "/artifacts/probability_raster"
            )
        )

    assert response.status_code == 404


def test_preview_endpoint_serves_generated_pngs(
    tmp_path: Path,
) -> None:
    database = FakeDatabase()

    artifact_root = (
        tmp_path
        / "artifacts"
    )

    before, after = create_input_directories(
        tmp_path
        / "input"
    )

    request_id = uuid4()

    with create_test_client(
        artifact_root,
        database,
    ) as client:
        inference_response = client.post(
            "/v1/inference",
            json={
                "request_id": str(
                    request_id
                ),
                "before_directory": str(
                    before
                ),
                "after_directory": str(
                    after
                ),
                "qualitative": True,
                "persist": True,
            },
        )

        probability_response = client.get(
            (
                f"/v1/requests/{request_id}"
                "/previews/probability"
            )
        )

        mask_response = client.get(
            (
                f"/v1/requests/{request_id}"
                "/previews/mask"
            )
        )

    assert inference_response.status_code == 201

    for response in (
        probability_response,
        mask_response,
    ):
        assert response.status_code == 200

        assert response.headers[
            "content-type"
        ].startswith(
            "image/png"
        )

        assert response.headers[
            "cache-control"
        ] == "no-store"

        assert response.headers[
            "x-geowatch-width"
        ] == "32"

        assert response.headers[
            "x-geowatch-height"
        ] == "32"

        assert response.headers[
            "x-geowatch-source-crs"
        ] == "EPSG:32644"

        bounds = [
            float(
                value
            )
            for value in response.headers[
                "x-geowatch-bounds"
            ].split(
                ","
            )
        ]

        assert len(
            bounds
        ) == 4

        south, west, north, east = (
            bounds
        )

        assert south < north
        assert west < east

        image = Image.open(
            BytesIO(
                response.content
            )
        )

        assert image.format == "PNG"
        assert image.mode == "RGBA"
        assert image.size == (
            32,
            32,
        )

    probability_rgba = np.asarray(
        Image.open(
            BytesIO(
                probability_response.content
            )
        )
    )

    mask_rgba = np.asarray(
        Image.open(
            BytesIO(
                mask_response.content
            )
        )
    )

    assert int(
        np.count_nonzero(
            probability_rgba[
                :,
                :,
                3,
            ]
        )
    ) == 64

    assert int(
        np.count_nonzero(
            mask_rgba[
                :,
                :,
                3,
            ]
        )
    ) == 64


def test_preview_endpoint_returns_not_found(
    tmp_path: Path,
) -> None:
    request_id = uuid4()

    with create_test_client(
        tmp_path
        / "artifacts",
        None,
    ) as client:
        response = client.get(
            (
                f"/v1/requests/{request_id}"
                "/previews/probability"
            )
        )

    assert response.status_code == 404

    assert (
        "Preview source was not found"
        in response.json()[
            "detail"
        ]
    )


def test_preview_endpoint_rejects_unknown_role(
    tmp_path: Path,
) -> None:
    request_id = uuid4()

    with create_test_client(
        tmp_path
        / "artifacts",
        None,
    ) as client:
        response = client.get(
            (
                f"/v1/requests/{request_id}"
                "/previews/arbitrary"
            )
        )

    assert response.status_code == 422


def test_openapi_contains_service_endpoints(
    tmp_path: Path,
) -> None:
    app = create_app(
        load_resources=False,
        artifact_root=tmp_path,
    )

    schema: dict[str, Any] = app.openapi()

    paths = schema[
        "paths"
    ]

    assert "/health" in paths
    assert "/v1/inference" in paths

    assert (
        "/v1/requests/{request_id}/changes"
        in paths
    )

    assert (
        "/v1/requests/{request_id}/artifacts/{role}"
        in paths
    )

    assert (
        "/v1/requests/{request_id}/previews/{role}"
        in paths
    )
