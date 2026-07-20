from pathlib import Path
from uuid import UUID

from src.backend.demo import (
    DEMO_REQUEST_ID,
    demo_seed_enabled,
    ensure_demo_data,
)
from src.backend.schemas import (
    GeoJSONFeatureCollection,
)


SOURCE_ROOT = (
    Path(
        __file__
    ).resolve().parents[
        1
    ]
    / "reports"
    / "week7"
    / "hyderabad_qualitative"
)


class FakeDemoDatabase:
    def __init__(
        self,
    ) -> None:
        self.collection: (
            GeoJSONFeatureCollection
            | None
        ) = None

        self.persist_calls = 0

    def get_changes(
        self,
        request_id: UUID,
    ) -> tuple[object, ...]:
        assert request_id == DEMO_REQUEST_ID

        if self.collection is None:
            return ()

        return tuple(
            object()
            for _ in self.collection.features
        )

    def persist_feature_collection(
        self,
        request_id: UUID,
        aoi_name: str | None,
        collection: GeoJSONFeatureCollection,
    ) -> int:
        assert request_id == DEMO_REQUEST_ID
        assert aoi_name is not None

        self.persist_calls += 1
        self.collection = collection

        return collection.metadata.feature_count


def test_demo_seed_flag(
    monkeypatch,
) -> None:
    monkeypatch.delenv(
        "GEOWATCH_SEED_DEMO",
        raising=False,
    )

    assert demo_seed_enabled() is False

    monkeypatch.setenv(
        "GEOWATCH_SEED_DEMO",
        "true",
    )

    assert demo_seed_enabled() is True


def test_seeds_database_and_artifacts(
    tmp_path: Path,
) -> None:
    database = FakeDemoDatabase()

    target = ensure_demo_data(
        database=database,
        artifact_root=tmp_path,
        source_root=SOURCE_ROOT,
    )

    assert database.persist_calls == 1
    assert database.collection is not None

    assert (
        database.collection.metadata.feature_count
        == 106
    )

    assert {
        path.name
        for path in target.iterdir()
        if path.is_file()
    } == {
        "changes.geojson",
        "probability.tif",
        "mask.tif",
    }

    for artifact_path in target.iterdir():
        assert artifact_path.stat().st_size > 0


def test_demo_seed_is_idempotent(
    tmp_path: Path,
) -> None:
    database = FakeDemoDatabase()

    ensure_demo_data(
        database=database,
        artifact_root=tmp_path,
        source_root=SOURCE_ROOT,
    )

    ensure_demo_data(
        database=database,
        artifact_root=tmp_path,
        source_root=SOURCE_ROOT,
    )

    assert database.persist_calls == 1
