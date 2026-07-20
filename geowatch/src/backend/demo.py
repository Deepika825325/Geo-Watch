from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from os import environ
from pathlib import Path
from typing import Protocol
from uuid import UUID

from src.backend.database import (
    DuplicateInferenceRequestError,
)
from src.backend.schemas import (
    GeoJSONFeatureCollection,
)


DEMO_REQUEST_ID = UUID(
    "896053f3-e56c-54cd-be88-d3efb4edd8c6"
)

DEMO_AOI_NAME = (
    "Hyderabad qualitative demonstration"
)

DEMO_FILES = {
    "hyderabad_qualitative_changes.geojson": (
        "changes.geojson"
    ),
    "hyderabad_qualitative_probability.tif": (
        "probability.tif"
    ),
    "hyderabad_qualitative_mask.tif": (
        "mask.tif"
    ),
}

TRUE_VALUES = frozenset(
    {
        "1",
        "true",
        "yes",
        "on",
    }
)


class DemoBootstrapError(RuntimeError):
    pass


class DemoDatabase(Protocol):
    def get_changes(
        self,
        request_id: UUID,
    ) -> Sequence[object]:
        ...

    def persist_feature_collection(
        self,
        request_id: UUID,
        aoi_name: str | None,
        collection: GeoJSONFeatureCollection,
    ) -> int:
        ...


def demo_seed_enabled() -> bool:
    return (
        environ.get(
            "GEOWATCH_SEED_DEMO",
            "",
        )
        .strip()
        .lower()
        in TRUE_VALUES
    )


def resolve_demo_source(
    source_root: Path | None = None,
) -> Path:
    candidates: list[Path] = []

    if source_root is not None:
        candidates.append(
            source_root
        )

    configured_source = environ.get(
        "GEOWATCH_DEMO_SOURCE"
    )

    if configured_source:
        candidates.append(
            Path(
                configured_source
            )
        )

    candidates.extend(
        (
            Path(
                "/app/demo/hyderabad"
            ),
            Path(
                "demo/hyderabad"
            ),
            Path(
                "reports/week7/"
                "hyderabad_qualitative"
            ),
        )
    )

    for candidate in candidates:
        selected = (
            candidate
            .expanduser()
            .resolve()
        )

        if all(
            (
                selected
                / source_name
            ).is_file()
            for source_name in DEMO_FILES
        ):
            return selected

    checked_locations = ", ".join(
        str(
            candidate
            .expanduser()
            .resolve()
        )
        for candidate in candidates
    )

    raise DemoBootstrapError(
        "Hyderabad demo source was not found. "
        f"Checked: {checked_locations}"
    )


def load_demo_collection(
    source_root: Path,
) -> GeoJSONFeatureCollection:
    geojson_path = (
        source_root
        / "hyderabad_qualitative_changes.geojson"
    )

    try:
        payload = json.loads(
            geojson_path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ) as error:
        raise DemoBootstrapError(
            "Hyderabad demo GeoJSON could not be loaded."
        ) from error

    try:
        collection = (
            GeoJSONFeatureCollection.model_validate(
                payload
            )
        )
    except Exception as error:
        raise DemoBootstrapError(
            "Hyderabad demo GeoJSON is invalid."
        ) from error

    if (
        collection.metadata.feature_count
        != len(
            collection.features
        )
    ):
        raise DemoBootstrapError(
            "Hyderabad demo feature count is inconsistent."
        )

    return collection


def copy_demo_artifacts(
    source_root: Path,
    artifact_root: Path,
) -> Path:
    target_directory = (
        artifact_root
        .expanduser()
        .resolve()
        / str(
            DEMO_REQUEST_ID
        )
    )

    target_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for source_name, target_name in DEMO_FILES.items():
        source_path = (
            source_root
            / source_name
        )

        if (
            not source_path.is_file()
            or source_path.stat().st_size <= 0
        ):
            raise DemoBootstrapError(
                f"Demo artifact is unavailable: {source_path}"
            )

        shutil.copy2(
            source_path,
            target_directory
            / target_name,
        )

    return target_directory


def ensure_demo_data(
    database: DemoDatabase,
    artifact_root: Path,
    source_root: Path | None = None,
) -> Path:
    selected_source = resolve_demo_source(
        source_root
    )

    collection = load_demo_collection(
        selected_source
    )

    expected_count = (
        collection.metadata.feature_count
    )

    existing_changes = database.get_changes(
        DEMO_REQUEST_ID
    )

    if existing_changes:
        if (
            len(
                existing_changes
            )
            != expected_count
        ):
            raise DemoBootstrapError(
                "Stored Hyderabad change count does not "
                "match the bundled GeoJSON."
            )
    else:
        try:
            inserted_count = (
                database.persist_feature_collection(
                    request_id=DEMO_REQUEST_ID,
                    aoi_name=DEMO_AOI_NAME,
                    collection=collection,
                )
            )
        except DuplicateInferenceRequestError:
            existing_changes = database.get_changes(
                DEMO_REQUEST_ID
            )

            if (
                len(
                    existing_changes
                )
                != expected_count
            ):
                raise DemoBootstrapError(
                    "Existing Hyderabad request is incomplete."
                )
        else:
            if inserted_count != expected_count:
                raise DemoBootstrapError(
                    "Hyderabad database seed count is incorrect."
                )

    return copy_demo_artifacts(
        source_root=selected_source,
        artifact_root=artifact_root,
    )
