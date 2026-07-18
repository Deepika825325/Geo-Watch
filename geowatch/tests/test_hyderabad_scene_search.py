from __future__ import annotations

from datetime import datetime, timezone

import pystac
from shapely.geometry import mapping, box

from scripts.search_hyderabad_scenes import (
    REQUIRED_ASSETS,
    build_grid,
    day_of_year_difference,
    scene_covers_aoi,
    select_scene_pair,
)


def build_item(
    item_id: str,
    timestamp: datetime,
    tile: str,
    cloud_cover: float,
) -> pystac.Item:
    item = pystac.Item(
        id=item_id,
        geometry=mapping(
            box(
                78.0,
                17.0,
                79.0,
                18.0,
            )
        ),
        bbox=[
            78.0,
            17.0,
            79.0,
            18.0,
        ],
        datetime=timestamp,
        properties={
            "eo:cloud_cover": cloud_cover,
            "s2:mgrs_tile": tile,
            "proj:epsg": 32644,
            "platform": "sentinel-2",
        },
    )

    for asset in REQUIRED_ASSETS:
        item.add_asset(
            asset,
            pystac.Asset(
                href=(
                    "https://example.invalid/"
                    f"{item_id}/{asset}.tif"
                )
            ),
        )

    return item


def test_grid_has_requested_dimensions(
) -> None:
    grid = build_grid(
        center_longitude=78.3303319,
        center_latitude=17.3856111,
        target_epsg=32644,
        pixel_size=10.0,
        width=512,
        height=512,
    )

    assert grid.right - grid.left == 5120.0
    assert grid.top - grid.bottom == 5120.0
    assert len(
        grid.wgs84_bbox
    ) == 4


def test_scene_must_cover_complete_aoi(
) -> None:
    item = build_item(
        item_id="scene",
        timestamp=datetime(
            2020,
            2,
            1,
            tzinfo=timezone.utc,
        ),
        tile="44QKE",
        cloud_cover=1.0,
    )

    assert scene_covers_aoi(
        item,
        (
            78.2,
            17.2,
            78.4,
            17.4,
        ),
    )

    assert not scene_covers_aoi(
        item,
        (
            77.9,
            17.2,
            78.4,
            17.4,
        ),
    )


def test_pair_selection_requires_same_tile_and_season(
) -> None:
    before = (
        build_item(
            item_id="before-poor-season",
            timestamp=datetime(
                2020,
                3,
                20,
                tzinfo=timezone.utc,
            ),
            tile="44QKE",
            cloud_cover=0.1,
        ),
        build_item(
            item_id="before-same-season",
            timestamp=datetime(
                2020,
                2,
                10,
                tzinfo=timezone.utc,
            ),
            tile="44QKE",
            cloud_cover=2.0,
        ),
    )

    after = (
        build_item(
            item_id="after",
            timestamp=datetime(
                2025,
                2,
                12,
                tzinfo=timezone.utc,
            ),
            tile="44QKE",
            cloud_cover=3.0,
        ),
    )

    selected_before, selected_after = select_scene_pair(
        before,
        after,
    )

    assert selected_before.id == "before-same-season"
    assert selected_after.id == "after"

    assert day_of_year_difference(
        selected_before,
        selected_after,
    ) <= 2
