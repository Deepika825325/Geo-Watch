from __future__ import annotations

import argparse
import json
import math
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pystac
import pystac_client
from pyproj import Transformer
from shapely.geometry import box, shape


STAC_URL = (
    "https://planetarycomputer.microsoft.com/"
    "api/stac/v1"
)

COLLECTION_ID = "sentinel-2-l2a"

REQUIRED_ASSETS = (
    "B02",
    "B03",
    "B04",
    "B08",
)

QUALITATIVE_LABEL = (
    "qualitative_only_no_ground_truth_metrics"
)

DEFAULT_CENTER_LONGITUDE = 78.3303319
DEFAULT_CENTER_LATITUDE = 17.3856111
DEFAULT_TARGET_EPSG = 32644
DEFAULT_PIXEL_SIZE = 10.0
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DEFAULT_BEFORE_WINDOW = "2020-01-01/2020-03-31"
DEFAULT_AFTER_WINDOW = "2025-01-01/2025-03-31"
DEFAULT_MAX_CLOUD_COVER = 20.0


class SceneSelectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class GridDefinition:
    center_longitude: float
    center_latitude: float
    target_epsg: int
    pixel_size: float
    width: int
    height: int
    left: float
    bottom: float
    right: float
    top: float
    wgs84_bbox: tuple[
        float,
        float,
        float,
        float,
    ]


@dataclass(frozen=True)
class SceneRecord:
    item_id: str
    datetime: str
    cloud_cover: float
    mgrs_tile: str
    platform: str | None
    projection_epsg: int | None
    asset_keys: tuple[str, ...]


def atomic_write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():
        raise FileExistsError(
            path
        )

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name,
        suffix=".tmp",
        delete=False,
    ) as temporary:
        json.dump(
            payload,
            temporary,
            indent=2,
            sort_keys=True,
        )

        temporary.write(
            "\n"
        )

        temporary_path = Path(
            temporary.name
        )

    temporary_path.replace(
        path
    )


def build_grid(
    center_longitude: float,
    center_latitude: float,
    target_epsg: int,
    pixel_size: float,
    width: int,
    height: int,
) -> GridDefinition:
    if pixel_size <= 0:
        raise ValueError(
            "Pixel size must be positive."
        )

    if width <= 0 or height <= 0:
        raise ValueError(
            "Grid dimensions must be positive."
        )

    forward = Transformer.from_crs(
        4326,
        target_epsg,
        always_xy=True,
    )

    inverse = Transformer.from_crs(
        target_epsg,
        4326,
        always_xy=True,
    )

    center_x, center_y = forward.transform(
        center_longitude,
        center_latitude,
    )

    half_width = (
        width
        * pixel_size
        / 2.0
    )

    half_height = (
        height
        * pixel_size
        / 2.0
    )

    left = center_x - half_width
    right = center_x + half_width
    bottom = center_y - half_height
    top = center_y + half_height

    corners = (
        inverse.transform(
            left,
            bottom,
        ),
        inverse.transform(
            left,
            top,
        ),
        inverse.transform(
            right,
            bottom,
        ),
        inverse.transform(
            right,
            top,
        ),
    )

    longitudes = tuple(
        float(
            longitude
        )
        for longitude, _ in corners
    )

    latitudes = tuple(
        float(
            latitude
        )
        for _, latitude in corners
    )

    return GridDefinition(
        center_longitude=center_longitude,
        center_latitude=center_latitude,
        target_epsg=target_epsg,
        pixel_size=pixel_size,
        width=width,
        height=height,
        left=float(
            left
        ),
        bottom=float(
            bottom
        ),
        right=float(
            right
        ),
        top=float(
            top
        ),
        wgs84_bbox=(
            min(
                longitudes
            ),
            min(
                latitudes
            ),
            max(
                longitudes
            ),
            max(
                latitudes
            ),
        ),
    )


def scene_cloud_cover(
    item: pystac.Item,
) -> float:
    value = item.properties.get(
        "eo:cloud_cover"
    )

    if value is None:
        return math.inf

    return float(
        value
    )


def scene_mgrs_tile(
    item: pystac.Item,
) -> str:
    value = item.properties.get(
        "s2:mgrs_tile"
    )

    if value is None:
        raise SceneSelectionError(
            f"Scene has no MGRS tile: {item.id}"
        )

    return str(
        value
    )


def scene_datetime(
    item: pystac.Item,
) -> datetime:
    value = item.datetime

    if value is None:
        raise SceneSelectionError(
            f"Scene has no datetime: {item.id}"
        )

    return value


def scene_has_required_assets(
    item: pystac.Item,
) -> bool:
    return all(
        asset in item.assets
        for asset in REQUIRED_ASSETS
    )


def scene_covers_aoi(
    item: pystac.Item,
    wgs84_bbox: tuple[
        float,
        float,
        float,
        float,
    ],
) -> bool:
    if item.geometry is None:
        return False

    item_geometry = shape(
        item.geometry
    )

    aoi_geometry = box(
        *wgs84_bbox
    )

    return bool(
        item_geometry.covers(
            aoi_geometry
        )
    )


def search_scenes(
    catalog: pystac_client.Client,
    grid: GridDefinition,
    datetime_window: str,
    max_cloud_cover: float,
) -> tuple[pystac.Item, ...]:
    search = catalog.search(
        collections=[
            COLLECTION_ID,
        ],
        bbox=list(
            grid.wgs84_bbox
        ),
        datetime=datetime_window,
        query={
            "eo:cloud_cover": {
                "lt": max_cloud_cover,
            },
        },
        max_items=200,
    )

    items = tuple(
        item
        for item in search.items()
        if scene_has_required_assets(
            item
        )
        and scene_covers_aoi(
            item,
            grid.wgs84_bbox,
        )
    )

    return tuple(
        sorted(
            items,
            key=lambda item: (
                scene_mgrs_tile(
                    item
                ),
                scene_datetime(
                    item
                ),
                item.id,
            ),
        )
    )


def day_of_year_difference(
    before: pystac.Item,
    after: pystac.Item,
) -> int:
    before_day = int(
        scene_datetime(
            before
        ).strftime(
            "%j"
        )
    )

    after_day = int(
        scene_datetime(
            after
        ).strftime(
            "%j"
        )
    )

    return abs(
        before_day
        - after_day
    )


def select_scene_pair(
    before_scenes: tuple[pystac.Item, ...],
    after_scenes: tuple[pystac.Item, ...],
) -> tuple[
    pystac.Item,
    pystac.Item,
]:
    candidates = tuple(
        (
            before,
            after,
        )
        for before in before_scenes
        for after in after_scenes
        if scene_mgrs_tile(
            before
        )
        == scene_mgrs_tile(
            after
        )
        and scene_datetime(
            before
        )
        < scene_datetime(
            after
        )
    )

    if not candidates:
        raise SceneSelectionError(
            "No valid same-tile Hyderabad scene pair was found."
        )

    return min(
        candidates,
        key=lambda pair: (
            day_of_year_difference(
                pair[
                    0
                ],
                pair[
                    1
                ],
            ),
            scene_cloud_cover(
                pair[
                    0
                ]
            )
            + scene_cloud_cover(
                pair[
                    1
                ]
            ),
            scene_cloud_cover(
                pair[
                    0
                ]
            ),
            scene_cloud_cover(
                pair[
                    1
                ]
            ),
            scene_datetime(
                pair[
                    0
                ],
            ),
            scene_datetime(
                pair[
                    1
                ],
            ),
            pair[
                0
            ].id,
            pair[
                1
            ].id,
        ),
    )


def build_scene_record(
    item: pystac.Item,
) -> SceneRecord:
    projection_epsg = item.properties.get(
        "proj:epsg"
    )

    platform = item.properties.get(
        "platform"
    )

    return SceneRecord(
        item_id=item.id,
        datetime=scene_datetime(
            item
        ).isoformat(),
        cloud_cover=scene_cloud_cover(
            item
        ),
        mgrs_tile=scene_mgrs_tile(
            item
        ),
        platform=(
            str(
                platform
            )
            if platform is not None
            else None
        ),
        projection_epsg=(
            int(
                projection_epsg
            )
            if projection_epsg is not None
            else None
        ),
        asset_keys=tuple(
            sorted(
                item.assets
            )
        ),
    )


def build_argument_parser(
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--center-longitude",
        type=float,
        default=DEFAULT_CENTER_LONGITUDE,
    )

    parser.add_argument(
        "--center-latitude",
        type=float,
        default=DEFAULT_CENTER_LATITUDE,
    )

    parser.add_argument(
        "--target-epsg",
        type=int,
        default=DEFAULT_TARGET_EPSG,
    )

    parser.add_argument(
        "--pixel-size",
        type=float,
        default=DEFAULT_PIXEL_SIZE,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
    )

    parser.add_argument(
        "--before-window",
        default=DEFAULT_BEFORE_WINDOW,
    )

    parser.add_argument(
        "--after-window",
        default=DEFAULT_AFTER_WINDOW,
    )

    parser.add_argument(
        "--max-cloud-cover",
        type=float,
        default=DEFAULT_MAX_CLOUD_COVER,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/week7/hyderabad_qualitative/"
            "scene_selection.json"
        ),
    )

    return parser


def main(
) -> int:
    arguments = build_argument_parser().parse_args()

    grid = build_grid(
        center_longitude=arguments.center_longitude,
        center_latitude=arguments.center_latitude,
        target_epsg=arguments.target_epsg,
        pixel_size=arguments.pixel_size,
        width=arguments.width,
        height=arguments.height,
    )

    catalog = pystac_client.Client.open(
        STAC_URL
    )

    before_scenes = search_scenes(
        catalog=catalog,
        grid=grid,
        datetime_window=arguments.before_window,
        max_cloud_cover=arguments.max_cloud_cover,
    )

    after_scenes = search_scenes(
        catalog=catalog,
        grid=grid,
        datetime_window=arguments.after_window,
        max_cloud_cover=arguments.max_cloud_cover,
    )

    before, after = select_scene_pair(
        before_scenes,
        after_scenes,
    )

    payload: dict[str, Any] = {
        "label": QUALITATIVE_LABEL,
        "ground_truth_available": False,
        "metrics_allowed": False,
        "collection": COLLECTION_ID,
        "required_assets": list(
            REQUIRED_ASSETS
        ),
        "grid": asdict(
            grid
        ),
        "search": {
            "before_window": arguments.before_window,
            "after_window": arguments.after_window,
            "maximum_cloud_cover": (
                arguments.max_cloud_cover
            ),
            "before_candidate_count": len(
                before_scenes
            ),
            "after_candidate_count": len(
                after_scenes
            ),
        },
        "selected": {
            "before": asdict(
                build_scene_record(
                    before
                )
            ),
            "after": asdict(
                build_scene_record(
                    after
                )
            ),
            "day_of_year_difference": (
                day_of_year_difference(
                    before,
                    after,
                )
            ),
        },
        "access": {
            "image_pixels_accessed": False,
            "model_inference_executed": False,
            "f1_reported": False,
            "iou_reported": False,
            "precision_reported": False,
            "recall_reported": False,
            "official_test_artifacts_modified": False,
        },
    }

    atomic_write_json(
        arguments.output,
        payload,
    )

    print("Hyderabad qualitative scene selection completed")
    print("  Label:", QUALITATIVE_LABEL)
    print("  Before candidates:", len(before_scenes))
    print("  After candidates:", len(after_scenes))
    print("  Before item:", before.id)
    print("  Before date:", scene_datetime(before).isoformat())
    print("  Before cloud:", scene_cloud_cover(before))
    print("  After item:", after.id)
    print("  After date:", scene_datetime(after).isoformat())
    print("  After cloud:", scene_cloud_cover(after))
    print("  MGRS tile:", scene_mgrs_tile(before))
    print(
        "  Day-of-year difference:",
        day_of_year_difference(
            before,
            after,
        ),
    )
    print("  Image pixels accessed:", False)
    print("  Model inference executed:", False)
    print("  Metrics reported:", False)
    print("  Output:", arguments.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
