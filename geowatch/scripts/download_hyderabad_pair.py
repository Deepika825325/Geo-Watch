from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pystac
import pystac_client
import rasterio
from affine import Affine
from numpy.typing import NDArray
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import reproject


STAC_URL = (
    "https://planetarycomputer.microsoft.com/"
    "api/stac/v1"
)

SAS_SIGN_URL = (
    "https://planetarycomputer.microsoft.com/"
    "api/sas/v1/sign"
)

COLLECTION_ID = "sentinel-2-l2a"

QUALITATIVE_LABEL = (
    "qualitative_only_no_ground_truth_metrics"
)

REQUIRED_BANDS = (
    "B02",
    "B03",
    "B04",
    "B08",
)


class AcquisitionError(RuntimeError):
    pass


@dataclass(frozen=True)
class TargetGrid:
    epsg: int
    width: int
    height: int
    pixel_size: float
    left: float
    bottom: float
    right: float
    top: float
    transform: tuple[float, ...]


@dataclass(frozen=True)
class SourceRaster:
    crs: str
    width: int
    height: int
    dtype: str
    transform: tuple[float, ...]
    nodata: float | int | None


@dataclass(frozen=True)
class OutputBand:
    period: str
    band: str
    path: str
    sha256: str
    size_bytes: int
    minimum: int
    maximum: int
    mean: float
    nonzero_pixels: int
    source: SourceRaster


def require_mapping(
    value: Any,
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(
        value,
        Mapping,
    ):
        raise TypeError(
            f"{name} must be a mapping."
        )

    return value


def calculate_sha256(
    path: Path,
) -> str:
    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as source:
        while True:
            block = source.read(
                1024
                * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def normalize_transform(
    transform: Affine,
) -> tuple[float, ...]:
    return tuple(
        round(
            float(value),
            12,
        )
        for value in transform
    )


def load_selection(
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(
            "Scene-selection JSON must contain an object."
        )

    return payload


def validate_selection(
    payload: Mapping[str, Any],
) -> None:
    if payload.get(
        "label"
    ) != QUALITATIVE_LABEL:
        raise AcquisitionError(
            "Hyderabad selection is not labelled qualitative."
        )

    if payload.get(
        "ground_truth_available"
    ) is not False:
        raise AcquisitionError(
            "Hyderabad ground-truth status must remain false."
        )

    if payload.get(
        "metrics_allowed"
    ) is not False:
        raise AcquisitionError(
            "Hyderabad metric reporting must remain disabled."
        )

    required_assets = tuple(
        str(
            value
        )
        for value in payload.get(
            "required_assets",
            (),
        )
    )

    if required_assets != REQUIRED_BANDS:
        raise AcquisitionError(
            "Selected asset order does not match the frozen bands."
        )

    selected = require_mapping(
        payload.get(
            "selected"
        ),
        "selected",
    )

    before = require_mapping(
        selected.get(
            "before"
        ),
        "selected.before",
    )

    after = require_mapping(
        selected.get(
            "after"
        ),
        "selected.after",
    )

    if before.get(
        "mgrs_tile"
    ) != after.get(
        "mgrs_tile"
    ):
        raise AcquisitionError(
            "Before and after scenes must use the same MGRS tile."
        )

    access = require_mapping(
        payload.get(
            "access"
        ),
        "access",
    )

    if access.get(
        "official_test_artifacts_modified"
    ) is not False:
        raise AcquisitionError(
            "Official test artifacts must remain unchanged."
        )


def build_target_grid(
    payload: Mapping[str, Any],
) -> TargetGrid:
    grid = require_mapping(
        payload.get(
            "grid"
        ),
        "grid",
    )

    epsg = int(
        grid[
            "target_epsg"
        ]
    )

    width = int(
        grid[
            "width"
        ]
    )

    height = int(
        grid[
            "height"
        ]
    )

    pixel_size = float(
        grid[
            "pixel_size"
        ]
    )

    left = float(
        grid[
            "left"
        ]
    )

    bottom = float(
        grid[
            "bottom"
        ]
    )

    right = float(
        grid[
            "right"
        ]
    )

    top = float(
        grid[
            "top"
        ]
    )

    if epsg != 32644:
        raise AcquisitionError(
            "Hyderabad target CRS must remain EPSG:32644."
        )

    if width != 512 or height != 512:
        raise AcquisitionError(
            "Hyderabad grid must remain 512 by 512."
        )

    if pixel_size != 10.0:
        raise AcquisitionError(
            "Hyderabad pixel size must remain 10 metres."
        )

    expected_right = (
        left
        + width
        * pixel_size
    )

    expected_bottom = (
        top
        - height
        * pixel_size
    )

    if not np.isclose(
        right,
        expected_right,
    ):
        raise AcquisitionError(
            "Grid right boundary is inconsistent."
        )

    if not np.isclose(
        bottom,
        expected_bottom,
    ):
        raise AcquisitionError(
            "Grid bottom boundary is inconsistent."
        )

    transform = Affine(
        pixel_size,
        0.0,
        left,
        0.0,
        -pixel_size,
        top,
    )

    return TargetGrid(
        epsg=epsg,
        width=width,
        height=height,
        pixel_size=pixel_size,
        left=left,
        bottom=bottom,
        right=right,
        top=top,
        transform=normalize_transform(
            transform
        ),
    )


def target_affine(
    grid: TargetGrid,
) -> Affine:
    return Affine(
        *grid.transform[
            :6
        ]
    )


def fetch_item(
    catalog: pystac_client.Client,
    item_id: str,
) -> pystac.Item:
    search = catalog.search(
        collections=[
            COLLECTION_ID,
        ],
        ids=[
            item_id,
        ],
        max_items=2,
    )

    items = tuple(
        search.items()
    )

    if len(
        items
    ) != 1:
        raise AcquisitionError(
            f"Expected one STAC item for {item_id}; "
            f"found {len(items)}."
        )

    return items[
        0
    ]


def sign_href(
    href: str,
) -> str:
    if not href:
        raise AcquisitionError(
            "Unsigned asset URL cannot be empty."
        )

    query = urlencode(
        {
            "href": href,
        }
    )

    request = Request(
        f"{SAS_SIGN_URL}?{query}",
        headers={
            "User-Agent": "GeoWatch-Week7/1.0",
        },
    )

    with urlopen(
        request,
        timeout=30,
    ) as response:
        payload = json.loads(
            response.read().decode(
                "utf-8"
            )
        )

    if not isinstance(
        payload,
        dict,
    ):
        raise AcquisitionError(
            "Planetary Computer signing response is invalid."
        )

    signed_href = payload.get(
        "href"
    )

    if not isinstance(
        signed_href,
        str,
    ):
        raise AcquisitionError(
            "Planetary Computer signing response has no href."
        )

    if not signed_href:
        raise AcquisitionError(
            "Planetary Computer returned an empty signed URL."
        )

    return signed_href


def signed_asset_href(
    item: pystac.Item,
    band: str,
) -> str:
    if band not in item.assets:
        raise AcquisitionError(
            f"Scene {item.id} has no {band} asset."
        )

    href = item.assets[
        band
    ].href

    if not href:
        raise AcquisitionError(
            f"Asset URL is missing for {item.id} {band}."
        )

    return sign_href(
        str(
            href
        )
    )

def reproject_asset(
    href: str,
    grid: TargetGrid,
) -> tuple[
    NDArray[np.uint16],
    SourceRaster,
]:
    destination = np.zeros(
        (
            grid.height,
            grid.width,
        ),
        dtype=np.float32,
    )

    destination_crs = CRS.from_epsg(
        grid.epsg
    )

    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_HTTP_MULTIRANGE="YES",
        GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
    ):
        with rasterio.open(
            href
        ) as source:
            if source.count != 1:
                raise AcquisitionError(
                    "Sentinel-2 band assets must contain one band."
                )

            if source.crs is None:
                raise AcquisitionError(
                    "Sentinel-2 source asset has no CRS."
                )

            source_record = SourceRaster(
                crs=str(
                    source.crs
                ),
                width=int(
                    source.width
                ),
                height=int(
                    source.height
                ),
                dtype=str(
                    source.dtypes[
                        0
                    ]
                ),
                transform=normalize_transform(
                    source.transform
                ),
                nodata=source.nodata,
            )

            reproject(
                source=rasterio.band(
                    source,
                    1,
                ),
                destination=destination,
                src_transform=source.transform,
                src_crs=source.crs,
                src_nodata=source.nodata,
                dst_transform=target_affine(
                    grid
                ),
                dst_crs=destination_crs,
                dst_nodata=0.0,
                resampling=Resampling.bilinear,
                num_threads=2,
                init_dest_nodata=True,
            )

    if not np.isfinite(
        destination
    ).all():
        raise AcquisitionError(
            "Aligned raster contains non-finite values."
        )

    aligned = np.clip(
        np.rint(
            destination
        ),
        0,
        np.iinfo(
            np.uint16
        ).max,
    ).astype(
        np.uint16
    )

    if int(
        np.count_nonzero(
            aligned
        )
    ) == 0:
        raise AcquisitionError(
            "Aligned raster contains no nonzero pixels."
        )

    return (
        aligned,
        source_record,
    )


def write_band(
    path: Path,
    values: NDArray[np.uint16],
    grid: TargetGrid,
    period: str,
    band: str,
    item_id: str,
    scene_datetime: str,
    cloud_cover: float,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():
        raise FileExistsError(
            path
        )

    profile: dict[str, Any] = {
        "driver": "GTiff",
        "height": grid.height,
        "width": grid.width,
        "count": 1,
        "dtype": "uint16",
        "crs": CRS.from_epsg(
            grid.epsg
        ),
        "transform": target_affine(
            grid
        ),
        "nodata": 0,
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    with rasterio.open(
        path,
        "w",
        **profile,
    ) as destination:
        destination.write(
            values,
            1,
        )

        destination.update_tags(
            qualitative="true",
            ground_truth_available="false",
            metrics_allowed="false",
            period=period,
            band=band,
            source_item_id=item_id,
            source_datetime=scene_datetime,
            source_cloud_cover=str(
                cloud_cover
            ),
            target_epsg=str(
                grid.epsg
            ),
            target_pixel_size=str(
                grid.pixel_size
            ),
        )


def atomic_write_json(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():
        raise FileExistsError(
            path
        )

    temporary_path = path.with_suffix(
        path.suffix
        + ".tmp"
    )

    if temporary_path.exists():
        raise FileExistsError(
            temporary_path
        )

    temporary_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(
        path
    )


def build_argument_parser(
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--selection",
        type=Path,
        default=Path(
            "reports/week7/hyderabad_qualitative/"
            "scene_selection.json"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "data/qualitative/hyderabad"
        ),
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "reports/week7/hyderabad_qualitative/"
            "aligned_pair_manifest.json"
        ),
    )

    return parser


def main(
) -> int:
    arguments = build_argument_parser().parse_args()

    selection = load_selection(
        arguments.selection
    )

    validate_selection(
        selection
    )

    grid = build_target_grid(
        selection
    )

    selected = require_mapping(
        selection[
            "selected"
        ],
        "selected",
    )

    selection_sha256 = calculate_sha256(
        arguments.selection
    )

    target_paths = tuple(
        arguments.output_root
        / period
        / f"{band}.tif"
        for period in (
            "before",
            "after",
        )
        for band in REQUIRED_BANDS
    )

    existing = tuple(
        path
        for path in target_paths
        if path.exists()
    )

    if existing:
        raise FileExistsError(
            "Aligned Hyderabad output already exists: "
            + ", ".join(
                str(path)
                for path in existing
            )
        )

    if arguments.manifest.exists():
        raise FileExistsError(
            arguments.manifest
        )

    arguments.output_root.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    catalog = pystac_client.Client.open(
        STAC_URL
    )

    records: list[
        OutputBand
    ] = []

    selected_items: dict[
        str,
        pystac.Item
    ] = {}

    for period in (
        "before",
        "after",
    ):
        period_record = require_mapping(
            selected[
                period
            ],
            f"selected.{period}",
        )

        item_id = str(
            period_record[
                "item_id"
            ]
        )

        selected_items[
            period
        ] = fetch_item(
            catalog,
            item_id,
        )

    with tempfile.TemporaryDirectory(
        prefix="geowatch_hyderabad_",
        dir=arguments.output_root.parent,
    ) as temporary_directory:
        staging_root = Path(
            temporary_directory
        )

        staged_paths: list[
            tuple[
                Path,
                Path,
            ]
        ] = []

        for period in (
            "before",
            "after",
        ):
            period_record = require_mapping(
                selected[
                    period
                ],
                f"selected.{period}",
            )

            item = selected_items[
                period
            ]

            item_id = str(
                period_record[
                    "item_id"
                ]
            )

            scene_datetime = str(
                period_record[
                    "datetime"
                ]
            )

            cloud_cover = float(
                period_record[
                    "cloud_cover"
                ]
            )

            for band in REQUIRED_BANDS:
                href = signed_asset_href(
                    item,
                    band,
                )

                values, source_record = reproject_asset(
                    href,
                    grid,
                )

                staging_path = (
                    staging_root
                    / period
                    / f"{band}.tif"
                )

                final_path = (
                    arguments.output_root
                    / period
                    / f"{band}.tif"
                )

                write_band(
                    path=staging_path,
                    values=values,
                    grid=grid,
                    period=period,
                    band=band,
                    item_id=item_id,
                    scene_datetime=scene_datetime,
                    cloud_cover=cloud_cover,
                )

                records.append(
                    OutputBand(
                        period=period,
                        band=band,
                        path=str(
                            final_path
                        ),
                        sha256=calculate_sha256(
                            staging_path
                        ),
                        size_bytes=staging_path.stat().st_size,
                        minimum=int(
                            values.min()
                        ),
                        maximum=int(
                            values.max()
                        ),
                        mean=float(
                            values.mean()
                        ),
                        nonzero_pixels=int(
                            np.count_nonzero(
                                values
                            )
                        ),
                        source=source_record,
                    )
                )

                staged_paths.append(
                    (
                        staging_path,
                        final_path,
                    )
                )

                print(
                    "Aligned:",
                    period,
                    band,
                    values.shape,
                    int(
                        values.min()
                    ),
                    int(
                        values.max()
                    ),
                )

        for staging_path, final_path in staged_paths:
            final_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            staging_path.replace(
                final_path
            )

    manifest: dict[str, Any] = {
        "label": QUALITATIVE_LABEL,
        "ground_truth_available": False,
        "metrics_allowed": False,
        "collection": COLLECTION_ID,
        "source_selection": str(
            arguments.selection
        ),
        "source_selection_sha256": selection_sha256,
        "selected": selection[
            "selected"
        ],
        "grid": asdict(
            grid
        ),
        "bands": list(
            REQUIRED_BANDS
        ),
        "outputs": [
            asdict(
                record
            )
            for record in records
        ],
        "access": {
            "image_pixels_accessed": True,
            "model_inference_executed": False,
            "f1_reported": False,
            "iou_reported": False,
            "precision_reported": False,
            "recall_reported": False,
            "official_test_artifacts_modified": False,
        },
    }

    atomic_write_json(
        arguments.manifest,
        manifest,
    )

    print("Hyderabad aligned-pair acquisition completed")
    print("  Label:", QUALITATIVE_LABEL)
    print("  Output root:", arguments.output_root)
    print("  Aligned bands:", len(records))
    print("  Width:", grid.width)
    print("  Height:", grid.height)
    print("  EPSG:", grid.epsg)
    print("  Pixel size:", grid.pixel_size)
    print("  Model inference executed:", False)
    print("  Metrics reported:", False)
    print("  Manifest:", arguments.manifest)

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
