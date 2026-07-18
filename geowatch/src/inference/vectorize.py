from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from numpy.typing import NDArray
from pyproj import CRS as PyprojCRS
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.features import geometry_mask, shapes
from shapely import make_valid
from shapely.geometry import (
    GeometryCollection,
    MultiPolygon,
    Polygon,
    mapping,
    shape,
)
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

from src.inference.predictor import (
    FROZEN_CHECKPOINT_EPOCH,
    FROZEN_CHECKPOINT_SHA256,
    FROZEN_THRESHOLD,
)


DESTINATION_CRS = "EPSG:4326"


class VectorizationError(RuntimeError):
    pass


@dataclass(frozen=True)
class VectorizationConfig:
    minimum_area_m2: float = 0.0
    simplify_tolerance_m: float = 0.0
    connectivity: int = 8
    destination_crs: str = DESTINATION_CRS


@dataclass(frozen=True)
class RasterProvenance:
    source_crs: str
    destination_crs: str
    height: int
    width: int
    transform: tuple[float, ...]
    threshold: float
    checkpoint_epoch: int
    checkpoint_sha256: str
    qualitative: bool
    ground_truth_available: bool
    performance_metrics_reported: bool


@dataclass(frozen=True)
class ChangeStatistics:
    change_id: str
    area_m2: float
    perimeter_m: float
    pixel_count: int
    mean_probability: float
    maximum_probability: float
    qualitative: bool


def normalize_transform(
    transform: Affine,
) -> tuple[float, ...]:
    return tuple(
        float(
            value
        )
        for value in transform
    )


def validate_config(
    config: VectorizationConfig,
) -> None:
    if config.minimum_area_m2 < 0.0:
        raise ValueError(
            "Minimum area must not be negative."
        )

    if config.simplify_tolerance_m < 0.0:
        raise ValueError(
            "Simplification tolerance must not be negative."
        )

    if config.connectivity not in {
        4,
        8,
    }:
        raise ValueError(
            "Connectivity must be four or eight."
        )

    destination = PyprojCRS.from_user_input(
        config.destination_crs
    )

    if destination.to_epsg() != 4326:
        raise VectorizationError(
            "GeoJSON destination CRS must be EPSG:4326."
        )


def validate_source_crs(
    source_crs: CRS,
) -> PyprojCRS:
    parsed = PyprojCRS.from_user_input(
        source_crs
    )

    if not parsed.is_projected:
        raise VectorizationError(
            "Source CRS must be projected for area calculation."
        )

    axis_units = {
        str(
            axis.unit_name
        ).lower()
        for axis in parsed.axis_info
        if axis.unit_name is not None
    }

    if axis_units and not axis_units.issubset(
        {
            "metre",
            "meter",
        }
    ):
        raise VectorizationError(
            "Source CRS axes must use metres."
        )

    return parsed


def validate_arrays(
    mask: NDArray[np.uint8],
    probability: NDArray[np.float32],
) -> None:
    if mask.ndim != 2:
        raise VectorizationError(
            "Prediction mask must be two-dimensional."
        )

    if probability.ndim != 2:
        raise VectorizationError(
            "Probability raster must be two-dimensional."
        )

    if mask.shape != probability.shape:
        raise VectorizationError(
            "Mask and probability raster shapes do not match."
        )

    mask_values = {
        int(
            value
        )
        for value in np.unique(
            mask
        )
    }

    if not mask_values.issubset(
        {
            0,
            1,
        }
    ):
        raise VectorizationError(
            "Prediction mask must contain only zero and one."
        )

    if not np.isfinite(
        probability
    ).all():
        raise VectorizationError(
            "Probability raster contains non-finite values."
        )

    if float(
        probability.min()
    ) < 0.0:
        raise VectorizationError(
            "Probability raster contains values below zero."
        )

    if float(
        probability.max()
    ) > 1.0:
        raise VectorizationError(
            "Probability raster contains values above one."
        )


def polygon_parts(
    geometry: BaseGeometry,
) -> tuple[Polygon, ...]:
    valid = make_valid(
        geometry
    )

    if valid.is_empty:
        return ()

    if isinstance(
        valid,
        Polygon,
    ):
        return (
            valid,
        )

    if isinstance(
        valid,
        MultiPolygon,
    ):
        return tuple(
            polygon
            for polygon in valid.geoms
            if not polygon.is_empty
        )

    if isinstance(
        valid,
        GeometryCollection,
    ):
        parts: list[
            Polygon
        ] = []

        for child in valid.geoms:
            parts.extend(
                polygon_parts(
                    child
                )
            )

        return tuple(
            parts
        )

    return ()


def extract_polygons(
    mask: NDArray[np.uint8],
    transform: Affine,
    connectivity: int,
) -> tuple[Polygon, ...]:
    generated: list[
        Polygon
    ] = []

    for geometry_mapping, value in shapes(
        mask,
        mask=(
            mask
            == 1
        ),
        transform=transform,
        connectivity=connectivity,
    ):
        if int(
            value
        ) != 1:
            continue

        geometry = shape(
            geometry_mapping
        )

        generated.extend(
            polygon_parts(
                geometry
            )
        )

    return tuple(
        generated
    )


def prepare_polygon(
    polygon: Polygon,
    config: VectorizationConfig,
) -> tuple[Polygon, ...]:
    candidate: BaseGeometry = polygon

    if config.simplify_tolerance_m > 0.0:
        candidate = candidate.simplify(
            config.simplify_tolerance_m,
            preserve_topology=True,
        )

    parts = polygon_parts(
        candidate
    )

    return tuple(
        part
        for part in parts
        if float(
            part.area
        )
        >= config.minimum_area_m2
        and not part.is_empty
    )


def polygon_probability_statistics(
    polygon: Polygon,
    probability: NDArray[np.float32],
    transform: Affine,
) -> tuple[
    int,
    float,
    float,
]:
    selected = geometry_mask(
        [
            mapping(
                polygon
            ),
        ],
        out_shape=probability.shape,
        transform=transform,
        invert=True,
        all_touched=False,
    )

    values = probability[
        selected
    ].astype(
        np.float64,
        copy=False,
    )

    if values.size == 0:
        raise VectorizationError(
            "Vectorized polygon contains no raster pixels."
        )

    mean_probability = float(
        np.mean(
            values,
            dtype=np.float64,
        )
    )

    maximum_probability = float(
        np.max(
            values
        )
    )

    if mean_probability > maximum_probability:
        if not np.isclose(
            mean_probability,
            maximum_probability,
            rtol=0.0,
            atol=float(
                np.finfo(
                    np.float32
                ).eps
            ),
        ):
            raise VectorizationError(
                "Mean probability exceeds maximum probability."
            )

        mean_probability = maximum_probability

    return (
        int(
            values.size
        ),
        mean_probability,
        maximum_probability,
    )


def sort_polygons(
    polygons: tuple[Polygon, ...],
) -> tuple[Polygon, ...]:
    return tuple(
        sorted(
            polygons,
            key=lambda polygon: (
                -float(
                    polygon.area
                ),
                float(
                    polygon.bounds[
                        0
                    ]
                ),
                float(
                    polygon.bounds[
                        1
                    ]
                ),
                float(
                    polygon.bounds[
                        2
                    ]
                ),
                float(
                    polygon.bounds[
                        3
                    ]
                ),
            ),
        )
    )


def vectorize_change_mask(
    mask: NDArray[np.uint8],
    probability: NDArray[np.float32],
    transform: Affine,
    source_crs: CRS,
    config: VectorizationConfig = VectorizationConfig(),
    threshold: float = FROZEN_THRESHOLD,
    checkpoint_epoch: int = FROZEN_CHECKPOINT_EPOCH,
    checkpoint_sha256: str = FROZEN_CHECKPOINT_SHA256,
    qualitative: bool = False,
    ground_truth_available: bool = False,
) -> dict[str, Any]:
    validate_config(
        config
    )

    validate_arrays(
        mask,
        probability,
    )

    if threshold != FROZEN_THRESHOLD:
        raise VectorizationError(
            "Vectorization threshold must remain frozen at 0.76."
        )

    if checkpoint_epoch != FROZEN_CHECKPOINT_EPOCH:
        raise VectorizationError(
            "Frozen checkpoint epoch mismatch."
        )

    if checkpoint_sha256 != FROZEN_CHECKPOINT_SHA256:
        raise VectorizationError(
            "Frozen checkpoint SHA-256 mismatch."
        )

    source = validate_source_crs(
        source_crs
    )

    destination = PyprojCRS.from_user_input(
        config.destination_crs
    )

    transformer = Transformer.from_crs(
        source,
        destination,
        always_xy=True,
    )

    extracted = extract_polygons(
        mask,
        transform,
        config.connectivity,
    )

    prepared: list[
        Polygon
    ] = []

    for polygon in extracted:
        prepared.extend(
            prepare_polygon(
                polygon,
                config,
            )
        )

    ordered = sort_polygons(
        tuple(
            prepared
        )
    )

    features: list[
        dict[str, Any]
    ] = []

    total_area_m2 = 0.0
    total_pixel_count = 0

    for index, polygon in enumerate(
        ordered,
        start=1,
    ):
        pixel_count, mean_probability, maximum_probability = (
            polygon_probability_statistics(
                polygon,
                probability,
                transform,
            )
        )

        area_m2 = float(
            polygon.area
        )

        perimeter_m = float(
            polygon.length
        )

        total_area_m2 += area_m2
        total_pixel_count += pixel_count

        transformed = shapely_transform(
            transformer.transform,
            polygon,
        )

        transformed_parts = polygon_parts(
            transformed
        )

        if len(
            transformed_parts
        ) != 1:
            raise VectorizationError(
                "Polygon transformation produced an invalid geometry."
            )

        transformed_polygon = transformed_parts[
            0
        ]

        change_id = (
            f"change-{index:06d}"
        )

        statistics = ChangeStatistics(
            change_id=change_id,
            area_m2=area_m2,
            perimeter_m=perimeter_m,
            pixel_count=pixel_count,
            mean_probability=mean_probability,
            maximum_probability=maximum_probability,
            qualitative=qualitative,
        )

        features.append(
            {
                "type": "Feature",
                "id": change_id,
                "geometry": mapping(
                    transformed_polygon
                ),
                "properties": asdict(
                    statistics
                ),
            }
        )

    provenance = RasterProvenance(
        source_crs=source.to_string(),
        destination_crs=destination.to_string(),
        height=int(
            mask.shape[
                0
            ]
        ),
        width=int(
            mask.shape[
                1
            ]
        ),
        transform=normalize_transform(
            transform
        ),
        threshold=threshold,
        checkpoint_epoch=checkpoint_epoch,
        checkpoint_sha256=checkpoint_sha256,
        qualitative=qualitative,
        ground_truth_available=ground_truth_available,
        performance_metrics_reported=False,
    )

    return {
        "type": "FeatureCollection",
        "name": "geowatch_predicted_changes",
        "metadata": {
            **asdict(
                provenance
            ),
            "feature_count": len(
                features
            ),
            "total_area_m2": total_area_m2,
            "total_pixel_count": total_pixel_count,
            "minimum_area_m2": config.minimum_area_m2,
            "simplify_tolerance_m": config.simplify_tolerance_m,
            "connectivity": config.connectivity,
        },
        "features": features,
    }


def read_prediction_rasters(
    mask_path: Path,
    probability_path: Path,
) -> tuple[
    NDArray[np.uint8],
    NDArray[np.float32],
    Affine,
    CRS,
    dict[str, str],
]:
    if not mask_path.is_file():
        raise FileNotFoundError(
            mask_path
        )

    if not probability_path.is_file():
        raise FileNotFoundError(
            probability_path
        )

    with rasterio.open(
        mask_path
    ) as mask_dataset:
        mask = mask_dataset.read(
            1
        ).astype(
            np.uint8,
            copy=False,
        )

        mask_reference = (
            int(
                mask_dataset.height
            ),
            int(
                mask_dataset.width
            ),
            mask_dataset.crs,
            mask_dataset.transform,
        )

        mask_tags = mask_dataset.tags()

    with rasterio.open(
        probability_path
    ) as probability_dataset:
        probability = probability_dataset.read(
            1,
            out_dtype="float32",
        )

        probability_reference = (
            int(
                probability_dataset.height
            ),
            int(
                probability_dataset.width
            ),
            probability_dataset.crs,
            probability_dataset.transform,
        )

        probability_tags = probability_dataset.tags()

    if mask_reference != probability_reference:
        raise VectorizationError(
            "Mask and probability GeoTIFFs are not aligned."
        )

    source_crs = mask_reference[
        2
    ]

    if source_crs is None:
        raise VectorizationError(
            "Prediction rasters have no CRS."
        )

    required_tags = (
        "threshold",
        "checkpoint_epoch",
        "checkpoint_sha256",
        "qualitative",
    )

    for key in required_tags:
        if mask_tags.get(
            key
        ) != probability_tags.get(
            key
        ):
            raise VectorizationError(
                f"Prediction raster tag mismatch: {key}"
            )

    tags = {
        key: str(
            mask_tags[
                key
            ]
        )
        for key in required_tags
    }

    return (
        mask,
        probability,
        mask_reference[
            3
        ],
        source_crs,
        tags,
    )


def vectorize_prediction_rasters(
    mask_path: Path,
    probability_path: Path,
    config: VectorizationConfig = VectorizationConfig(),
) -> dict[str, Any]:
    mask, probability, transform, source_crs, tags = (
        read_prediction_rasters(
            mask_path,
            probability_path,
        )
    )

    threshold = float(
        tags[
            "threshold"
        ]
    )

    checkpoint_epoch = int(
        tags[
            "checkpoint_epoch"
        ]
    )

    checkpoint_sha256 = tags[
        "checkpoint_sha256"
    ]

    qualitative = (
        tags[
            "qualitative"
        ].strip().lower()
        == "true"
    )

    return vectorize_change_mask(
        mask=mask,
        probability=probability,
        transform=transform,
        source_crs=source_crs,
        config=config,
        threshold=threshold,
        checkpoint_epoch=checkpoint_epoch,
        checkpoint_sha256=checkpoint_sha256,
        qualitative=qualitative,
        ground_truth_available=False,
    )


def write_geojson(
    feature_collection: dict[str, Any],
    path: Path,
) -> None:
    if feature_collection.get(
        "type"
    ) != "FeatureCollection":
        raise VectorizationError(
            "Output must be a GeoJSON FeatureCollection."
        )

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
            feature_collection,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(
        path
    )
