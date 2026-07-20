from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal
from uuid import UUID

import numpy as np
import rasterio
from PIL import Image
from rasterio.warp import transform_bounds


PreviewRole = Literal[
    "probability",
    "mask",
]


PREVIEW_SOURCE_FILES: dict[
    PreviewRole,
    str,
] = {
    "probability": "probability.tif",
    "mask": "mask.tif",
}


@dataclass(
    frozen=True,
    slots=True,
)
class RasterPreview:
    content: bytes
    bounds: tuple[
        float,
        float,
        float,
        float,
    ]
    width: int
    height: int
    source_crs: str


class RasterPreviewError(
    RuntimeError
):
    pass


def resolve_preview_source(
    artifact_root: Path,
    request_id: UUID,
    role: PreviewRole,
) -> Path:
    resolved_root = (
        artifact_root
        .expanduser()
        .resolve()
    )

    source_path = (
        resolved_root
        / str(
            request_id
        )
        / PREVIEW_SOURCE_FILES[
            role
        ]
    ).resolve()

    if not source_path.is_relative_to(
        resolved_root
    ):
        raise FileNotFoundError(
            source_path
        )

    if (
        not source_path.is_file()
        or source_path.stat().st_size <= 0
    ):
        raise FileNotFoundError(
            source_path
        )

    return source_path


def build_probability_rgba(
    values: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    probabilities = np.clip(
        values.astype(
            np.float32,
            copy=False,
        ),
        0.0,
        1.0,
    )

    alpha = np.clip(
        (
            probabilities
            - 0.1
        )
        / 0.9
        * 220.0,
        0.0,
        220.0,
    ).astype(
        np.uint8
    )

    alpha = np.where(
        valid,
        alpha,
        0,
    ).astype(
        np.uint8
    )

    red = np.full(
        probabilities.shape,
        181,
        dtype=np.uint8,
    )

    green = np.full(
        probabilities.shape,
        118,
        dtype=np.uint8,
    )

    blue = np.full(
        probabilities.shape,
        59,
        dtype=np.uint8,
    )

    return np.stack(
        (
            red,
            green,
            blue,
            alpha,
        ),
        axis=-1,
    )


def build_mask_rgba(
    values: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    active = (
        values
        > 0
    ) & valid

    red = np.full(
        values.shape,
        181,
        dtype=np.uint8,
    )

    green = np.full(
        values.shape,
        118,
        dtype=np.uint8,
    )

    blue = np.full(
        values.shape,
        59,
        dtype=np.uint8,
    )

    alpha = np.where(
        active,
        190,
        0,
    ).astype(
        np.uint8
    )

    return np.stack(
        (
            red,
            green,
            blue,
            alpha,
        ),
        axis=-1,
    )


def encode_png(
    rgba: np.ndarray,
) -> bytes:
    if (
        rgba.ndim != 3
        or rgba.shape[
            2
        ] != 4
        or rgba.dtype != np.uint8
    ):
        raise RasterPreviewError(
            "Preview array must be uint8 RGBA."
        )

    buffer = BytesIO()

    Image.fromarray(
        rgba
    ).save(
        buffer,
        format="PNG",
        optimize=True,
    )

    content = buffer.getvalue()

    if not content:
        raise RasterPreviewError(
            "Generated preview is empty."
        )

    return content


def render_raster_preview(
    source_path: Path,
    role: PreviewRole,
) -> RasterPreview:
    try:
        with rasterio.open(
            source_path
        ) as dataset:
            if dataset.count != 1:
                raise RasterPreviewError(
                    "Preview source must contain one raster band."
                )

            if dataset.crs is None:
                raise RasterPreviewError(
                    "Preview source has no CRS."
                )

            band = dataset.read(
                1,
                masked=True,
            )

            values = band.filled(
                0
            )

            valid = ~np.ma.getmaskarray(
                band
            )

            if role == "probability":
                rgba = build_probability_rgba(
                    values,
                    valid,
                )
            else:
                rgba = build_mask_rgba(
                    values,
                    valid,
                )

            west, south, east, north = (
                transform_bounds(
                    dataset.crs,
                    "EPSG:4326",
                    *dataset.bounds,
                    densify_pts=21,
                )
            )

            return RasterPreview(
                content=encode_png(
                    rgba
                ),
                bounds=(
                    float(
                        south
                    ),
                    float(
                        west
                    ),
                    float(
                        north
                    ),
                    float(
                        east
                    ),
                ),
                width=dataset.width,
                height=dataset.height,
                source_crs=str(
                    dataset.crs
                ),
            )
    except rasterio.errors.RasterioError as error:
        raise RasterPreviewError(
            f"Unable to read preview source: {source_path}"
        ) from error
