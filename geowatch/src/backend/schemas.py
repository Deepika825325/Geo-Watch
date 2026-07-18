from __future__ import annotations

import math
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from src.inference.predictor import (
    FROZEN_BANDS,
    FROZEN_CHECKPOINT_EPOCH,
    FROZEN_CHECKPOINT_SHA256,
    FROZEN_PATCH_SIZE,
    FROZEN_STRIDE,
    FROZEN_THRESHOLD,
)


PROBABILITY_FLOAT32_TOLERANCE = (
    1.1920928955078125e-07
)


class InferenceStatus(StrEnum):
    accepted = "accepted"
    running = "running"
    completed = "completed"
    failed = "failed"


class ArtifactRole(StrEnum):
    probability_raster = "probability_raster"
    binary_mask = "binary_mask"
    change_geojson = "change_geojson"
    qualitative_visualization = "qualitative_visualization"


class GeoWatchSchema(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        use_enum_values=True,
    )


class VectorizationOptions(GeoWatchSchema):
    minimum_area_m2: float = Field(
        default=0.0,
        ge=0.0,
    )

    simplify_tolerance_m: float = Field(
        default=0.0,
        ge=0.0,
    )

    connectivity: Literal[
        4,
        8,
    ] = 8

    destination_crs: Literal[
        "EPSG:4326"
    ] = "EPSG:4326"


class InferenceRequest(GeoWatchSchema):
    request_id: UUID = Field(
        default_factory=uuid4
    )

    before_directory: Path
    after_directory: Path

    aoi_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
    )

    qualitative: bool = False
    persist: bool = True

    vectorization: VectorizationOptions = Field(
        default_factory=VectorizationOptions
    )

    @model_validator(
        mode="after"
    )
    def validate_directories(
        self,
    ) -> Self:
        before = self.before_directory.expanduser().resolve()
        after = self.after_directory.expanduser().resolve()

        if before == after:
            raise ValueError(
                "Before and after directories must be different."
            )

        return self


class FrozenProtocolResponse(GeoWatchSchema):
    checkpoint_epoch: Literal[
        24
    ] = FROZEN_CHECKPOINT_EPOCH

    checkpoint_sha256: Literal[
        "61e53ba86bc108d6ccbbb636c8da88c967e07da41471504878c74df6a903ea94"
    ] = FROZEN_CHECKPOINT_SHA256

    threshold: Literal[
        0.76
    ] = FROZEN_THRESHOLD

    bands: tuple[
        Literal["B02"],
        Literal["B03"],
        Literal["B04"],
        Literal["B08"],
    ] = FROZEN_BANDS

    patch_size: Literal[
        256
    ] = FROZEN_PATCH_SIZE

    stride: Literal[
        256
    ] = FROZEN_STRIDE


class RasterSummary(GeoWatchSchema):
    height: int = Field(
        gt=0
    )

    width: int = Field(
        gt=0
    )

    crs: str = Field(
        min_length=1
    )

    transform: tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ]

    patch_count: int = Field(
        gt=0
    )


class GeoJSONGeometry(GeoWatchSchema):
    type: Literal[
        "Polygon",
        "MultiPolygon",
    ]

    coordinates: list[Any]

    @field_validator(
        "coordinates"
    )
    @classmethod
    def validate_coordinates(
        cls,
        value: list[Any],
    ) -> list[Any]:
        if not value:
            raise ValueError(
                "GeoJSON coordinates cannot be empty."
            )

        return value


class ChangeFeatureProperties(GeoWatchSchema):
    change_id: str = Field(
        pattern=r"^change-[0-9]{6}$"
    )

    area_m2: float = Field(
        gt=0.0
    )

    perimeter_m: float = Field(
        gt=0.0
    )

    pixel_count: int = Field(
        gt=0
    )

    mean_probability: float = Field(
        ge=0.0,
        le=1.0,
    )

    maximum_probability: float = Field(
        ge=0.0,
        le=1.0,
    )

    qualitative: bool

    @model_validator(
        mode="after"
    )
    def validate_probabilities(
        self,
    ) -> Self:
        if self.maximum_probability < self.mean_probability:
            if not math.isclose(
                self.maximum_probability,
                self.mean_probability,
                rel_tol=0.0,
                abs_tol=PROBABILITY_FLOAT32_TOLERANCE,
            ):
                raise ValueError(
                    "Maximum probability cannot be below mean probability."
                )

            self.maximum_probability = self.mean_probability

        return self


class GeoJSONFeature(GeoWatchSchema):
    type: Literal[
        "Feature"
    ] = "Feature"

    id: str = Field(
        pattern=r"^change-[0-9]{6}$"
    )

    geometry: GeoJSONGeometry
    properties: ChangeFeatureProperties

    @model_validator(
        mode="after"
    )
    def validate_identifier(
        self,
    ) -> Self:
        if self.id != self.properties.change_id:
            raise ValueError(
                "Feature ID and property change ID must match."
            )

        return self


class GeoJSONMetadata(GeoWatchSchema):
    source_crs: str = Field(
        min_length=1
    )

    destination_crs: Literal[
        "EPSG:4326"
    ] = "EPSG:4326"

    height: int = Field(
        gt=0
    )

    width: int = Field(
        gt=0
    )

    transform: tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ]

    threshold: Literal[
        0.76
    ] = FROZEN_THRESHOLD

    checkpoint_epoch: Literal[
        24
    ] = FROZEN_CHECKPOINT_EPOCH

    checkpoint_sha256: Literal[
        "61e53ba86bc108d6ccbbb636c8da88c967e07da41471504878c74df6a903ea94"
    ] = FROZEN_CHECKPOINT_SHA256

    qualitative: bool
    ground_truth_available: bool

    performance_metrics_reported: Literal[
        False
    ] = False

    feature_count: int = Field(
        ge=0
    )

    total_area_m2: float = Field(
        ge=0.0
    )

    total_pixel_count: int = Field(
        ge=0
    )

    minimum_area_m2: float = Field(
        ge=0.0
    )

    simplify_tolerance_m: float = Field(
        ge=0.0
    )

    connectivity: Literal[
        4,
        8,
    ]


class GeoJSONFeatureCollection(GeoWatchSchema):
    type: Literal[
        "FeatureCollection"
    ] = "FeatureCollection"

    name: Literal[
        "geowatch_predicted_changes"
    ] = "geowatch_predicted_changes"

    metadata: GeoJSONMetadata
    features: list[GeoJSONFeature]

    @model_validator(
        mode="after"
    )
    def validate_collection(
        self,
    ) -> Self:
        if self.metadata.feature_count != len(
            self.features
        ):
            raise ValueError(
                "Feature count does not match the feature list."
            )

        identifiers = {
            feature.id
            for feature in self.features
        }

        if len(
            identifiers
        ) != len(
            self.features
        ):
            raise ValueError(
                "GeoJSON feature identifiers must be unique."
            )

        total_area = sum(
            feature.properties.area_m2
            for feature in self.features
        )

        total_pixels = sum(
            feature.properties.pixel_count
            for feature in self.features
        )

        if not math.isclose(
            total_area,
            self.metadata.total_area_m2,
            rel_tol=1e-9,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "Total area does not match feature properties."
            )

        if total_pixels != self.metadata.total_pixel_count:
            raise ValueError(
                "Total pixel count does not match feature properties."
            )

        return self


class ArtifactResponse(GeoWatchSchema):
    role: ArtifactRole

    uri: str = Field(
        min_length=1
    )

    sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )

    size_bytes: int = Field(
        gt=0
    )

    media_type: str = Field(
        min_length=1
    )

    qualitative: bool


class InferenceResponse(GeoWatchSchema):
    request_id: UUID

    status: Literal[
        "completed"
    ] = "completed"

    qualitative: bool
    ground_truth_available: bool

    performance_metrics_reported: Literal[
        False
    ] = False

    protocol: FrozenProtocolResponse = Field(
        default_factory=FrozenProtocolResponse
    )

    raster: RasterSummary
    changes: GeoJSONFeatureCollection
    artifacts: list[ArtifactResponse]

    persisted: bool
    stored_change_count: int = Field(
        ge=0
    )

    @model_validator(
        mode="after"
    )
    def validate_response(
        self,
    ) -> Self:
        metadata = self.changes.metadata

        if metadata.qualitative != self.qualitative:
            raise ValueError(
                "Response and GeoJSON qualitative flags must match."
            )

        if metadata.ground_truth_available != self.ground_truth_available:
            raise ValueError(
                "Ground-truth flags must match."
            )

        if any(
            artifact.qualitative
            != self.qualitative
            for artifact in self.artifacts
        ):
            raise ValueError(
                "Artifact qualitative flags must match the response."
            )

        expected_count = (
            metadata.feature_count
            if self.persisted
            else 0
        )

        if self.stored_change_count != expected_count:
            raise ValueError(
                "Stored change count is inconsistent with persistence."
            )

        return self


class StoredChangeResponse(GeoWatchSchema):
    id: UUID
    request_id: UUID

    change_id: str = Field(
        pattern=r"^change-[0-9]{6}$"
    )

    geometry: GeoJSONGeometry

    area_m2: float = Field(
        gt=0.0
    )

    perimeter_m: float = Field(
        gt=0.0
    )

    pixel_count: int = Field(
        gt=0
    )

    mean_probability: float = Field(
        ge=0.0,
        le=1.0,
    )

    maximum_probability: float = Field(
        ge=0.0,
        le=1.0,
    )

    qualitative: bool
    created_at: datetime


class HealthResponse(GeoWatchSchema):
    status: Literal[
        "ok"
    ] = "ok"

    service: Literal[
        "geowatch-inference"
    ] = "geowatch-inference"

    model_loaded: bool

    model_backend: Literal[
        "cuda",
        "onnx_cpu",
    ] | None = None

    database_connected: bool

    protocol: FrozenProtocolResponse = Field(
        default_factory=FrozenProtocolResponse
    )


class ErrorResponse(GeoWatchSchema):
    status: Literal[
        "error"
    ] = "error"

    error_type: str = Field(
        min_length=1
    )

    message: str = Field(
        min_length=1
    )

    request_id: UUID | None = None
