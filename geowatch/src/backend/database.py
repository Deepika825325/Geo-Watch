from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from os import environ
from typing import Any
from uuid import UUID, uuid4

from geoalchemy2 import Geometry
from geoalchemy2.elements import WKBElement
from geoalchemy2.shape import from_shape, to_shape
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    func,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.engine import Engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from src.backend.schemas import (
    GeoJSONFeature,
    GeoJSONFeatureCollection,
    GeoJSONGeometry,
    StoredChangeResponse,
)
from src.inference.predictor import (
    FROZEN_CHECKPOINT_EPOCH,
    FROZEN_CHECKPOINT_SHA256,
    FROZEN_THRESHOLD,
)


DEFAULT_DATABASE_URL = (
    "postgresql+psycopg://"
    "geowatch:geowatch@localhost:5432/geowatch"
)


class DatabaseConfigurationError(RuntimeError):
    pass


class DuplicateInferenceRequestError(RuntimeError):
    pass


class StoredGeometryError(RuntimeError):
    pass


class Base(DeclarativeBase):
    pass


class InferenceRequestRecord(Base):
    __tablename__ = "inference_requests"

    request_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(
            as_uuid=True
        ),
        primary_key=True,
    )

    aoi_name: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    qualitative: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
    )

    ground_truth_available: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
    )

    performance_metrics_reported: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    checkpoint_epoch: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    checkpoint_sha256: Mapped[str] = mapped_column(
        String(
            64
        ),
        nullable=False,
    )

    threshold: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    source_crs: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    destination_crs: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    feature_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    total_area_m2: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    total_pixel_count: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(
            timezone=True
        ),
        nullable=False,
        server_default=func.now(),
    )

    changes: Mapped[
        list[ChangeRecord]
    ] = relationship(
        back_populates="request",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        CheckConstraint(
            "checkpoint_epoch = 24",
            name="ck_inference_requests_checkpoint_epoch",
        ),
        CheckConstraint(
            "threshold = 0.76",
            name="ck_inference_requests_threshold",
        ),
        CheckConstraint(
            "destination_crs = 'EPSG:4326'",
            name="ck_inference_requests_destination_crs",
        ),
        CheckConstraint(
            "performance_metrics_reported = FALSE",
            name="ck_inference_requests_metrics",
        ),
        CheckConstraint(
            "feature_count >= 0",
            name="ck_inference_requests_feature_count",
        ),
        CheckConstraint(
            "total_area_m2 >= 0.0",
            name="ck_inference_requests_total_area",
        ),
        CheckConstraint(
            "total_pixel_count >= 0",
            name="ck_inference_requests_total_pixels",
        ),
    )


class ChangeRecord(Base):
    __tablename__ = "changes"

    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(
            as_uuid=True
        ),
        primary_key=True,
        default=uuid4,
    )

    request_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(
            as_uuid=True
        ),
        ForeignKey(
            "inference_requests.request_id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )

    change_id: Mapped[str] = mapped_column(
        String(
            32
        ),
        nullable=False,
    )

    geometry: Mapped[WKBElement] = mapped_column(
        Geometry(
            geometry_type="GEOMETRY",
            srid=4326,
            spatial_index=False,
        ),
        nullable=False,
    )

    area_m2: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    perimeter_m: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    pixel_count: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )

    mean_probability: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    maximum_probability: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    qualitative: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(
            timezone=True
        ),
        nullable=False,
        server_default=func.now(),
    )

    request: Mapped[
        InferenceRequestRecord
    ] = relationship(
        back_populates="changes"
    )

    __table_args__ = (
        UniqueConstraint(
            "request_id",
            "change_id",
            name="uq_changes_request_change",
        ),
        CheckConstraint(
            "area_m2 > 0.0",
            name="ck_changes_area",
        ),
        CheckConstraint(
            "perimeter_m > 0.0",
            name="ck_changes_perimeter",
        ),
        CheckConstraint(
            "pixel_count > 0",
            name="ck_changes_pixels",
        ),
        CheckConstraint(
            (
                "mean_probability >= 0.0 "
                "AND mean_probability <= 1.0"
            ),
            name="ck_changes_mean_probability",
        ),
        CheckConstraint(
            (
                "maximum_probability >= 0.0 "
                "AND maximum_probability <= 1.0"
            ),
            name="ck_changes_maximum_probability",
        ),
        CheckConstraint(
            "maximum_probability >= mean_probability",
            name="ck_changes_probability_order",
        ),
    )


@dataclass(frozen=True)
class DatabaseConfig:
    url: str = DEFAULT_DATABASE_URL
    pool_pre_ping: bool = True
    pool_size: int = 5
    max_overflow: int = 10

    @classmethod
    def from_environment(
        cls,
    ) -> DatabaseConfig:
        return cls(
            url=environ.get(
                "GEOWATCH_DATABASE_URL",
                DEFAULT_DATABASE_URL,
            )
        )


def normalize_database_url(
    url: str,
) -> str:
    if url.startswith(
        "postgresql+psycopg://"
    ):
        return url

    if url.startswith(
        "postgresql://"
    ):
        return url.replace(
            "postgresql://",
            "postgresql+psycopg://",
            1,
        )

    raise DatabaseConfigurationError(
        "Database URL must use postgresql."
    )


def validate_database_config(
    config: DatabaseConfig,
) -> None:
    normalize_database_url(
        config.url
    )

    if config.pool_size <= 0:
        raise DatabaseConfigurationError(
            "Pool size must be positive."
        )

    if config.max_overflow < 0:
        raise DatabaseConfigurationError(
            "Maximum overflow must not be negative."
        )


def create_database_engine(
    config: DatabaseConfig,
) -> Engine:
    validate_database_config(
        config
    )

    database_url = normalize_database_url(
        config.url
    )

    return create_engine(
        database_url,
        pool_pre_ping=config.pool_pre_ping,
        pool_size=config.pool_size,
        max_overflow=config.max_overflow,
        future=True,
    )


def validate_feature_geometry(
    feature: GeoJSONFeature,
) -> BaseGeometry:
    payload = feature.geometry.model_dump(
        mode="json"
    )

    geometry = shape(
        payload
    )

    if geometry.is_empty:
        raise StoredGeometryError(
            f"Empty geometry: {feature.id}"
        )

    if not geometry.is_valid:
        raise StoredGeometryError(
            f"Invalid geometry: {feature.id}"
        )

    if geometry.geom_type not in {
        "Polygon",
        "MultiPolygon",
    }:
        raise StoredGeometryError(
            f"Unsupported geometry type: {geometry.geom_type}"
        )

    return geometry


def build_request_record(
    request_id: UUID,
    aoi_name: str | None,
    collection: GeoJSONFeatureCollection,
) -> InferenceRequestRecord:
    metadata = collection.metadata

    if metadata.checkpoint_epoch != FROZEN_CHECKPOINT_EPOCH:
        raise ValueError(
            "Frozen checkpoint epoch mismatch."
        )

    if metadata.checkpoint_sha256 != FROZEN_CHECKPOINT_SHA256:
        raise ValueError(
            "Frozen checkpoint SHA-256 mismatch."
        )

    if metadata.threshold != FROZEN_THRESHOLD:
        raise ValueError(
            "Frozen threshold mismatch."
        )

    return InferenceRequestRecord(
        request_id=request_id,
        aoi_name=aoi_name,
        qualitative=metadata.qualitative,
        ground_truth_available=metadata.ground_truth_available,
        performance_metrics_reported=(
            metadata.performance_metrics_reported
        ),
        checkpoint_epoch=metadata.checkpoint_epoch,
        checkpoint_sha256=metadata.checkpoint_sha256,
        threshold=metadata.threshold,
        source_crs=metadata.source_crs,
        destination_crs=metadata.destination_crs,
        feature_count=metadata.feature_count,
        total_area_m2=metadata.total_area_m2,
        total_pixel_count=metadata.total_pixel_count,
    )


def build_change_records(
    request_id: UUID,
    collection: GeoJSONFeatureCollection,
) -> list[ChangeRecord]:
    records: list[
        ChangeRecord
    ] = []

    for feature in collection.features:
        geometry = validate_feature_geometry(
            feature
        )

        properties = feature.properties

        records.append(
            ChangeRecord(
                request_id=request_id,
                change_id=properties.change_id,
                geometry=from_shape(
                    geometry,
                    srid=4326,
                    extended=True,
                ),
                area_m2=properties.area_m2,
                perimeter_m=properties.perimeter_m,
                pixel_count=properties.pixel_count,
                mean_probability=properties.mean_probability,
                maximum_probability=(
                    properties.maximum_probability
                ),
                qualitative=properties.qualitative,
            )
        )

    if len(
        records
    ) != collection.metadata.feature_count:
        raise ValueError(
            "Generated record count does not match metadata."
        )

    return records


def change_record_to_response(
    record: ChangeRecord,
) -> StoredChangeResponse:
    geometry = to_shape(
        record.geometry
    )

    if geometry.is_empty or not geometry.is_valid:
        raise StoredGeometryError(
            f"Stored geometry is invalid: {record.change_id}"
        )

    geometry_mapping = mapping(
        geometry
    )

    return StoredChangeResponse(
        id=record.id,
        request_id=record.request_id,
        change_id=record.change_id,
        geometry=GeoJSONGeometry.model_validate(
            geometry_mapping
        ),
        area_m2=record.area_m2,
        perimeter_m=record.perimeter_m,
        pixel_count=record.pixel_count,
        mean_probability=record.mean_probability,
        maximum_probability=record.maximum_probability,
        qualitative=record.qualitative,
        created_at=record.created_at,
    )


class GeoWatchDatabase:
    def __init__(
        self,
        config: DatabaseConfig | None = None,
        engine: Engine | None = None,
    ) -> None:
        selected_config = (
            config
            if config is not None
            else DatabaseConfig.from_environment()
        )

        self._config = selected_config

        self._engine = (
            engine
            if engine is not None
            else create_database_engine(
                selected_config
            )
        )

        self._session_factory = sessionmaker(
            bind=self._engine,
            class_=Session,
            expire_on_commit=False,
            autoflush=False,
            future=True,
        )

    @property
    def engine(
        self,
    ) -> Engine:
        return self._engine

    @property
    def config(
        self,
    ) -> DatabaseConfig:
        return self._config

    @contextmanager
    def session(
        self,
    ) -> Iterator[Session]:
        database_session = self._session_factory()

        try:
            yield database_session
            database_session.commit()
        except Exception:
            database_session.rollback()
            raise
        finally:
            database_session.close()

    def ping(
        self,
    ) -> bool:
        with self.session() as database_session:
            value = database_session.execute(
                text(
                    "SELECT 1"
                )
            ).scalar_one()

        return int(
            value
        ) == 1

    def persist_feature_collection(
        self,
        request_id: UUID,
        aoi_name: str | None,
        collection: GeoJSONFeatureCollection,
    ) -> int:
        request_record = build_request_record(
            request_id=request_id,
            aoi_name=aoi_name,
            collection=collection,
        )

        change_records = build_change_records(
            request_id=request_id,
            collection=collection,
        )

        with self.session() as database_session:
            existing = database_session.get(
                InferenceRequestRecord,
                request_id,
            )

            if existing is not None:
                raise DuplicateInferenceRequestError(
                    f"Inference request already exists: {request_id}"
                )

            request_record.changes.extend(
                change_records
            )

            database_session.add(
                request_record
            )

        return len(
            change_records
        )

    def get_changes(
        self,
        request_id: UUID,
    ) -> list[StoredChangeResponse]:
        statement = (
            select(
                ChangeRecord
            )
            .where(
                ChangeRecord.request_id
                == request_id
            )
            .order_by(
                ChangeRecord.change_id
            )
        )

        with self.session() as database_session:
            records = list(
                database_session.scalars(
                    statement
                ).all()
            )

            responses = [
                change_record_to_response(
                    record
                )
                for record in records
            ]

        return responses

    def dispose(
        self,
    ) -> None:
        self._engine.dispose()
