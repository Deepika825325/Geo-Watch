from __future__ import annotations

import shutil
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from os import environ
from pathlib import Path
from threading import Lock
from typing import Annotated, Any, cast
from uuid import UUID

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    status,
)
from fastapi.responses import (
    FileResponse,
    Response,
)
from sqlalchemy.exc import SQLAlchemyError

from src.backend.database import (
    DuplicateInferenceRequestError,
    GeoWatchDatabase,
)
from src.backend.previews import (
    PreviewRole,
    RasterPreviewError,
    render_raster_preview,
    resolve_preview_source,
)
from src.backend.schemas import (
    ArtifactResponse,
    ArtifactRole,
    ErrorResponse,
    FrozenProtocolResponse,
    GeoJSONFeatureCollection,
    HealthResponse,
    InferenceRequest,
    InferenceResponse,
    RasterSummary,
    StoredChangeResponse,
)
from src.inference.predictor import (
    FrozenChangePredictor,
    FrozenProtocolError,
    RasterPairError,
    calculate_sha256,
    write_mask_geotiff,
    write_probability_geotiff,
)
from src.inference.vectorize import (
    VectorizationConfig,
    VectorizationError,
    vectorize_prediction_rasters,
    write_geojson,
)


DEFAULT_ARTIFACT_ROOT = Path(
    "artifacts/inference"
)


ARTIFACT_DOWNLOADS: dict[
    ArtifactRole,
    tuple[
        str,
        str,
    ],
] = {
    ArtifactRole.probability_raster: (
        "probability.tif",
        "image/tiff; application=geotiff",
    ),
    ArtifactRole.binary_mask: (
        "mask.tif",
        "image/tiff; application=geotiff",
    ),
    ArtifactRole.change_geojson: (
        "changes.geojson",
        "application/geo+json",
    ),
}


def resolve_artifact_file(
    artifact_root: Path,
    request_id: UUID,
    role: ArtifactRole,
) -> tuple[
    Path,
    str,
]:
    artifact_definition = (
        ARTIFACT_DOWNLOADS.get(
            role
        )
    )

    if artifact_definition is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Artifact role is not available for download.",
        )

    filename, media_type = (
        artifact_definition
    )

    resolved_root = (
        artifact_root
        .expanduser()
        .resolve()
    )

    artifact_path = (
        resolved_root
        / str(
            request_id
        )
        / filename
    ).resolve()

    if not artifact_path.is_relative_to(
        resolved_root
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Artifact was not found.",
        )

    if (
        not artifact_path.is_file()
        or artifact_path.stat().st_size <= 0
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "Artifact was not found for request "
                f"{request_id}."
            ),
        )

    return (
        artifact_path,
        media_type,
    )


def get_predictor(
    request: Request,
) -> FrozenChangePredictor:
    predictor = getattr(
        request.app.state,
        "predictor",
        None,
    )

    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Frozen inference model is not loaded.",
        )

    return cast(
        FrozenChangePredictor,
        predictor,
    )


def get_database(
    request: Request,
) -> GeoWatchDatabase | None:
    database = getattr(
        request.app.state,
        "database",
        None,
    )

    if database is None:
        return None

    return cast(
        GeoWatchDatabase,
        database,
    )


def get_inference_lock(
    request: Request,
) -> Lock:
    inference_lock = getattr(
        request.app.state,
        "inference_lock",
        None,
    )

    if inference_lock is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference lock is unavailable.",
        )

    return cast(
        Lock,
        inference_lock,
    )


def get_artifact_root(
    request: Request,
) -> Path:
    artifact_root = getattr(
        request.app.state,
        "artifact_root",
        None,
    )

    if artifact_root is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Artifact storage is unavailable.",
        )

    return cast(
        Path,
        artifact_root,
    )


def validate_input_directory(
    path: Path,
    role: str,
) -> Path:
    resolved = path.expanduser().resolve()

    if not resolved.exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{role} directory does not exist: {resolved}",
        )

    if not resolved.is_dir():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{role} path is not a directory: {resolved}",
        )

    return resolved


def build_artifact_response(
    role: str,
    path: Path,
    media_type: str,
    qualitative: bool,
) -> ArtifactResponse:
    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    if path.stat().st_size <= 0:
        raise RuntimeError(
            f"Artifact is empty: {path}"
        )

    return ArtifactResponse(
        role=role,
        uri=str(
            path
        ),
        sha256=calculate_sha256(
            path
        ),
        size_bytes=path.stat().st_size,
        media_type=media_type,
        qualitative=qualitative,
    )


def remove_directory(
    path: Path,
) -> None:
    if path.exists():
        shutil.rmtree(
            path
        )


def create_lifespan(
    load_resources: bool,
):
    @asynccontextmanager
    async def lifespan(
        app: FastAPI,
    ) -> AsyncIterator[None]:
        if load_resources:
            device = environ.get(
                "GEOWATCH_DEVICE",
                "auto",
            )

            backend = environ.get(
                "MODEL_BACKEND"
            )

            batch_size = int(
                environ.get(
                    "GEOWATCH_BATCH_SIZE",
                    "8",
                )
            )

            app.state.predictor = FrozenChangePredictor(
                backend=backend,
                device=device,
                batch_size=batch_size,
            )

            app.state.database = GeoWatchDatabase()

        try:
            yield
        finally:
            database = getattr(
                app.state,
                "database",
                None,
            )

            if database is not None:
                database.dispose()

            app.state.predictor = None
            app.state.database = None

    return lifespan


def create_app(
    load_resources: bool = True,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> FastAPI:
    app = FastAPI(
        title="GeoWatch Inference API",
        version="0.7.0",
        lifespan=create_lifespan(
            load_resources
        ),
    )

    app.state.predictor = None
    app.state.database = None
    app.state.inference_lock = Lock()
    app.state.artifact_root = artifact_root.expanduser().resolve()

    @app.get(
        "/health",
        response_model=HealthResponse,
        responses={
            500: {
                "model": ErrorResponse,
            },
        },
    )
    def health(
        request: Request,
    ) -> HealthResponse:
        predictor = getattr(
            request.app.state,
            "predictor",
            None,
        )

        database = getattr(
            request.app.state,
            "database",
            None,
        )

        database_connected = False

        if database is not None:
            try:
                database_connected = bool(
                    database.ping()
                )
            except Exception:
                database_connected = False

        return HealthResponse(
            model_loaded=(
                predictor
                is not None
            ),
            model_backend=(
                getattr(
                    predictor,
                    "backend",
                    None,
                )
                if predictor is not None
                else None
            ),
            database_connected=database_connected,
        )

    @app.post(
        "/v1/inference",
        response_model=InferenceResponse,
        status_code=status.HTTP_201_CREATED,
        responses={
            400: {
                "model": ErrorResponse,
            },
            409: {
                "model": ErrorResponse,
            },
            422: {
                "model": ErrorResponse,
            },
            503: {
                "model": ErrorResponse,
            },
        },
    )
    def run_inference(
        payload: InferenceRequest,
        predictor: Annotated[
            FrozenChangePredictor,
            Depends(
                get_predictor
            ),
        ],
        database: Annotated[
            GeoWatchDatabase | None,
            Depends(
                get_database
            ),
        ],
        inference_lock: Annotated[
            Lock,
            Depends(
                get_inference_lock
            ),
        ],
        selected_artifact_root: Annotated[
            Path,
            Depends(
                get_artifact_root
            ),
        ],
    ) -> InferenceResponse:
        before_directory = validate_input_directory(
            payload.before_directory,
            "Before",
        )

        after_directory = validate_input_directory(
            payload.after_directory,
            "After",
        )

        if payload.persist and database is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="PostGIS persistence was requested but the database is unavailable.",
            )

        selected_artifact_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        final_directory = (
            selected_artifact_root
            / str(
                payload.request_id
            )
        )

        if final_directory.exists():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "Artifacts already exist for request "
                    f"{payload.request_id}."
                ),
            )

        staging_directory = Path(
            tempfile.mkdtemp(
                prefix=(
                    f".{payload.request_id}-"
                ),
                dir=selected_artifact_root,
            )
        )

        final_created = False

        try:
            probability_path = (
                staging_directory
                / "probability.tif"
            )

            mask_path = (
                staging_directory
                / "mask.tif"
            )

            geojson_path = (
                staging_directory
                / "changes.geojson"
            )

            with inference_lock:
                result = predictor.predict_pair(
                    before_directory=before_directory,
                    after_directory=after_directory,
                    qualitative=payload.qualitative,
                )

            write_probability_geotiff(
                result,
                probability_path,
            )

            write_mask_geotiff(
                result,
                mask_path,
            )

            vectorization_config = VectorizationConfig(
                minimum_area_m2=(
                    payload.vectorization.minimum_area_m2
                ),
                simplify_tolerance_m=(
                    payload.vectorization.simplify_tolerance_m
                ),
                connectivity=(
                    payload.vectorization.connectivity
                ),
                destination_crs=(
                    payload.vectorization.destination_crs
                ),
            )

            collection_payload: dict[
                str,
                Any,
            ] = vectorize_prediction_rasters(
                mask_path=mask_path,
                probability_path=probability_path,
                config=vectorization_config,
            )

            collection = GeoJSONFeatureCollection.model_validate(
                collection_payload
            )

            write_geojson(
                collection_payload,
                geojson_path,
            )

            staging_directory.replace(
                final_directory
            )

            final_created = True

            final_probability_path = (
                final_directory
                / probability_path.name
            )

            final_mask_path = (
                final_directory
                / mask_path.name
            )

            final_geojson_path = (
                final_directory
                / geojson_path.name
            )

            stored_change_count = 0

            if payload.persist:
                if database is None:
                    raise RuntimeError(
                        "Database dependency became unavailable."
                    )

                stored_change_count = (
                    database.persist_feature_collection(
                        request_id=payload.request_id,
                        aoi_name=payload.aoi_name,
                        collection=collection,
                    )
                )

            artifacts = [
                build_artifact_response(
                    role="probability_raster",
                    path=final_probability_path,
                    media_type=(
                        "image/tiff; application=geotiff"
                    ),
                    qualitative=payload.qualitative,
                ),
                build_artifact_response(
                    role="binary_mask",
                    path=final_mask_path,
                    media_type=(
                        "image/tiff; application=geotiff"
                    ),
                    qualitative=payload.qualitative,
                ),
                build_artifact_response(
                    role="change_geojson",
                    path=final_geojson_path,
                    media_type="application/geo+json",
                    qualitative=payload.qualitative,
                ),
            ]

            return InferenceResponse(
                request_id=payload.request_id,
                qualitative=payload.qualitative,
                ground_truth_available=False,
                performance_metrics_reported=False,
                protocol=FrozenProtocolResponse(),
                raster=RasterSummary(
                    height=result.metadata.height,
                    width=result.metadata.width,
                    crs=str(
                        result.metadata.crs
                    ),
                    transform=tuple(
                        float(
                            value
                        )
                        for value in result.metadata.transform
                    ),
                    patch_count=result.patch_count,
                ),
                changes=collection,
                artifacts=artifacts,
                persisted=payload.persist,
                stored_change_count=stored_change_count,
            )

        except DuplicateInferenceRequestError as error:
            if final_created:
                remove_directory(
                    final_directory
                )

            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(
                    error
                ),
            ) from error

        except FileNotFoundError as error:
            if final_created:
                remove_directory(
                    final_directory
                )

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(
                    error
                ),
            ) from error

        except (
            FrozenProtocolError,
            RasterPairError,
            VectorizationError,
            ValueError,
        ) as error:
            if final_created:
                remove_directory(
                    final_directory
                )

            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(
                    error
                ),
            ) from error

        except SQLAlchemyError as error:
            if final_created:
                remove_directory(
                    final_directory
                )

            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="PostGIS persistence failed.",
            ) from error

        finally:
            if staging_directory.exists():
                remove_directory(
                    staging_directory
                )

    @app.get(
        "/v1/requests/{request_id}/artifacts/{role}",
        response_class=FileResponse,
        response_model=None,
        responses={
            404: {
                "model": ErrorResponse,
            },
            503: {
                "model": ErrorResponse,
            },
        },
    )
    def get_artifact(
        request_id: UUID,
        role: ArtifactRole,
        selected_artifact_root: Annotated[
            Path,
            Depends(
                get_artifact_root
            ),
        ],
    ) -> FileResponse:
        artifact_path, media_type = (
            resolve_artifact_file(
                artifact_root=selected_artifact_root,
                request_id=request_id,
                role=role,
            )
        )

        return FileResponse(
            path=artifact_path,
            media_type=media_type,
            filename=artifact_path.name,
        )

    @app.get(
        "/v1/requests/{request_id}/previews/{role}",
        response_class=Response,
        response_model=None,
        responses={
            404: {
                "model": ErrorResponse,
            },
            422: {
                "model": ErrorResponse,
            },
            503: {
                "model": ErrorResponse,
            },
        },
    )
    def get_raster_preview(
        request_id: UUID,
        role: PreviewRole,
        selected_artifact_root: Annotated[
            Path,
            Depends(
                get_artifact_root
            ),
        ],
    ) -> Response:
        try:
            source_path = resolve_preview_source(
                artifact_root=selected_artifact_root,
                request_id=request_id,
                role=role,
            )

            preview = render_raster_preview(
                source_path=source_path,
                role=role,
            )
        except FileNotFoundError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    "Preview source was not found for request "
                    f"{request_id}."
                ),
            ) from error
        except RasterPreviewError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(
                    error
                ),
            ) from error

        bounds = ",".join(
            f"{value:.12f}"
            for value in preview.bounds
        )

        return Response(
            content=preview.content,
            media_type="image/png",
            headers={
                "Cache-Control": "no-store",
                "Content-Disposition": (
                    f'inline; filename="{role}.png"'
                ),
                "X-GeoWatch-Bounds": bounds,
                "X-GeoWatch-Width": str(
                    preview.width
                ),
                "X-GeoWatch-Height": str(
                    preview.height
                ),
                "X-GeoWatch-Source-CRS": (
                    preview.source_crs
                ),
            },
        )

    @app.get(
        "/v1/requests/{request_id}/changes",
        response_model=list[
            StoredChangeResponse
        ],
        responses={
            503: {
                "model": ErrorResponse,
            },
        },
    )
    def get_changes(
        request_id: UUID,
        database: Annotated[
            GeoWatchDatabase | None,
            Depends(
                get_database
            ),
        ],
    ) -> list[StoredChangeResponse]:
        if database is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="PostGIS database is unavailable.",
            )

        try:
            return database.get_changes(
                request_id
            )
        except SQLAlchemyError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="PostGIS query failed.",
            ) from error

    return app


app = create_app()
