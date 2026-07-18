CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS inference_requests (
    request_id UUID PRIMARY KEY,
    aoi_name TEXT,
    qualitative BOOLEAN NOT NULL,
    ground_truth_available BOOLEAN NOT NULL,
    performance_metrics_reported BOOLEAN NOT NULL DEFAULT FALSE,
    checkpoint_epoch INTEGER NOT NULL,
    checkpoint_sha256 CHAR(64) NOT NULL,
    threshold DOUBLE PRECISION NOT NULL,
    source_crs TEXT NOT NULL,
    destination_crs TEXT NOT NULL,
    feature_count INTEGER NOT NULL,
    total_area_m2 DOUBLE PRECISION NOT NULL,
    total_pixel_count BIGINT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT ck_inference_requests_checkpoint_epoch
        CHECK (checkpoint_epoch = 24),
    CONSTRAINT ck_inference_requests_checkpoint_sha256
        CHECK (
            checkpoint_sha256 =
            '61e53ba86bc108d6ccbbb636c8da88c967e07da41471504878c74df6a903ea94'
        ),
    CONSTRAINT ck_inference_requests_threshold
        CHECK (threshold = 0.76),
    CONSTRAINT ck_inference_requests_destination_crs
        CHECK (destination_crs = 'EPSG:4326'),
    CONSTRAINT ck_inference_requests_metrics
        CHECK (performance_metrics_reported = FALSE),
    CONSTRAINT ck_inference_requests_feature_count
        CHECK (feature_count >= 0),
    CONSTRAINT ck_inference_requests_total_area
        CHECK (total_area_m2 >= 0.0),
    CONSTRAINT ck_inference_requests_total_pixels
        CHECK (total_pixel_count >= 0)
);

CREATE TABLE IF NOT EXISTS changes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id UUID NOT NULL,
    change_id TEXT NOT NULL,
    geometry geometry(GEOMETRY, 4326) NOT NULL,
    area_m2 DOUBLE PRECISION NOT NULL,
    perimeter_m DOUBLE PRECISION NOT NULL,
    pixel_count BIGINT NOT NULL,
    mean_probability DOUBLE PRECISION NOT NULL,
    maximum_probability DOUBLE PRECISION NOT NULL,
    qualitative BOOLEAN NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_changes_request
        FOREIGN KEY (request_id)
        REFERENCES inference_requests(request_id)
        ON DELETE CASCADE,
    CONSTRAINT uq_changes_request_change
        UNIQUE (request_id, change_id),
    CONSTRAINT ck_changes_identifier
        CHECK (change_id ~ '^change-[0-9]{6}$'),
    CONSTRAINT ck_changes_geometry_type
        CHECK (
            ST_GeometryType(geometry) IN (
                'ST_Polygon',
                'ST_MultiPolygon'
            )
        ),
    CONSTRAINT ck_changes_geometry_srid
        CHECK (ST_SRID(geometry) = 4326),
    CONSTRAINT ck_changes_geometry_valid
        CHECK (ST_IsValid(geometry)),
    CONSTRAINT ck_changes_area
        CHECK (area_m2 > 0.0),
    CONSTRAINT ck_changes_perimeter
        CHECK (perimeter_m > 0.0),
    CONSTRAINT ck_changes_pixels
        CHECK (pixel_count > 0),
    CONSTRAINT ck_changes_mean_probability
        CHECK (
            mean_probability >= 0.0
            AND mean_probability <= 1.0
        ),
    CONSTRAINT ck_changes_maximum_probability
        CHECK (
            maximum_probability >= 0.0
            AND maximum_probability <= 1.0
        ),
    CONSTRAINT ck_changes_probability_order
        CHECK (
            maximum_probability >= mean_probability
        )
);

CREATE INDEX IF NOT EXISTS ix_changes_request_id
    ON changes (request_id);

CREATE INDEX IF NOT EXISTS ix_changes_change_id
    ON changes (change_id);

CREATE INDEX IF NOT EXISTS ix_changes_geometry_gist
    ON changes
    USING GIST (geometry);

CREATE INDEX IF NOT EXISTS ix_changes_created_at
    ON changes (created_at);
