# GeoWatch Week 7 Backend and PostGIS Runbook

## Frozen inference protocol

- Checkpoint epoch: 24
- Checkpoint SHA-256: `61e53ba86bc108d6ccbbb636c8da88c967e07da41471504878c74df6a903ea94`
- Decision threshold: `0.76`
- Bands: `B02`, `B03`, `B04`, `B08`
- Patch size: `256`
- Stride: `256`

The backend must not modify the checkpoint or threshold.

## Environment

Activate the project virtual environment and export the runtime variables:

    cd /path/to/geowatch

    while [[ "${CONDA_SHLVL:-0}" -gt 0 ]]; do
      conda deactivate
    done

    source .venv/bin/activate

    export GEOWATCH_POSTGRES_PASSWORD='<local-password>'
    export GEOWATCH_DATABASE_URL='postgresql+psycopg://geowatch:<local-password>@127.0.0.1:55432/geowatch'
    export GEOWATCH_DEVICE='cuda'
    export GEOWATCH_BATCH_SIZE='4'

Do not commit the real password.

## Start PostGIS

    docker compose -f docker-compose.week7.yml up -d postgis
    docker compose -f docker-compose.week7.yml ps

## Apply the database migration

    docker compose \
      -f docker-compose.week7.yml \
      exec \
      -T \
      postgis \
      psql \
      -v ON_ERROR_STOP=1 \
      -U geowatch \
      -d geowatch \
      < migrations/001_create_changes_table.sql

The migration is idempotent.

## Start the API

    ./.venv/bin/python \
      -m uvicorn \
      src.backend.main:app \
      --host 127.0.0.1 \
      --port 8007

Swagger UI is available at:

    http://127.0.0.1:8007/docs

## Health check

    curl --fail http://127.0.0.1:8007/health

A healthy service reports:

- `model_loaded: true`
- `database_connected: true`
- checkpoint epoch `24`
- threshold `0.76`

## Hyderabad inference request

Use real absolute paths rather than placeholder paths:

    PROJECT_ROOT="$(pwd)"

    cat > /tmp/geowatch_inference_request.json <<JSON
    {
      "before_directory": "${PROJECT_ROOT}/data/qualitative/hyderabad/before",
      "after_directory": "${PROJECT_ROOT}/data/qualitative/hyderabad/after",
        "aoi_name": "Hyderabad Kokapet qualitative demonstration",
        "qualitative": true,
        "persist": true,
        "vectorization": {
          "minimum_area_m2": 0.0,
          "simplify_tolerance_m": 0.0,
          "connectivity": 8,
          "destination_crs": "EPSG:4326"
        }
    }
    JSON

    curl \
      --request POST \
      --header 'Content-Type: application/json' \
      --data @/tmp/geowatch_inference_request.json \
      http://127.0.0.1:8007/v1/inference

## Retrieve stored changes

Replace the value after the endpoint with an actual UUID. Do not type angle brackets because Bash interprets them as redirection operators.

    REQUEST_ID='actual-request-uuid'

    curl \
      "http://127.0.0.1:8007/v1/requests/${REQUEST_ID}/changes"

## Generated artifacts

Artifacts are written under:

    artifacts/inference/<request-uuid>/
    ├── probability.tif
    ├── mask.tif
    └── changes.geojson

## Stop the API

Stop the foreground Uvicorn process with `Ctrl+C`.

## Stop PostGIS

The Compose environment variable must be available even when stopping the service:

    export GEOWATCH_POSTGRES_PASSWORD='<local-password>'
    docker compose -f docker-compose.week7.yml down

Do not use `--volumes` unless permanent deletion of the database is intentional.

## Test suite

    ./.venv/bin/python -m pytest tests -q -W error

Week 7 acceptance baseline: `140 passed`.

## Hyderabad evaluation status

The Hyderabad case is an unlabelled qualitative demonstration.

- Ground truth is unavailable.
- F1, IoU, precision and recall must not be reported.
- Area, polygon count and probability summaries describe predictions only.
