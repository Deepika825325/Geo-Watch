import type {
  ChangeFeature,
  ChangeFeatureCollection,
  RasterPreviewAsset,
  RasterPreviewRole,
  StoredChange,
} from "@/lib/types"

export class GeoWatchApiError extends Error {
  status: number

  constructor(
    message: string,
    status: number,
  ) {
    super(message)
    this.name = "GeoWatchApiError"
    this.status = status
  }
}

function getRequestId(): string {
  const requestId =
    process.env.NEXT_PUBLIC_GEOWATCH_REQUEST_ID

  if (!requestId) {
    throw new Error(
      "NEXT_PUBLIC_GEOWATCH_REQUEST_ID is not configured",
    )
  }

  return requestId
}

function toFeature(
  change: StoredChange,
): ChangeFeature {
  return {
    type: "Feature",
    id: change.change_id,
    geometry: change.geometry,
    properties: {
      change_id: change.change_id,
      area_m2: change.area_m2,
      perimeter_m: change.perimeter_m,
      pixel_count: change.pixel_count,
      mean_probability:
        change.mean_probability,
      maximum_probability:
        change.maximum_probability,
      qualitative: change.qualitative,
    },
  }
}

export async function fetchStoredChanges(
  signal?: AbortSignal,
): Promise<ChangeFeatureCollection> {
  const requestId = getRequestId()

  const response = await fetch(
    `/backend/v1/requests/${encodeURIComponent(requestId)}/changes`,
    {
      cache: "no-store",
      headers: {
        Accept: "application/json",
      },
      signal,
    },
  )

  if (!response.ok) {
    const body = await response.text()

    throw new GeoWatchApiError(
      body || `GeoWatch API returned ${response.status}`,
      response.status,
    )
  }

  const storedChanges =
    await response.json() as StoredChange[]

  return {
    type: "FeatureCollection",
    name: "Hyderabad detected changes",
    features: storedChanges.map(
      toFeature,
    ),
  }
}


function getRequiredHeader(
  response: Response,
  name: string,
): string {
  const value =
    response.headers.get(
      name,
    )

  if (!value) {
    throw new GeoWatchApiError(
      `GeoWatch preview response is missing ${name}`,
      response.status,
    )
  }

  return value
}

function parsePositiveIntegerHeader(
  response: Response,
  name: string,
): number {
  const value = Number(
    getRequiredHeader(
      response,
      name,
    ),
  )

  if (
    !Number.isInteger(
      value,
    )
    || value <= 0
  ) {
    throw new GeoWatchApiError(
      `GeoWatch preview returned an invalid ${name}`,
      response.status,
    )
  }

  return value
}

export async function fetchRasterPreview(
  role: RasterPreviewRole,
  signal?: AbortSignal,
): Promise<RasterPreviewAsset> {
  const requestId = getRequestId()

  const response = await fetch(
    `/backend/v1/requests/${encodeURIComponent(requestId)}/previews/${role}`,
    {
      cache: "no-store",
      headers: {
        Accept: "image/png",
      },
      signal,
    },
  )

  if (!response.ok) {
    const body = await response.text()

    throw new GeoWatchApiError(
      body || `GeoWatch preview API returned ${response.status}`,
      response.status,
    )
  }

  const contentType =
    getRequiredHeader(
      response,
      "content-type",
    )

  if (
    !contentType.startsWith(
      "image/png",
    )
  ) {
    throw new GeoWatchApiError(
      "GeoWatch preview did not return a PNG image",
      response.status,
    )
  }

  const boundsValues =
    getRequiredHeader(
      response,
      "x-geowatch-bounds",
    )
      .split(",")
      .map(Number)

  if (
    boundsValues.length !== 4
    || boundsValues.some(
      (value) =>
        !Number.isFinite(
          value,
        ),
    )
  ) {
    throw new GeoWatchApiError(
      "GeoWatch preview returned invalid geographic bounds",
      response.status,
    )
  }

  const [
    south,
    west,
    north,
    east,
  ] = boundsValues

  if (
    south >= north
    || west >= east
  ) {
    throw new GeoWatchApiError(
      "GeoWatch preview returned unordered geographic bounds",
      response.status,
    )
  }

  const blob =
    await response.blob()

  if (
    blob.size <= 0
  ) {
    throw new GeoWatchApiError(
      "GeoWatch preview returned an empty image",
      response.status,
    )
  }

  return {
    role,
    objectUrl:
      URL.createObjectURL(
        blob,
      ),
    bounds: [
      [
        south,
        west,
      ],
      [
        north,
        east,
      ],
    ],
    width:
      parsePositiveIntegerHeader(
        response,
        "x-geowatch-width",
      ),
    height:
      parsePositiveIntegerHeader(
        response,
        "x-geowatch-height",
      ),
    sourceCrs:
      getRequiredHeader(
        response,
        "x-geowatch-source-crs",
      ),
  }
}
