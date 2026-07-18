export type GeoJsonPosition = [
  longitude: number,
  latitude: number,
]

export type GeoJsonLinearRing =
  GeoJsonPosition[]

export type GeoJsonPolygonCoordinates =
  GeoJsonLinearRing[]

export type GeoJsonMultiPolygonCoordinates =
  GeoJsonPolygonCoordinates[]

export interface PolygonGeometry {
  type: "Polygon"
  coordinates: GeoJsonPolygonCoordinates
}

export interface MultiPolygonGeometry {
  type: "MultiPolygon"
  coordinates: GeoJsonMultiPolygonCoordinates
}

export type ChangeGeometry =
  | PolygonGeometry
  | MultiPolygonGeometry

export interface ChangeFeatureProperties {
  change_id: string
  area_m2: number
  perimeter_m: number
  pixel_count: number
  mean_probability: number
  maximum_probability: number
  qualitative: boolean
}

export interface ChangeFeature {
  type: "Feature"
  id?: string
  geometry: ChangeGeometry
  properties: ChangeFeatureProperties
}

export interface ChangeCollectionMetadata {
  source_crs: string
  destination_crs: string
  height: number
  width: number
  transform: number[]
  feature_count: number
  total_area_m2: number
  total_pixel_count: number
  qualitative: boolean
  ground_truth_available: boolean
  performance_metrics_reported: boolean
}

export interface ChangeFeatureCollection {
  type: "FeatureCollection"
  name?: string
  metadata?: ChangeCollectionMetadata
  features: ChangeFeature[]
}

export interface FrozenProtocol {
  checkpoint_epoch: number
  checkpoint_sha256: string
  threshold: number
  bands: string[]
  patch_size: number
  stride: number
}

export interface RasterSummary {
  height: number
  width: number
  crs: string
  patch_count: number
}

export type ArtifactRole =
  | "probability_raster"
  | "binary_mask"
  | "change_geojson"

export interface InferenceArtifact {
  role: ArtifactRole
  uri: string
  sha256: string
  qualitative: boolean
}

export interface InferenceResponse {
  request_id: string
  status: "completed"
  qualitative: boolean
  ground_truth_available: boolean
  performance_metrics_reported: boolean
  persisted: boolean
  stored_change_count: number
  protocol: FrozenProtocol
  raster: RasterSummary
  changes: ChangeFeatureCollection
  artifacts: InferenceArtifact[]
}

export interface StoredChange {
  request_id: string
  change_id: string
  geometry: ChangeGeometry
  area_m2: number
  perimeter_m: number
  pixel_count: number
  mean_probability: number
  maximum_probability: number
  qualitative: boolean
  created_at: string
}

export interface HealthResponse {
  status: "ok"
  service: string
  model_loaded: boolean
  database_connected: boolean
  protocol: FrozenProtocol
}

export type RasterPreviewRole =
  | "probability"
  | "mask"

export type RasterPreviewBounds = [
  southWest: [
    latitude: number,
    longitude: number,
  ],
  northEast: [
    latitude: number,
    longitude: number,
  ],
]

export interface RasterPreviewAsset {
  role: RasterPreviewRole
  objectUrl: string
  bounds: RasterPreviewBounds
  width: number
  height: number
  sourceCrs: string
}
