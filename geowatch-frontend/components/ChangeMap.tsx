"use client"

import type {
  Feature,
  Geometry,
} from "geojson"
import {
  geoJSON,
} from "leaflet"
import type {
  Layer,
  PathOptions,
  StyleFunction,
} from "leaflet"
import {
  GeoJSON,
  ImageOverlay,
  MapContainer,
  Pane,
  TileLayer,
  useMap,
} from "react-leaflet"
import {
  useCallback,
  useEffect,
  useState,
} from "react"
import CompareSlider from "@/components/CompareSlider"
import type {
  MapDisplayMode,
} from "@/components/DetectionSidebar"
import type {
  ChangeFeatureCollection,
  ChangeFeatureProperties,
  RasterPreviewAsset,
} from "@/lib/types"

const mapCenter: [
  number,
  number,
] = [
  17.3948,
  78.3319,
]

export interface ChangeMapProps {
  changes: ChangeFeatureCollection
  predictionPreview: RasterPreviewAsset | null
  selectedChangeId: string | null
  displayMode: MapDisplayMode
  onSelectChange: (
    changeId: string,
  ) => void
}

function formatNumber(
  value: number,
  maximumFractionDigits: number,
): string {
  return new Intl.NumberFormat(
    "en-IN",
    {
      maximumFractionDigits,
    },
  ).format(
    value,
  )
}

function bindChangeInteraction(
  feature: Feature<
    Geometry,
    ChangeFeatureProperties
  >,
  layer: Layer,
  onSelectChange: (
    changeId: string,
  ) => void,
): void {
  if (
    feature.geometry.type !== "Polygon"
    && feature.geometry.type !== "MultiPolygon"
  ) {
    return
  }

  const properties =
    feature.properties

  const meanProbability = (
    properties.mean_probability
    * 100
  ).toFixed(
    1,
  )

  const maximumProbability = (
    properties.maximum_probability
    * 100
  ).toFixed(
    1,
  )

  layer.on(
    "click",
    () => {
      onSelectChange(
        properties.change_id,
      )
    },
  )

  layer.bindPopup(`
    <div style="min-width: 210px">
      <p style="margin: 0 0 4px; font-family: var(--font-ibm-plex-mono); font-size: 10px; letter-spacing: 0.12em; color: var(--color-ochre); text-transform: uppercase">
        Detected change
      </p>
      <strong style="display: block; margin-bottom: 10px; font-family: var(--font-space-grotesk); font-size: 15px">
        ${properties.change_id}
      </strong>
      <dl style="display: grid; grid-template-columns: 1fr auto; gap: 7px 14px; margin: 0; font-size: 12px">
        <dt style="color: var(--color-ink-soft)">Area</dt>
        <dd style="margin: 0; font-family: var(--font-ibm-plex-mono)">${formatNumber(properties.area_m2, 1)} m²</dd>
        <dt style="color: var(--color-ink-soft)">Perimeter</dt>
        <dd style="margin: 0; font-family: var(--font-ibm-plex-mono)">${formatNumber(properties.perimeter_m, 1)} m</dd>
        <dt style="color: var(--color-ink-soft)">Mean probability</dt>
        <dd style="margin: 0; font-family: var(--font-ibm-plex-mono)">${meanProbability}%</dd>
        <dt style="color: var(--color-ink-soft)">Maximum</dt>
        <dd style="margin: 0; font-family: var(--font-ibm-plex-mono)">${maximumProbability}%</dd>
        <dt style="color: var(--color-ink-soft)">Pixels</dt>
        <dd style="margin: 0; font-family: var(--font-ibm-plex-mono)">${formatNumber(properties.pixel_count, 0)}</dd>
      </dl>
    </div>
  `)
}

function FitChangeBounds({
  changes,
}: Pick<
  ChangeMapProps,
  "changes"
>) {
  const map = useMap()

  useEffect(
    () => {
      if (
        changes.features.length
        === 0
      ) {
        return
      }

      const layer = geoJSON(
        changes,
      )

      const bounds =
        layer.getBounds()

      if (bounds.isValid()) {
        map.fitBounds(
          bounds,
          {
            padding: [
              40,
              40,
            ],
            maxZoom: 16,
          },
        )
      }
    },
    [
      changes,
      map,
    ],
  )

  return null
}

export default function ChangeMap({
  changes,
  predictionPreview,
  selectedChangeId,
  displayMode,
  onSelectChange,
}: ChangeMapProps) {
  const [
    comparisonPosition,
    setComparisonPosition,
  ] = useState(
    58,
  )

  const getFeatureStyle =
    useCallback<StyleFunction>(
      (
        feature,
      ): PathOptions => {
        const selected =
          feature?.properties
            ?.change_id
          === selectedChangeId

        return {
          color: selected
            ? "var(--color-forest)"
            : "var(--color-ochre)",
          fillColor:
            "var(--color-ochre)",
          fillOpacity: selected
            ? 0.64
            : 0.4,
          opacity: 0.96,
          weight: selected
            ? 4
            : 2,
        }
      },
      [
        selectedChangeId,
      ],
    )

  return (
    <div className="relative h-[680px] overflow-hidden rounded-panel border border-border-strong bg-forest-soft shadow-[0_22px_60px_rgba(30,38,32,0.12)]">
      <MapContainer
        center={mapCenter}
        className="h-full w-full"
        scrollWheelZoom
        zoom={14}
        zoomControl
      >
        <TileLayer
          attribution="Tiles © Esri"
          className={
            displayMode === "enhanced"
              ? "satellite-imagery satellite-imagery--enhanced"
              : "satellite-imagery"
          }
          key={
            `baseline-${displayMode}`
          }
          url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        />

        <Pane
          name="after-imagery"
          style={{
            clipPath: `inset(0 ${100 - comparisonPosition}% 0 0)`,
            zIndex: 350,
          }}
        >
          <TileLayer
            attribution="Tiles © Esri"
            className={
              displayMode === "enhanced"
                ? "satellite-imagery satellite-imagery--enhanced"
                : "satellite-imagery"
            }
            key={
              `recent-${displayMode}`
            }
            url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
          />
        </Pane>

        {predictionPreview ? (
          <Pane
            name="prediction-preview"
            style={{
              zIndex: 425,
            }}
          >
            <ImageOverlay
              bounds={
                predictionPreview.bounds
              }
              key={
                `${predictionPreview.role}-${predictionPreview.objectUrl}`
              }
              opacity={
                predictionPreview.role === "mask"
                  ? 0.86
                  : 0.72
              }
              url={
                predictionPreview.objectUrl
              }
            />
          </Pane>
        ) : null}

        <Pane
          name="change-polygons"
          style={{
            zIndex: 450,
          }}
        >
          <GeoJSON
            key={
              selectedChangeId
              ?? "no-selection"
            }
            data={changes}
            onEachFeature={(
              feature,
              layer,
            ) => {
              bindChangeInteraction(
                feature,
                layer,
                onSelectChange,
              )
            }}
            style={getFeatureStyle}
          />
        </Pane>

        <FitChangeBounds
          changes={changes}
        />
      </MapContainer>

      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-y-0 z-[700] w-px bg-surface shadow-[0_0_0_1px_rgba(30,38,32,0.18)]"
        style={{
          left: `${comparisonPosition}%`,
        }}
      >
        <div className="absolute top-1/2 left-1/2 grid size-9 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full border-2 border-surface bg-forest font-mono text-[10px] text-surface shadow-lg">
          ↔
        </div>
      </div>

      <div className="pointer-events-none absolute top-5 left-5 z-[800] rounded-panel border border-border bg-surface/92 px-3 py-2 shadow-md backdrop-blur">
        <p className="font-mono text-[9px] uppercase tracking-[0.14em] text-ink-faint">
          Baseline layer
        </p>

        <p className="mt-1 font-display text-sm font-medium text-ink">
          Historical observation
        </p>
      </div>

      <div className="pointer-events-none absolute top-5 right-5 z-[800] rounded-panel border border-border bg-surface/92 px-3 py-2 text-right shadow-md backdrop-blur">
        <p className="font-mono text-[9px] uppercase tracking-[0.14em] text-forest">
          Recent layer
        </p>

        <p className="mt-1 font-display text-sm font-medium text-ink">
          Latest observation
        </p>
      </div>

      <div className="pointer-events-none absolute right-5 bottom-24 z-[800] rounded-panel border border-ochre/30 bg-surface/94 p-3 shadow-md backdrop-blur">
        <div className="flex items-center gap-2">
          <span className="size-3 rounded-sm border border-ochre bg-ochre/40" />

          <span className="font-mono text-[10px] uppercase tracking-[0.12em] text-ink-soft">
            {changes.features.length} detected changes
          </span>
        </div>
      </div>

      <CompareSlider
        onChange={setComparisonPosition}
        value={comparisonPosition}
      />
    </div>
  )
}
