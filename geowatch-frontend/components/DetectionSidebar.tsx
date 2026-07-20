"use client"

import {
  Activity,
  Database,
  Layers3,
  MapPin,
  Maximize2,
} from "lucide-react"
import StatBlock from "@/components/StatBlock"
import type {
  ChangeFeature,
  ChangeFeatureCollection,
} from "@/lib/types"

export type MapDisplayMode =
  | "natural"
  | "enhanced"

export type PredictionOverlayMode =
  | "probability"
  | "mask"
  | "hidden"

interface DetectionSidebarProps {
  changes: ChangeFeatureCollection
  selectedChangeId: string | null
  displayMode: MapDisplayMode
  predictionOverlayMode: PredictionOverlayMode
  requestId: string
  onSelectChange: (changeId: string) => void
  onDisplayModeChange: (
    mode: MapDisplayMode,
  ) => void
  onPredictionOverlayModeChange: (
    mode: PredictionOverlayMode,
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

function formatArea(
  areaM2: number,
): string {
  if (
    areaM2
    >= 1_000_000
  ) {
    return `${formatNumber(
      areaM2 / 1_000_000,
      2,
    )} km²`
  }

  return `${formatNumber(
    areaM2,
    0,
  )} m²`
}

function formatProbability(
  probability: number,
): string {
  return `${(
    probability
    * 100
  ).toFixed(
    1,
  )}%`
}

function getSelectedFeature(
  features: ChangeFeature[],
  selectedChangeId: string | null,
): ChangeFeature | null {
  if (!selectedChangeId) {
    return null
  }

  return (
    features.find(
      (feature) =>
        feature.properties.change_id
        === selectedChangeId,
    )
    ?? null
  )
}

export default function DetectionSidebar({
  changes,
  selectedChangeId,
  displayMode,
  predictionOverlayMode,
  requestId,
  onSelectChange,
  onDisplayModeChange,
  onPredictionOverlayModeChange,
}: DetectionSidebarProps) {
  const features =
    changes.features

  const totalAreaM2 =
    features.reduce(
      (
        total,
        feature,
      ) =>
        total
        + feature.properties.area_m2,
      0,
    )

  const totalPixels =
    features.reduce(
      (
        total,
        feature,
      ) =>
        total
        + feature.properties.pixel_count,
      0,
    )

  const averageProbability =
    features.length > 0
      ? features.reduce(
          (
            total,
            feature,
          ) =>
            total
            + feature.properties.mean_probability,
          0,
        )
        / features.length
      : 0

  const maximumProbability =
    features.reduce(
      (
        maximum,
        feature,
      ) =>
        Math.max(
          maximum,
          feature.properties.maximum_probability,
        ),
      0,
    )

  const rankedFeatures = [
    ...features,
  ]
    .sort(
      (
        first,
        second,
      ) =>
        second.properties.area_m2
        - first.properties.area_m2,
    )
    .slice(
      0,
      5,
    )

  const selectedFeature =
    getSelectedFeature(
      features,
      selectedChangeId,
    )

  return (
    <aside className="flex h-full flex-col overflow-hidden rounded-panel border border-border bg-surface shadow-[0_18px_46px_rgba(30,38,32,0.08)]">
      <div className="border-b border-border p-5">
        <div className="flex items-center gap-2 text-forest">
          <Activity
            aria-hidden="true"
            size={16}
          />

          <p className="font-mono text-[10px] uppercase tracking-[0.14em]">
            Detection summary
          </p>
        </div>

        <h2 className="mt-3 font-display text-2xl font-semibold tracking-[-0.04em]">
          Spatial evidence
        </h2>

        <p className="mt-2 text-sm leading-6 text-ink-soft">
          Model-generated change regions stored as valid PostGIS polygons.
        </p>
      </div>

      <div className="border-b border-border p-5">
        <div className="flex items-center gap-2">
          <MapPin
            aria-hidden="true"
            className="text-slate"
            size={15}
          />

          <p className="font-display text-sm font-semibold">
            Kokapet, Hyderabad
          </p>
        </div>

        <dl className="mt-4 space-y-3">
          <div className="flex items-center justify-between gap-4">
            <dt className="text-xs text-ink-soft">
              Coordinates
            </dt>

            <dd className="font-mono text-[10px] text-ink">
              17.3948° N · 78.3319° E
            </dd>
          </div>

          <div className="flex items-center justify-between gap-4">
            <dt className="text-xs text-ink-soft">
              Request
            </dt>

            <dd className="font-mono text-[10px] text-ink">
              {requestId.slice(
                0,
                8,
              )}
            </dd>
          </div>

          <div className="flex items-center justify-between gap-4">
            <dt className="text-xs text-ink-soft">
              Evaluation
            </dt>

            <dd className="font-mono text-[10px] uppercase text-ochre">
              Qualitative
            </dd>
          </div>
        </dl>
      </div>

      <div className="grid grid-cols-2 gap-3 border-b border-border p-5">
        <StatBlock
          helper="Stored polygons"
          label="Detections"
          tone="forest"
          value={formatNumber(
            features.length,
            0,
          )}
        />

        <StatBlock
          helper="Predicted extent"
          label="Total area"
          tone="ochre"
          value={formatArea(
            totalAreaM2,
          )}
        />

        <StatBlock
          helper="Mean polygon score"
          label="Mean probability"
          tone="slate"
          value={formatProbability(
            averageProbability,
          )}
        />

        <StatBlock
          helper={`${formatNumber(
            totalPixels,
            0,
          )} raster pixels`}
          label="Maximum"
          value={formatProbability(
            maximumProbability,
          )}
        />
      </div>

      <div className="border-b border-border p-5">
        <div className="mb-3 flex items-center gap-2">
          <Layers3
            aria-hidden="true"
            className="text-slate"
            size={15}
          />

          <p className="font-mono text-[9px] uppercase tracking-[0.14em] text-ink-faint">
            Map rendering
          </p>
        </div>

        <div className="grid grid-cols-2 rounded-panel border border-border bg-bg p-1">
          {(
            [
              "natural",
              "enhanced",
            ] as MapDisplayMode[]
          ).map((mode) => {
            const active =
              displayMode === mode

            return (
              <button
                key={mode}
                className={`rounded-[4px] px-3 py-2 font-mono text-[10px] uppercase tracking-[0.1em] transition ${
                  active
                    ? "bg-forest text-surface shadow-sm"
                    : "text-ink-soft hover:text-ink"
                }`}
                onClick={() => {
                  onDisplayModeChange(
                    mode,
                  )
                }}
                type="button"
              >
                {mode}
              </button>
            )
          })}
        </div>

        <p className="mt-2 text-[11px] leading-5 text-ink-faint">
          Enhanced mode increases contrast for visual inspection. It does not alter model predictions.
        </p>
      </div>

      <div className="border-b border-border p-5">
        <div className="mb-3 flex items-center gap-2">
          <Layers3
            aria-hidden="true"
            className="text-ochre"
            size={15}
          />

          <p className="font-mono text-[9px] uppercase tracking-[0.14em] text-ink-faint">
            Prediction overlay
          </p>
        </div>

        <div className="grid grid-cols-3 rounded-panel border border-border bg-bg p-1">
          {(
            [
              "probability",
              "mask",
              "hidden",
            ] as PredictionOverlayMode[]
          ).map((mode) => {
            const active =
              predictionOverlayMode === mode

            return (
              <button
                key={mode}
                className={`rounded-[4px] px-2 py-2 font-mono text-[9px] uppercase tracking-[0.08em] transition ${
                  active
                    ? "bg-ochre text-surface shadow-sm"
                    : "text-ink-soft hover:text-ink"
                }`}
                onClick={() => {
                  onPredictionOverlayModeChange(
                    mode,
                  )
                }}
                type="button"
              >
                {mode}
              </button>
            )
          })}
        </div>

        <p className="mt-2 text-[11px] leading-5 text-ink-faint">
          Probability shows continuous model confidence. Mask shows thresholded change pixels. Hidden removes the raster layer.
        </p>
      </div>

      <div className="border-b border-border p-5">
        <div className="mb-3 flex items-center gap-2">
          <Maximize2
            aria-hidden="true"
            className="text-ochre"
            size={14}
          />

          <p className="font-mono text-[9px] uppercase tracking-[0.14em] text-ink-faint">
            Selected detection
          </p>
        </div>

        {selectedFeature ? (
          <div className="rounded-panel border border-ochre/25 bg-ochre-soft p-4">
            <p className="font-display text-base font-semibold">
              {
                selectedFeature.properties
                  .change_id
              }
            </p>

            <dl className="mt-4 grid grid-cols-2 gap-x-4 gap-y-3">
              <div>
                <dt className="text-[10px] text-ink-faint">
                  Area
                </dt>

                <dd className="mt-1 font-mono text-xs text-ink">
                  {formatArea(
                    selectedFeature.properties
                      .area_m2,
                  )}
                </dd>
              </div>

              <div>
                <dt className="text-[10px] text-ink-faint">
                  Pixels
                </dt>

                <dd className="mt-1 font-mono text-xs text-ink">
                  {formatNumber(
                    selectedFeature.properties
                      .pixel_count,
                    0,
                  )}
                </dd>
              </div>

              <div>
                <dt className="text-[10px] text-ink-faint">
                  Mean
                </dt>

                <dd className="mt-1 font-mono text-xs text-ink">
                  {formatProbability(
                    selectedFeature.properties
                      .mean_probability,
                  )}
                </dd>
              </div>

              <div>
                <dt className="text-[10px] text-ink-faint">
                  Maximum
                </dt>

                <dd className="mt-1 font-mono text-xs text-ink">
                  {formatProbability(
                    selectedFeature.properties
                      .maximum_probability,
                  )}
                </dd>
              </div>
            </dl>
          </div>
        ) : (
          <p className="text-xs leading-5 text-ink-soft">
            Select a polygon on the map or from the ranked list.
          </p>
        )}
      </div>

      <div className="min-h-0 flex-1 p-5">
        <div className="mb-3 flex items-center justify-between">
          <p className="font-mono text-[9px] uppercase tracking-[0.14em] text-ink-faint">
            Largest detections
          </p>

          <div className="flex items-center gap-1 text-forest">
            <Database
              aria-hidden="true"
              size={12}
            />

            <span className="font-mono text-[8px] uppercase">
              Live
            </span>
          </div>
        </div>

        <div className="space-y-2">
          {rankedFeatures.map(
            (
              feature,
              index,
            ) => {
              const active =
                feature.properties
                  .change_id
                === selectedChangeId

              return (
                <button
                  key={
                    feature.properties
                      .change_id
                  }
                  className={`flex w-full items-center justify-between gap-3 rounded-panel border px-3 py-3 text-left transition ${
                    active
                      ? "border-ochre/40 bg-ochre-soft"
                      : "border-border bg-bg hover:border-border-strong"
                  }`}
                  onClick={() => {
                    onSelectChange(
                      feature.properties
                        .change_id,
                    )
                  }}
                  type="button"
                >
                  <div className="flex min-w-0 items-center gap-3">
                    <span className="font-mono text-[9px] text-ink-faint">
                      {String(
                        index + 1,
                      ).padStart(
                        2,
                        "0",
                      )}
                    </span>

                    <span className="truncate font-mono text-[10px] text-ink">
                      {
                        feature.properties
                          .change_id
                      }
                    </span>
                  </div>

                  <span className="shrink-0 font-mono text-[9px] text-ink-soft">
                    {formatArea(
                      feature.properties
                        .area_m2,
                    )}
                  </span>
                </button>
              )
            },
          )}
        </div>
      </div>
    </aside>
  )
}
