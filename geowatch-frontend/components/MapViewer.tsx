"use client"

import dynamic from "next/dynamic"
import {
  AlertTriangle,
  Database,
  RefreshCw,
} from "lucide-react"
import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react"
import type {
  ChangeMapProps,
} from "@/components/ChangeMap"
import DetectionSidebar from "@/components/DetectionSidebar"
import type {
  MapDisplayMode,
  PredictionOverlayMode,
} from "@/components/DetectionSidebar"
import {
  fetchRasterPreview,
  fetchStoredChanges,
} from "@/lib/api"
import type {
  ChangeFeatureCollection,
  RasterPreviewAsset,
} from "@/lib/types"

const ChangeMap = dynamic<
  ChangeMapProps
>(
  () => import(
    "@/components/ChangeMap"
  ),
  {
    loading: () => (
      <div className="grid h-[680px] place-items-center rounded-panel border border-border-strong bg-forest-soft">
        <div className="text-center">
          <div className="mx-auto size-7 animate-spin rounded-full border-2 border-border-strong border-t-forest" />

          <p className="mt-4 font-mono text-[10px] uppercase tracking-[0.14em] text-ink-soft">
            Initialising geospatial canvas
          </p>
        </div>
      </div>
    ),
    ssr: false,
  },
)

const requestId =
  process.env
    .NEXT_PUBLIC_GEOWATCH_REQUEST_ID
  ?? "not-configured"

function getErrorMessage(
  caughtError: unknown,
): string {
  if (
    caughtError instanceof Error
  ) {
    return caughtError.message
  }

  return "Unable to load change polygons"
}

export default function MapViewer() {
  const [
    changes,
    setChanges,
  ] = useState<
    ChangeFeatureCollection | null
  >(
    null,
  )

  const [
    probabilityPreview,
    setProbabilityPreview,
  ] = useState<
    RasterPreviewAsset | null
  >(
    null,
  )

  const [
    maskPreview,
    setMaskPreview,
  ] = useState<
    RasterPreviewAsset | null
  >(
    null,
  )

  const previewUrlsRef =
    useRef<
      string[]
    >(
      [],
    )

  const [
    predictionOverlayMode,
    setPredictionOverlayMode,
  ] = useState<
    PredictionOverlayMode
  >(
    "probability",
  )

  const [
    selectedChangeId,
    setSelectedChangeId,
  ] = useState<
    string | null
  >(
    null,
  )

  const [
    displayMode,
    setDisplayMode,
  ] = useState<
    MapDisplayMode
  >(
    "enhanced",
  )

  const [
    error,
    setError,
  ] = useState<
    string | null
  >(
    null,
  )

  const [
    loading,
    setLoading,
  ] = useState(
    true,
  )

  const applyPayload = useCallback(
    (
      collection: ChangeFeatureCollection,
      probability: RasterPreviewAsset,
      mask: RasterPreviewAsset,
    ): void => {
      for (
        const objectUrl
        of previewUrlsRef.current
      ) {
        URL.revokeObjectURL(
          objectUrl,
        )
      }

      previewUrlsRef.current = [
        probability.objectUrl,
        mask.objectUrl,
      ]

      setChanges(
        collection,
      )

      setProbabilityPreview(
        probability,
      )

      setMaskPreview(
        mask,
      )

      setSelectedChangeId(
        collection.features[
          0
        ]?.properties.change_id
        ?? null,
      )
    },
    [],
  )

  const retryLoad = useCallback(
    async (): Promise<void> => {
      setLoading(
        true,
      )

      setError(
        null,
      )

      try {
        const [
          collection,
          probability,
          mask,
        ] = await Promise.all(
          [
            fetchStoredChanges(),
            fetchRasterPreview(
              "probability",
            ),
            fetchRasterPreview(
              "mask",
            ),
          ],
        )

        applyPayload(
          collection,
          probability,
          mask,
        )
      } catch (caughtError) {
        setError(
          getErrorMessage(
            caughtError,
          ),
        )
      } finally {
        setLoading(
          false,
        )
      }
    },
    [
      applyPayload,
    ],
  )

  useEffect(
    () => {
      const controller =
        new AbortController()

      const initialLoad =
        async (): Promise<void> => {
          try {
            const [
              collection,
              probability,
              mask,
            ] = await Promise.all(
              [
                fetchStoredChanges(
                  controller.signal,
                ),
                fetchRasterPreview(
                  "probability",
                  controller.signal,
                ),
                fetchRasterPreview(
                  "mask",
                  controller.signal,
                ),
              ],
            )

            if (
              !controller.signal.aborted
            ) {
              applyPayload(
                collection,
                probability,
                mask,
              )
            } else {
              URL.revokeObjectURL(
                probability.objectUrl,
              )

              URL.revokeObjectURL(
                mask.objectUrl,
              )
            }
          } catch (caughtError) {
            if (
              caughtError instanceof DOMException
              && caughtError.name === "AbortError"
            ) {
              return
            }

            if (
              !controller.signal.aborted
            ) {
              setError(
                getErrorMessage(
                  caughtError,
                ),
              )
            }
          } finally {
            if (
              !controller.signal.aborted
            ) {
              setLoading(
                false,
              )
            }
          }
        }

      void initialLoad()

      return () => {
        controller.abort()
      }
    },
    [
      applyPayload,
    ],
  )

  useEffect(
    () => {
      return () => {
        for (
          const objectUrl
          of previewUrlsRef.current
        ) {
          URL.revokeObjectURL(
            objectUrl,
          )
        }
      }
    },
    [],
  )

  let predictionPreview:
    RasterPreviewAsset | null =
      probabilityPreview

  if (
    predictionOverlayMode === "mask"
  ) {
    predictionPreview =
      maskPreview
  }

  if (
    predictionOverlayMode === "hidden"
  ) {
    predictionPreview =
      null
  }

  if (
    loading
    && (
      !changes
      || !probabilityPreview
      || !maskPreview
    )
  ) {
    return (
      <div className="grid h-[680px] place-items-center rounded-panel border border-border-strong bg-forest-soft">
        <div className="text-center">
          <div className="mx-auto size-8 animate-spin rounded-full border-2 border-border-strong border-t-forest" />

          <p className="mt-4 font-mono text-[10px] uppercase tracking-[0.14em] text-ink-soft">
            Loading live PostGIS changes
          </p>
        </div>
      </div>
    )
  }

  if (
    error
    || !changes
    || !probabilityPreview
    || !maskPreview
  ) {
    return (
      <div className="grid h-[680px] place-items-center rounded-panel border border-ochre/30 bg-ochre-soft px-6">
        <div className="max-w-md text-center">
          <AlertTriangle
            aria-hidden="true"
            className="mx-auto text-ochre"
            size={30}
          />

          <h2 className="mt-4 font-display text-xl font-semibold">
            Live change layer unavailable
          </h2>

          <p className="mt-3 text-sm leading-6 text-ink-soft">
            {error}
          </p>

          <button
            className="mt-5 inline-flex items-center gap-2 rounded-panel bg-forest px-4 py-2 text-sm font-medium text-surface"
            onClick={() => {
              void retryLoad()
            }}
            type="button"
          >
            <RefreshCw
              aria-hidden="true"
              size={15}
            />

            Retry connection
          </button>
        </div>
      </div>
    )
  }

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <div className="inline-flex items-center gap-2 rounded-panel border border-forest/20 bg-forest-soft px-3 py-2 text-forest">
          <Database
            aria-hidden="true"
            size={14}
          />

          <span className="font-mono text-[10px] uppercase tracking-[0.12em]">
            Live PostGIS · {changes.features.length} polygons
          </span>
        </div>

        <p className="font-mono text-[9px] uppercase tracking-[0.12em] text-ink-faint">
          Request {requestId.slice(
            0,
            8,
          )} · qualitative demonstration
        </p>
      </div>

      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_360px]">
        <ChangeMap
          changes={changes}
          predictionPreview={
            predictionPreview
          }
          displayMode={displayMode}
          onSelectChange={
            setSelectedChangeId
          }
          selectedChangeId={
            selectedChangeId
          }
        />

        <DetectionSidebar
          changes={changes}
          displayMode={displayMode}
          predictionOverlayMode={
            predictionOverlayMode
          }
          onDisplayModeChange={
            setDisplayMode
          }
          onPredictionOverlayModeChange={
            setPredictionOverlayMode
          }
          onSelectChange={
            setSelectedChangeId
          }
          requestId={requestId}
          selectedChangeId={
            selectedChangeId
          }
        />
      </div>
    </div>
  )
}
