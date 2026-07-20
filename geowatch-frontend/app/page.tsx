import {
  MapPin,
  Satellite,
} from "lucide-react"
import MapViewer from "@/components/MapViewer"

export default function HomePage() {
  return (
    <main className="min-h-screen bg-bg px-5 py-6 lg:px-9">
      <header className="mx-auto flex max-w-[1600px] flex-wrap items-center justify-between gap-4 border-b border-border pb-5">
        <div className="flex items-center gap-3">
          <div className="grid size-10 place-items-center rounded-panel bg-forest text-surface">
            <Satellite
              aria-hidden="true"
              size={20}
            />
          </div>

          <div>
            <p className="font-display text-xl font-semibold tracking-[-0.03em]">
              GeoWatch
            </p>

            <p className="font-mono text-[9px] uppercase tracking-[0.18em] text-ink-faint">
              Change Detection Console
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 rounded-panel border border-forest/20 bg-forest-soft px-3 py-2 text-forest">
          <span className="size-2 rounded-full bg-forest" />

          <span className="font-mono text-[10px] uppercase tracking-[0.12em]">
            Live backend connected
          </span>
        </div>
      </header>

      <section className="mx-auto max-w-[1600px] py-8">
        <div className="mb-6">
          <div className="mb-3 flex items-center gap-2 text-forest">
            <MapPin
              aria-hidden="true"
              size={15}
            />

            <span className="font-mono text-[10px] uppercase tracking-[0.14em]">
              17.3948° N · 78.3319° E
            </span>
          </div>

          <h1 className="font-display text-4xl font-semibold tracking-[-0.045em] lg:text-5xl">
            Hyderabad change analysis
          </h1>

          <p className="mt-3 max-w-3xl text-sm leading-6 text-ink-soft">
            Inspect real model-generated change polygons, spatial extent and
            probability summaries retrieved from the PostGIS inference store.
          </p>
        </div>

        <MapViewer />
      </section>
    </main>
  )
}
