"use client"

interface CompareSliderProps {
  value: number
  onChange: (value: number) => void
}

export default function CompareSlider({
  value,
  onChange,
}: CompareSliderProps) {
  return (
    <div className="absolute inset-x-5 bottom-5 z-[1100] rounded-panel border border-border bg-surface/95 px-4 py-3 shadow-[0_12px_34px_rgba(30,38,32,0.16)] backdrop-blur">
      <div className="mb-2 flex items-center justify-between">
        <span className="font-mono text-[10px] font-medium uppercase tracking-[0.14em] text-ink-soft">
          Baseline
        </span>

        <span className="font-mono text-[10px] text-ink-faint">
          {value}%
        </span>

        <span className="font-mono text-[10px] font-medium uppercase tracking-[0.14em] text-forest">
          Recent
        </span>
      </div>

      <input
        aria-label="Compare baseline and recent imagery"
        className="compare-range block"
        max={100}
        min={0}
        onChange={(event) => {
          onChange(
            Number(
              event.target.value,
            ),
          )
        }}
        type="range"
        value={value}
      />
    </div>
  )
}
