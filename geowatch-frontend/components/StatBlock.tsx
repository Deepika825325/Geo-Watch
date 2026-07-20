interface StatBlockProps {
  label: string
  value: string
  helper?: string
  tone?: "default" | "forest" | "ochre" | "slate"
}

const toneClasses = {
  default: "border-border bg-surface",
  forest: "border-forest/20 bg-forest-soft",
  ochre: "border-ochre/25 bg-ochre-soft",
  slate: "border-slate/20 bg-slate-soft",
}

export default function StatBlock({
  label,
  value,
  helper,
  tone = "default",
}: StatBlockProps) {
  return (
    <div
      className={`rounded-panel border p-4 ${toneClasses[tone]}`}
    >
      <p className="font-mono text-[9px] uppercase tracking-[0.14em] text-ink-faint">
        {label}
      </p>

      <p className="mt-2 font-display text-2xl font-semibold tracking-[-0.035em] text-ink">
        {value}
      </p>

      {helper ? (
        <p className="mt-1 text-xs leading-5 text-ink-soft">
          {helper}
        </p>
      ) : null}
    </div>
  )
}
