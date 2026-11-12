import type { ReactNode } from "react"

export function MetricState({ empty = false }: { empty?: boolean }) {
  return <p className="text-sm text-muted-foreground">{empty ? "Sin respuestas en este periodo" : "Sin comparable histórico"}</p>
}

export function MetricSkeleton() {
  return <div role="status" aria-label="Cargando comparativo" className="animate-pulse space-y-3 rounded-xl border p-4"><span className="sr-only">Cargando comparativo</span><div className="h-3 w-2/3 rounded bg-muted" /><div className="h-3 rounded bg-muted" /></div>
}

export function MetricDelta({ previous, current }: { previous: number | null; current: number | null }) {
  if (previous === null || current === null) return null
  const delta = Number((current - previous).toFixed(2))
  return <span className={`shrink-0 whitespace-nowrap text-sm font-semibold ${delta > 0 ? "text-emerald-700" : delta < 0 ? "text-red-700" : "text-muted-foreground"}`}>
    {delta === 0 ? "→ estable" : `${delta > 0 ? "↑ +" : "↓ "}${delta.toFixed(2)} pts`}
  </span>
}

export function MetricChange({ delta, unit = "pts", neutral = false }: { delta: number | null; unit?: "pts" | "pp"; neutral?: boolean }) {
  if (delta === null) return null
  return <span title={unit === "pp" ? "puntos porcentuales" : "puntos de índice"} className={`shrink-0 whitespace-nowrap text-sm font-semibold ${neutral ? "text-muted-foreground" : delta > 0 ? "text-emerald-700" : delta < 0 ? "text-red-700" : "text-muted-foreground"}`}>
    {delta === 0 ? "→ estable" : `${delta > 0 ? "↑ +" : "↓ −"}${Math.abs(delta).toFixed(2)} ${unit}`}
  </span>
}

export function MetricBar({ label, value, percentage = false }: { label: string; value: number; percentage?: boolean }) {
  return <div className="calidad-metric-bar grid grid-cols-[3rem_minmax(0,1fr)_4.5rem] items-center gap-3 text-xs">
    <span>{label}</span>
    <div role="meter" aria-label={label} aria-valuemin={0} aria-valuemax={100} aria-valuenow={value} className="h-3 overflow-hidden rounded-full bg-muted">
      <div className={`h-full rounded-full ${label === "2025" ? "bg-slate-400" : "bg-primary"}`} style={{ width: `${Math.max(0, Math.min(100, value))}%` }} />
    </div>
    <span className="whitespace-nowrap text-right font-medium tabular-nums">{value.toFixed(2)}{percentage ? "%" : ""}</span>
  </div>
}

export function PercentageComparison({ previous, current }: { previous: number | null; current: number | null }) {
  return <div className="space-y-2 border-t pt-3">
    {previous !== null && <MetricBar label="2025" value={previous} percentage />}
    {current !== null && <MetricBar label="2026" value={current} percentage />}
    <p className="text-[10px] text-muted-foreground">Escala 0–100%</p>
  </div>
}

export function MetricComparison({ previous, current, currentYear = "2026", pending = false, showHistoricalState = true }: { previous: number | null; current: number | null; currentYear?: string; pending?: boolean; showHistoricalState?: boolean }) {
  if (current === null) return <div className="space-y-2">{previous !== null && currentYear === "2026" && <MetricBar label="2025" value={previous} />}<MetricState empty /></div>
  return <div className="space-y-2">
    {previous !== null && currentYear === "2026" && <MetricBar label="2025" value={previous} />}
    <MetricBar label={currentYear} value={current} />
    <p className="text-[10px] text-muted-foreground">Escala 0–100</p>
    {showHistoricalState && !pending && (previous === null || currentYear !== "2026") && <MetricState />}
  </div>
}

export function MetricDistribution({ favorable, neutral, unfavorable }: { favorable: number | null; neutral: number | null; unfavorable: number | null }) {
  const values = [favorable, neutral, unfavorable]
  if (values.every(value => value === null)) return <p className="text-xs text-muted-foreground">Distribución no disponible</p>
  const labels = ["Favorable", "Neutral", "Desfavorable"]
  const colors = ["bg-emerald-600", "bg-amber-400", "bg-red-500"]
  return <div className="space-y-2">
    <div aria-hidden="true" className="flex h-2 overflow-hidden rounded-full bg-muted">{values.map((value, i) => <div key={labels[i]} className={colors[i]} style={{ width: `${Math.max(0, Math.min(100, value ?? 0))}%` }} />)}</div>
    <div className="flex flex-wrap gap-3 text-xs text-muted-foreground">{values.map((value, i) => <span key={labels[i]}>{labels[i]} {value === null ? "—" : `${value.toFixed(1)}%`}</span>)}</div>
  </div>
}

export function MetricCard({ title, current, previous, responses, currentYear = "2026", distribution, loading = false, comparisonError = false }: {
  title: string; current: number | null; previous: number | null; responses: number; currentYear?: string;
  distribution?: { favorable_pct: number | null; neutral_pct: number | null; desfavorable_pct: number | null }; loading?: boolean; comparisonError?: boolean;
}) {
  return <article className="space-y-4 rounded-xl border bg-card p-4">
    <div className="flex flex-wrap items-baseline justify-between gap-2"><h4 className="font-semibold">{title}</h4>{currentYear === "2026" && !loading && !comparisonError && <MetricDelta previous={previous} current={current} />}</div>
    <p className="text-xs text-muted-foreground">{responses.toLocaleString("es-MX")} respuestas a criterios · Resultado actual {current?.toFixed(2) ?? "—"}</p>
    <MetricComparison previous={loading || comparisonError ? null : previous} current={current} currentYear={currentYear} pending={loading || comparisonError} />
    {loading && <MetricSkeleton />}
    {comparisonError && <p className="text-xs text-muted-foreground">No fue posible cargar el comparativo histórico</p>}
    {current !== null && distribution && <MetricDistribution favorable={distribution.favorable_pct} neutral={distribution.neutral_pct} unfavorable={distribution.desfavorable_pct} />}
  </article>
}

export function GlobalSectionSummary({ title, children }: { title: string; children: ReactNode }) {
  return <section aria-label={title} className="calidad-global-summary space-y-5 rounded-2xl border bg-card p-6 shadow-sm"><div><p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Todas / Vista global</p><h3 className="mt-2 text-xl font-semibold">{title}</h3></div>{children}</section>
}
