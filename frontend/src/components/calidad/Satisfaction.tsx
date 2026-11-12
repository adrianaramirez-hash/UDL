import { useEffect, useState } from "react"
import { GlobalSectionSummary, MetricChange, MetricComparison, MetricDelta, MetricDistribution, MetricSkeleton, MetricState, PercentageComparison } from "@/components/common/CalidadMetrics"
import { fetchSatisfaction } from "@/services/satisfaction"
import type { SatisfactionCurrent, SatisfactionHistory, ServiceScope } from "@/services/satisfaction"

function SatisfactionMetric({ current, history, year, pending }: {
  current: SatisfactionCurrent["satisfaccion"]; history: SatisfactionHistory["satisfaccion"] | null; year: string; pending: boolean;
}) {
  const previous = year === "2026" && !pending ? history?.indice_2025 ?? null : null
  return <div className="space-y-4">
    <div className="flex flex-wrap items-baseline justify-between gap-2"><h4 className="font-semibold">Satisfacción general</h4>{year === "2026" && !pending && <MetricDelta previous={previous} current={current.indice} />}</div>
    <p className="text-3xl font-semibold">{current.indice?.toFixed(2) ?? "—"}</p>
    <p className="text-sm text-muted-foreground">Respuestas a criterios: {current.n.toLocaleString("es-MX")} · {year}</p>
    <MetricComparison previous={previous} current={current.indice} currentYear={year} pending={pending} />
    {current.indice !== null && <MetricDistribution favorable={current.favorable_pct} neutral={current.neutral_pct} unfavorable={current.desfavorable_pct} />}
  </div>
}

function RecommendationMetric({ current, history, year, pending }: {
  current: SatisfactionCurrent["recomendacion"]; history: SatisfactionHistory["recomendacion"] | null; year: string; pending: boolean;
}) {
  const delta = year === "2026" && !pending ? history?.delta_pp ?? null : null
  return <div className="space-y-4">
    <div className="flex flex-wrap items-baseline justify-between gap-2"><h4 className="font-semibold">Recomendación</h4>
      <MetricChange delta={delta} unit="pp" />
    </div>
    <p className="text-3xl font-semibold">{current.si_pct === null ? "—" : `${current.si_pct.toFixed(2)}%`}</p>
    <p className="text-sm text-muted-foreground">Respuestas a criterios: {current.total_valido.toLocaleString("es-MX")} · {year}</p>
    {current.si_pct === null ? <MetricState empty /> : <>
      <div aria-hidden="true" className="flex h-2 overflow-hidden rounded-full bg-muted"><div className="bg-primary" style={{ width: `${current.si_pct}%` }} /><div className="bg-slate-300" style={{ width: `${current.no_pct ?? 0}%` }} /></div>
      <div className="flex flex-wrap gap-4 text-sm"><span>Sí: {current.si_pct.toFixed(2)}% ({current.si.toLocaleString("es-MX")})</span><span>No: {current.no_pct?.toFixed(2)}% ({current.no.toLocaleString("es-MX")})</span></div>
    </>}
    {year === "2026" && !pending && (history?.comparable ? <PercentageComparison previous={history.porcentaje_2025} current={history.porcentaje_2026} /> : current.si_pct !== null ? <MetricState /> : null)}
  </div>
}

export function SatisfactionView({ current, historical, year, scope, onScopeChange, loadingHistory = false, historyError = false }: {
  current: SatisfactionCurrent; historical: SatisfactionHistory | null; year: string;
  scope: Pick<ServiceScope, "carrera" | "modalidad_academica">;
  onScopeChange: (scope: Pick<ServiceScope, "carrera" | "modalidad_academica">) => void;
  loadingHistory?: boolean; historyError?: boolean;
}) {
  const [detail, setDetail] = useState<"satisfaccion" | "recomendacion" | null>(null)
  const pending = year === "2026" && (loadingHistory || historyError)
  const comments = current.comentarios.filter(row => row.texto_informativo)
  const metric = (kind: "satisfaccion" | "recomendacion") => kind === "satisfaccion"
    ? <SatisfactionMetric current={current.satisfaccion} history={historical?.satisfaccion ?? null} year={year} pending={pending} />
    : <RecommendationMetric current={current.recomendacion} history={historical?.recomendacion ?? null} year={year} pending={pending} />
  const historyState = () => year === "2026" && (loadingHistory ? <MetricSkeleton /> : historyError ? <p role="status" className="text-xs text-muted-foreground">No fue posible cargar el comparativo histórico</p> : null)
  return <div className="space-y-6">
    <GlobalSectionSummary title="Satisfacción y recomendación">
      <div><p className="text-sm text-muted-foreground">Encuestas analizadas</p><p className="mt-1 text-3xl font-semibold">{current.encuestas_analizadas.toLocaleString("es-MX")}</p></div>
      {(scope.carrera || scope.modalidad_academica) && <p className="text-sm text-muted-foreground">Población seleccionada: {scope.carrera || "Todas las carreras"} · {scope.modalidad_academica || "Todas las modalidades"}</p>}
      <div className="flex flex-wrap gap-4">
        <label className="text-sm">Carrera<select className="mt-1 block rounded-lg border bg-background p-2" value={scope.carrera} onChange={event => onScopeChange({ ...scope, carrera: event.target.value })}><option value="">Todas / Vista global</option>{current.opciones.carreras.map(name => <option key={name}>{name}</option>)}</select></label>
        <label className="text-sm">Modalidad<select className="mt-1 block rounded-lg border bg-background p-2" value={scope.modalidad_academica} onChange={event => onScopeChange({ ...scope, modalidad_academica: event.target.value })}><option value="">Todas las modalidades</option>{current.opciones.modalidades.map(name => <option key={name}>{name}</option>)}</select></label>
      </div>
      {historyState()}
      <div className="grid gap-4 lg:grid-cols-2">{(["satisfaccion", "recomendacion"] as const).map(kind => <article key={kind} aria-label={current[kind].nombre} className="space-y-4 rounded-xl border bg-background p-4">
        {metric(kind)}<button className="text-sm font-medium text-primary hover:underline" type="button" onClick={() => setDetail(kind)}>Ver detalle de {current[kind].nombre}</button>
      </article>)}</div>
    </GlobalSectionSummary>
    {detail && <div className="fixed inset-0 z-[130] flex justify-end bg-black/30">
      <button type="button" aria-label="Cerrar detalle del indicador" onClick={() => setDetail(null)} className="absolute inset-0" />
      <aside role="dialog" aria-modal="true" aria-label={current[detail].nombre} className="relative z-10 flex h-full w-full max-w-2xl flex-col border-l bg-background shadow-2xl">
        <div className="flex items-start justify-between gap-4 border-b p-6"><h3 className="text-xl font-semibold">{current[detail].nombre}</h3><button type="button" onClick={() => setDetail(null)} className="rounded-xl border px-3 py-2 text-sm">Cerrar</button></div>
        <div className="flex-1 space-y-7 overflow-y-auto p-6">{metric(detail)}{historyState()}
          {detail === "recomendacion" && <section className="space-y-3"><h4 className="font-semibold">¿Por qué?</h4><p className="text-xs text-muted-foreground">Análisis cualitativo con SIA · En construcción</p>
            {comments.length ? comments.map((comment, i) => <p key={`${comment.fila_origen}-${i}`} className="rounded-xl border p-4 text-sm leading-6">{comment.texto}</p>) : <p className="text-sm text-muted-foreground">No hay motivos de recomendación para la población seleccionada.</p>}
          </section>}
        </div>
      </aside>
    </div>}
  </div>
}

export function Satisfaction({ periodo, modalidad, servicio }: { periodo: string; modalidad: string; servicio: string }) {
  const [selection, setSelection] = useState({ carrera: "", modalidad_academica: "" })
  const scope: ServiceScope = { periodo, modalidad, servicio, ...selection }
  const key = JSON.stringify(scope)
  const [current, setCurrent] = useState<{ key: string; data: SatisfactionCurrent | null; error: string | null } | null>(null)
  const [history, setHistory] = useState<{ key: string; data: SatisfactionHistory | null; error: boolean } | null>(null)
  useEffect(() => {
    const controller = new AbortController()
    let active = true
    const currentScope: ServiceScope = JSON.parse(key)
    fetchSatisfaction<SatisfactionCurrent>("actual", currentScope, controller.signal).then(data => {
      if (active) setCurrent({ key, data, error: null })
    }).catch(error => {
      if (active) setCurrent({ key, data: null, error: String(error.message) })
    })
    if (currentScope.periodo === "2026") fetchSatisfaction<SatisfactionHistory>("historico", currentScope, controller.signal).then(data => {
      if (active) setHistory({ key, data, error: false })
    }).catch(() => {
      if (active) setHistory({ key, data: null, error: true })
    })
    return () => { active = false; controller.abort() }
  }, [key])
  if (current?.key !== key) return <MetricSkeleton />
  if (current.error || !current.data) return <p role="alert" className="text-sm text-red-700">{current.error}</p>
  return <SatisfactionView current={current.data} historical={history?.key === key ? history.data : null}
    loadingHistory={periodo === "2026" && history?.key !== key} historyError={history?.key === key && history.error}
    year={periodo} scope={selection} onScopeChange={setSelection} />
}
