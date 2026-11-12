import { useEffect, useState } from "react"
import { GlobalSectionSummary, MetricComparison, MetricDelta, MetricDistribution, MetricSkeleton, MetricState } from "@/components/common/CalidadMetrics"
import { fetchAcademicServices } from "@/services/academicServices"
import type { AcademicCurrent, AcademicHistory, AcademicCriterion, ServiceScope } from "@/services/academicServices"

function AcademicCriterionMetric({ criterion, previous, year, pending }: {
  criterion: AcademicCriterion; previous: number | null; year: string; pending: boolean;
}) {
  return <div className="space-y-4">
    <div className="flex flex-wrap items-baseline justify-between gap-2">
      <h4 className="font-semibold">{criterion.nombre}</h4>
      {year === "2026" && !pending && <MetricDelta previous={previous} current={criterion.indice} />}
    </div>
    <p className="text-sm">Resultado actual: <strong>{criterion.indice?.toFixed(2) ?? "—"}</strong> · {criterion.n.toLocaleString("es-MX")} respuestas a criterios</p>
    <MetricComparison previous={previous} current={criterion.indice} currentYear={year} pending={pending} />
    {criterion.indice !== null && <MetricDistribution favorable={criterion.favorable_pct} neutral={criterion.neutral_pct} unfavorable={criterion.desfavorable_pct} />}
    {criterion.id === "ACA_PRACTICAS" && <p className="text-xs text-muted-foreground">Solo respuestas válidas de quienes realizaron prácticas.</p>}
  </div>
}

export function AcademicServicesView({ current, historical, loadingHistory = false, historyError = false, year, scope, onScopeChange }: {
  current: AcademicCurrent; historical: AcademicHistory | null; loadingHistory?: boolean; historyError?: boolean;
  year: string; scope: Pick<ServiceScope, "carrera" | "modalidad_academica">;
  onScopeChange: (scope: Pick<ServiceScope, "carrera" | "modalidad_academica">) => void;
}) {
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const selected = current.criterios.find(row => row.id === selectedId)
  const pending = year === "2026" && (loadingHistory || historyError)
  const previousFor = (criterion: AcademicCriterion) => year === "2026" && !pending
    ? historical?.criterios.find(row => row.id === criterion.id)?.indice_2025 ?? null : null
  const comments = current.comentarios.filter(comment => comment.incluir_resumen && comment.texto_informativo)
  return <div className="space-y-6">
    <GlobalSectionSummary title="Servicios académicos">
      {(scope.carrera || scope.modalidad_academica) && <p className="text-sm text-muted-foreground">Población seleccionada: {scope.carrera || "Todas las carreras"} · {scope.modalidad_academica || "Todas las modalidades"}</p>}
      <div className="flex flex-wrap items-end gap-8">
        <div><p className="text-sm text-muted-foreground">Resultado general · {year}</p><p className="mt-1 text-3xl font-semibold">{current.indice?.toFixed(2) ?? "—"}</p></div>
        <div><p className="text-sm text-muted-foreground">Respuestas a criterios</p><p className="mt-1 text-3xl font-semibold">{current.respuestas.toLocaleString("es-MX")}</p></div>
        <div><p className="text-sm text-muted-foreground">Encuestas analizadas</p><p className="mt-1 text-3xl font-semibold">{current.encuestas_analizadas.toLocaleString("es-MX")}</p></div>
      </div>
      <p className="text-xs text-muted-foreground">Una encuesta puede aportar respuestas a varios criterios académicos.</p>
      {current.indice === null ? <MetricState empty /> : <MetricDistribution favorable={current.distribucion.favorable_pct} neutral={current.distribucion.neutral_pct} unfavorable={current.distribucion.desfavorable_pct} />}
      {year === "2026" && <div className="space-y-3 border-t pt-4">
        <div className="flex flex-wrap items-baseline justify-between gap-2"><h4 className="font-semibold">Evolución global comparable</h4>{!pending && <MetricDelta previous={historical?.indice_2025 ?? null} current={historical?.indice_2026 ?? null} />}</div>
        <p className="text-xs text-muted-foreground">Solo la intersección aprobada con respuestas en ambos años; puede diferir del resultado general de todos los criterios.</p>
        {loadingHistory ? <MetricSkeleton /> : historyError ? <p role="status" className="text-xs text-muted-foreground">No fue posible cargar el comparativo histórico</p>
          : historical?.indice_2025 !== null && historical?.indice_2025 !== undefined && historical.indice_2026 !== null
            ? <><MetricComparison previous={historical.indice_2025} current={historical.indice_2026} /><p className="text-xs text-muted-foreground">{historical.ids_comparables.length} criterios comparables · misma carrera y modalidad en ambos años</p></>
            : <MetricState />}
      </div>}
    </GlobalSectionSummary>

    <section className="space-y-5 rounded-2xl border bg-card p-6 shadow-sm" aria-label="Criterios académicos y desglose opcional">
      <div><h3 className="font-semibold">Criterios académicos</h3><p className="mt-1 text-sm text-muted-foreground">Consulta los doce criterios o aplica un desglose opcional por carrera y modalidad.</p></div>
      <div className="flex flex-wrap gap-4">
        <label className="text-sm">Carrera<select className="mt-1 block rounded-lg border bg-background p-2" value={scope.carrera} onChange={event => onScopeChange({ ...scope, carrera: event.target.value })}><option value="">Todas / Vista global</option>{current.opciones.carreras.map(name => <option key={name}>{name}</option>)}</select></label>
        <label className="text-sm">Modalidad<select className="mt-1 block rounded-lg border bg-background p-2" value={scope.modalidad_academica} onChange={event => onScopeChange({ ...scope, modalidad_academica: event.target.value })}><option value="">Todas las modalidades</option>{current.opciones.modalidades.map(name => <option key={name}>{name}</option>)}</select></label>
      </div>
      <div className="grid gap-4 lg:grid-cols-2">{current.criterios.map(criterion => <article key={criterion.id} className="space-y-4 rounded-xl border bg-background p-4">
        <AcademicCriterionMetric criterion={criterion} previous={previousFor(criterion)} year={year} pending={pending} />
        <button className="text-sm font-medium text-primary hover:underline" type="button" onClick={() => setSelectedId(criterion.id)}>Ver detalle de {criterion.nombre}</button>
      </article>)}</div>
    </section>

    {selected && <div className="fixed inset-0 z-[130] flex justify-end bg-black/30">
      <button type="button" aria-label="Cerrar detalle del criterio" onClick={() => setSelectedId(null)} className="absolute inset-0" />
      <aside role="dialog" aria-modal="true" aria-label={selected.nombre} className="relative z-10 flex h-full w-full max-w-2xl flex-col border-l bg-background shadow-2xl">
        <div className="flex items-start justify-between gap-4 border-b p-6"><div><p className="text-xs text-muted-foreground">DETALLE DEL CRITERIO · {year}</p><h3 className="mt-2 text-xl font-semibold">{selected.nombre}</h3></div><button type="button" onClick={() => setSelectedId(null)} className="rounded-xl border px-3 py-2 text-sm">Cerrar</button></div>
        <div className="flex-1 space-y-7 overflow-y-auto p-6">
          <AcademicCriterionMetric criterion={selected} previous={previousFor(selected)} year={year} pending={pending} />
          <p className="text-xs text-muted-foreground">Encuestas analizadas con respuesta válida al criterio: {selected.encuestas_validas.toLocaleString("es-MX")}</p>
          <section className="space-y-3"><h4 className="font-semibold">Comentarios de Servicios académicos</h4><p className="text-xs text-muted-foreground">Análisis cualitativo con SIA · En construcción</p><p className="text-xs text-muted-foreground">Correspondientes a la carrera y modalidad seleccionadas.</p>
            {comments.length ? comments.map((comment, i) => <p key={`${comment.fila_origen}-${comment.pregunta_id}-${i}`} className="rounded-xl border p-4 text-sm leading-6">{comment.texto}</p>) : <p className="text-sm text-muted-foreground">No hay comentarios de Servicios académicos para la población seleccionada.</p>}
          </section>
        </div>
      </aside>
    </div>}
  </div>
}

export function AcademicServices({ periodo, modalidad, servicio }: { periodo: string; modalidad: string; servicio: string }) {
  const [selection, setSelection] = useState({ carrera: "", modalidad_academica: "" })
  const scope: ServiceScope = { periodo, modalidad, servicio, ...selection }
  const key = JSON.stringify(scope)
  const [current, setCurrent] = useState<{ key: string; data: AcademicCurrent | null; error: string | null } | null>(null)
  const [history, setHistory] = useState<{ key: string; data: AcademicHistory | null; error: boolean } | null>(null)
  useEffect(() => {
    const controller = new AbortController()
    let active = true
    const currentScope: ServiceScope = JSON.parse(key)
    fetchAcademicServices<AcademicCurrent>("actual", currentScope, controller.signal).then(data => {
      if (active) setCurrent({ key, data, error: null })
    }).catch(error => {
      if (active) setCurrent({ key, data: null, error: String(error.message) })
    })
    if (currentScope.periodo === "2026") fetchAcademicServices<AcademicHistory>("historico", currentScope, controller.signal).then(data => {
      if (active) setHistory({ key, data, error: false })
    }).catch(() => {
      if (active) setHistory({ key, data: null, error: true })
    })
    return () => { active = false; controller.abort() }
  }, [key])
  if (current?.key !== key) return <MetricSkeleton />
  if (current.error || !current.data) return <p role="alert" className="text-sm text-red-700">{current.error}</p>
  return <AcademicServicesView current={current.data} historical={history?.key === key ? history.data : null}
    loadingHistory={periodo === "2026" && history?.key !== key} historyError={history?.key === key && history.error}
    year={periodo} scope={selection} onScopeChange={setSelection} />
}
