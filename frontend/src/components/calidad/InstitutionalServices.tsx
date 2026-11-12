import { useEffect, useState } from "react"
import { GlobalSectionSummary, MetricComparison, MetricDelta, MetricDistribution, MetricSkeleton, MetricState } from "@/components/common/CalidadMetrics"
import { fetchInstitutionalServices } from "@/services/institutionalServices"
import type { InstitutionalCurrent, InstitutionalHistory, InstitutionalService, ServiceScope } from "@/services/institutionalServices"
import type { CalidadComment } from "@/services/calidadService"

function ServiceMetric({ service, previous, year, pending }: {
  service: InstitutionalService; previous: number | null; year: string; pending: boolean;
}) {
  return <div className="space-y-4">
    <div className="flex flex-wrap items-baseline justify-between gap-2">
      <h4 className="font-semibold">{service.nombre}</h4>
      {year === "2026" && !pending && <MetricDelta previous={previous} current={service.indice} />}
    </div>
    <p className="text-sm">Resultado actual: <strong>{service.indice?.toFixed(2) ?? "—"}</strong> · {service.n.toLocaleString("es-MX")} respuestas a criterios</p>
    <MetricComparison previous={previous} current={service.indice} currentYear={year} pending={pending} />
    {service.indice !== null && <MetricDistribution favorable={service.favorable_pct} neutral={service.neutral_pct} unfavorable={service.desfavorable_pct} />}
  </div>
}

export function InstitutionalServicesView({ current, historical, loadingHistory = false, historyError = false, year, scope, onScopeChange, relatedComments }: {
  current: InstitutionalCurrent; historical: InstitutionalHistory | null; loadingHistory?: boolean; historyError?: boolean;
  year: string; scope: Pick<ServiceScope, "carrera" | "modalidad_academica">;
  onScopeChange: (scope: Pick<ServiceScope, "carrera" | "modalidad_academica">) => void;
  relatedComments: (service: InstitutionalService, comments: CalidadComment[]) => CalidadComment[];
}) {
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const selected = current.servicios.find(row => row.id === selectedId)
  const pending = year === "2026" && (loadingHistory || historyError)
  const previousFor = (service: InstitutionalService) => year === "2026" && !pending
    ? historical?.servicios.find(row => row.id === service.id)?.indice_2025 ?? null : null
  const comments = selected ? relatedComments(selected, current.comentarios) : []
  return <div className="space-y-6">
    <GlobalSectionSummary title="Servicios institucionales">
      {(scope.carrera || scope.modalidad_academica) && <p className="text-sm text-muted-foreground">Población seleccionada: {scope.carrera || "Todas las carreras"} · {scope.modalidad_academica || "Todas las modalidades"}</p>}
      <div className="flex flex-wrap items-end gap-8">
        <div><p className="text-sm text-muted-foreground">Resultado general · {year}</p><p className="mt-1 text-3xl font-semibold">{current.indice?.toFixed(2) ?? "—"}</p></div>
        <div><p className="text-sm text-muted-foreground">Respuestas a criterios</p><p className="mt-1 text-3xl font-semibold">{current.respuestas.toLocaleString("es-MX")}</p></div>
        <div><p className="text-sm text-muted-foreground">Encuestas analizadas</p><p className="mt-1 text-3xl font-semibold">{current.encuestas_con_evaluacion.toLocaleString("es-MX")}</p></div>
      </div>
      <p className="text-xs text-muted-foreground">Una encuesta puede aportar respuestas a varios criterios de servicio.</p>
      {current.indice === null ? <MetricState empty /> : <MetricDistribution favorable={current.distribucion.favorable_pct} neutral={current.distribucion.neutral_pct} unfavorable={current.distribucion.desfavorable_pct} />}
      {year === "2026" && <div className="space-y-3 border-t pt-4">
        <div className="flex flex-wrap items-baseline justify-between gap-2"><h4 className="font-semibold">Evolución global comparable</h4>{!pending && <MetricDelta previous={historical?.indice_2025 ?? null} current={historical?.indice_2026 ?? null} />}</div>
        <p className="text-xs text-muted-foreground">Solo la intersección aprobada con respuestas en ambos años; puede diferir del índice actual de todos los servicios.</p>
        {loadingHistory ? <MetricSkeleton /> : historyError ? <p role="status" className="text-xs text-muted-foreground">No fue posible cargar el comparativo histórico</p>
          : historical?.indice_2025 !== null && historical?.indice_2025 !== undefined && historical.indice_2026 !== null
            ? <><MetricComparison previous={historical.indice_2025} current={historical.indice_2026} /><p className="text-xs text-muted-foreground">{historical.criterios.length} servicios comparables · misma carrera y modalidad en ambos años</p></>
            : <MetricState />}
      </div>}
    </GlobalSectionSummary>

    <section className="space-y-5 rounded-2xl border bg-card p-6 shadow-sm" aria-label="Servicios y desglose opcional">
      <div><h3 className="font-semibold">Servicios</h3><p className="mt-1 text-sm text-muted-foreground">Consulta todos los servicios o aplica un desglose opcional por carrera y modalidad.</p></div>
      <div className="flex flex-wrap gap-4">
        <label className="text-sm">Carrera<select className="mt-1 block rounded-lg border bg-background p-2" value={scope.carrera} onChange={event => onScopeChange({ ...scope, carrera: event.target.value })}><option value="">Todas / Vista global</option>{current.opciones.carreras.map(name => <option key={name}>{name}</option>)}</select></label>
        <label className="text-sm">Modalidad<select className="mt-1 block rounded-lg border bg-background p-2" value={scope.modalidad_academica} onChange={event => onScopeChange({ ...scope, modalidad_academica: event.target.value })}><option value="">Todas las modalidades</option>{current.opciones.modalidades.map(name => <option key={name}>{name}</option>)}</select></label>
      </div>
      <div className="grid gap-4 lg:grid-cols-2">{current.servicios.map(service => <article key={service.id} className="space-y-4 rounded-xl border bg-background p-4">
        <ServiceMetric service={service} previous={previousFor(service)} year={year} pending={pending} />
        <button className="text-sm font-medium text-primary hover:underline" type="button" onClick={() => setSelectedId(service.id)}>Ver detalle de {service.nombre}</button>
      </article>)}</div>
    </section>

    {selected && <div className="fixed inset-0 z-[130] flex justify-end bg-black/30">
      <button type="button" aria-label="Cerrar detalle de servicio" onClick={() => setSelectedId(null)} className="absolute inset-0" />
      <aside role="dialog" aria-modal="true" aria-label={selected.nombre} className="relative z-10 flex h-full w-full max-w-2xl flex-col border-l bg-background shadow-2xl">
        <div className="flex items-start justify-between gap-4 border-b p-6"><div><p className="text-xs text-muted-foreground">DETALLE DEL SERVICIO · {year}</p><h3 className="mt-2 text-xl font-semibold">{selected.nombre}</h3></div><button type="button" onClick={() => setSelectedId(null)} className="rounded-xl border px-3 py-2 text-sm">Cerrar</button></div>
        <div className="flex-1 space-y-7 overflow-y-auto p-6">
          <ServiceMetric service={selected} previous={previousFor(selected)} year={year} pending={pending} />
          <section className="space-y-3"><h4 className="font-semibold">Comentarios relacionados</h4><p className="text-xs text-muted-foreground">Análisis cualitativo con SIA · En construcción</p>
            {comments.length ? comments.map((comment, i) => <p key={`${comment.fila_origen}-${comment.pregunta_id}-${i}`} className="rounded-xl border p-4 text-sm leading-6">{comment.texto}</p>) : <p className="text-sm text-muted-foreground">No hay comentarios relacionados con este servicio para la población seleccionada.</p>}
          </section>
        </div>
      </aside>
    </div>}
  </div>
}

export function InstitutionalServices({ periodo, modalidad, servicio, relatedComments }: {
  periodo: string; modalidad: string; servicio: string;
  relatedComments: (service: InstitutionalService, comments: CalidadComment[]) => CalidadComment[];
}) {
  const [selection, setSelection] = useState({ carrera: "", modalidad_academica: "" })
  const scope: ServiceScope = { periodo, modalidad, servicio, ...selection }
  const key = JSON.stringify(scope)
  const [current, setCurrent] = useState<{ key: string; data: InstitutionalCurrent | null; error: string | null } | null>(null)
  const [history, setHistory] = useState<{ key: string; data: InstitutionalHistory | null; error: boolean } | null>(null)
  useEffect(() => {
    const controller = new AbortController()
    let active = true
    const currentScope: ServiceScope = JSON.parse(key)
    // Each request has its own outcome: historical failures never discard current data.
    fetchInstitutionalServices<InstitutionalCurrent>("actual", currentScope, controller.signal).then(data => {
      if (active) setCurrent({ key, data, error: null })
    }).catch(error => {
      if (active) setCurrent({ key, data: null, error: String(error.message) })
    })
    if (currentScope.periodo === "2026") fetchInstitutionalServices<InstitutionalHistory>("historico", currentScope, controller.signal).then(data => {
      if (active) setHistory({ key, data, error: false })
    }).catch(() => {
      if (active) setHistory({ key, data: null, error: true })
    })
    return () => { active = false; controller.abort() }
  }, [key])
  if (current?.key !== key) return <MetricSkeleton />
  if (current.error || !current.data) return <p role="alert" className="text-sm text-red-700">{current.error}</p>
  return <InstitutionalServicesView current={current.data} historical={history?.key === key ? history.data : null}
    loadingHistory={periodo === "2026" && history?.key !== key} historyError={history?.key === key && history.error}
    year={periodo} scope={selection} onScopeChange={setSelection} relatedComments={relatedComments} />
}
