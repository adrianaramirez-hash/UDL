import { useEffect, useState } from "react"
import { GlobalSectionSummary, MetricComparison, MetricDistribution, MetricSkeleton, MetricState } from "@/components/common/CalidadMetrics"
import { fetchContinuingEducation } from "@/services/continuingEducation"
import type { ContinuingBinary, ContinuingCriterion, ContinuingCurrent, ContinuingScope, ContinuingSelection } from "@/services/continuingEducation"

function BinaryDistribution({ data }: { data: ContinuingBinary }) {
  if (data.si_pct === null) return <MetricState empty />
  return <div className="space-y-3">
    <div aria-hidden="true" className="flex h-2 overflow-hidden rounded-full bg-muted"><div className="bg-primary" style={{ width: `${data.si_pct}%` }} /><div className="bg-slate-300" style={{ width: `${data.no_pct ?? 0}%` }} /></div>
    <div className="flex flex-wrap gap-4 text-sm"><span>Sí: {data.si_pct.toFixed(2)}% ({data.si.toLocaleString("es-MX")})</span><span>No: {data.no_pct?.toFixed(2)}% ({data.no.toLocaleString("es-MX")})</span></div>
  </div>
}

function CriterionMetric({ criterion, year }: { criterion: ContinuingCriterion; year: string }) {
  return <div className="space-y-4"><h4 className="font-semibold">{criterion.nombre}</h4>
    {criterion.id === "EC_RECOMIENDA" && <p className="text-xs text-muted-foreground">Recomendaría los programas de Educación Continua a otros estudiantes</p>}
    <p className="text-sm">Resultado actual: <strong>{criterion.resultado?.toFixed(2) ?? "—"}{criterion.resultado !== null && criterion.unidad === "porcentaje" ? "%" : ""}</strong> · {year}</p>
    <p className="text-xs text-muted-foreground">Respuestas a criterios: {criterion.n.toLocaleString("es-MX")}</p>
    {criterion.tipo_dato === "BINARIA" && criterion.binaria ? <BinaryDistribution data={criterion.binaria} />
      : <><MetricComparison previous={null} current={criterion.resultado} currentYear={year} showHistoricalState={false} />
          {criterion.resultado !== null && <MetricDistribution favorable={criterion.favorable_pct} neutral={criterion.neutral_pct} unfavorable={criterion.desfavorable_pct} />}</>}
  </div>
}

export function ContinuingEducationView({ current, year, scope, onScopeChange }: {
  current: ContinuingCurrent; year: string; scope: ContinuingSelection; onScopeChange: (scope: ContinuingSelection) => void;
}) {
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const selected = current.criterios.find(row => row.id === selectedId)
  const comments = current.comentarios.filter(row => row.texto_informativo)
  return <div className="space-y-6">
    <GlobalSectionSummary title="Educación Continua">
      <div className="flex flex-wrap gap-8">
        <div><p className="text-sm text-muted-foreground">Encuestas analizadas</p><p className="mt-1 text-3xl font-semibold">{current.encuestas_analizadas.toLocaleString("es-MX")}</p></div>
        <div><p className="text-sm text-muted-foreground">Respuestas a criterios</p><p className="mt-1 text-3xl font-semibold">{current.respuestas.toLocaleString("es-MX")}</p></div>
      </div>
      <p className="text-xs text-muted-foreground">Una encuesta puede aportar respuestas a varios criterios de Educación Continua.</p>
      <div className="flex flex-wrap gap-4">
        <label className="text-sm">Carrera<select className="mt-1 block rounded-lg border bg-background p-2" value={scope.carrera} onChange={event => onScopeChange({ ...scope, carrera: event.target.value })}><option value="">Todas / Vista global</option>{current.opciones.carreras.map(name => <option key={name}>{name}</option>)}</select></label>
        <label className="text-sm">Modalidad<select className="mt-1 block rounded-lg border bg-background p-2" value={scope.modalidad_academica} onChange={event => onScopeChange({ ...scope, modalidad_academica: event.target.value })}><option value="">Todas las modalidades</option>{current.opciones.modalidades.map(name => <option key={name}>{name}</option>)}</select></label>
        {current.opciones.participacion.length > 0 && <label className="text-sm">Participación<select className="mt-1 block rounded-lg border bg-background p-2" value={scope.participacion} onChange={event => onScopeChange({ ...scope, participacion: event.target.value })}><option value="">Todas las respuestas</option>{current.opciones.participacion.map(value => <option key={value} value={value}>{value === "si" ? "Sí participó" : "No participó"}</option>)}</select></label>}
      </div>
      {Object.values(scope).some(Boolean) && <p className="text-xs text-muted-foreground">Población seleccionada: {scope.carrera || "Todas las carreras"} · {scope.modalidad_academica || "Todas las modalidades"} · {scope.participacion === "si" ? "Sí participó" : scope.participacion === "no" ? "No participó" : "Todas las respuestas"}</p>}
      <section aria-label="Participación" className="space-y-3 rounded-xl border bg-background p-4"><h4 className="font-semibold">Participación en Educación Continua</h4>
        <p className="text-xs text-muted-foreground">¿Has tomado algún curso, taller o diplomado de Educación Continua?</p>
        <p className="text-xs text-muted-foreground">Indicador descriptivo independiente · {current.participacion.total_valido.toLocaleString("es-MX")} respuestas válidas</p>
        <BinaryDistribution data={current.participacion} />
      </section>
      <section aria-label="Medios de difusión" className="space-y-3 rounded-xl border bg-background p-4"><h4 className="font-semibold">Medios por los que se enteran de cursos y diplomados</h4>
        <p className="text-xs text-muted-foreground">Distribución entre {current.medios.n.toLocaleString("es-MX")} respuestas válidas</p>
        {current.medios.n ? current.medios.valores.map(row => <div key={row.nombre} className="space-y-2 text-sm">
          <div className="flex justify-between gap-3"><span>{row.nombre}</span><span>{row.n.toLocaleString("es-MX")} · {row.porcentaje.toFixed(2)}%</span></div>
          <div aria-hidden="true" className="h-2 overflow-hidden rounded-full bg-muted"><div className="h-full rounded-full bg-primary" style={{ width: `${row.porcentaje}%` }} /></div>
        </div>) : <MetricState empty />}
      </section>
    </GlobalSectionSummary>
    <section aria-label="Indicadores especializados" className="space-y-4 rounded-2xl border bg-card p-6 shadow-sm"><h3 className="font-semibold">Indicadores de Educación Continua</h3>
      <div className="grid gap-4 lg:grid-cols-2">{current.criterios.map(criterion => <article key={criterion.id} className="space-y-4 rounded-xl border bg-background p-4">
        <CriterionMetric criterion={criterion} year={year} />
        <button type="button" className="text-sm font-medium text-primary hover:underline" onClick={() => setSelectedId(criterion.id)}>Ver detalle de {criterion.nombre}</button>
      </article>)}</div>
    </section>
    {selected && <div className="fixed inset-0 z-[130] flex justify-end bg-black/30">
      <button type="button" aria-label="Cerrar detalle del indicador" onClick={() => setSelectedId(null)} className="absolute inset-0" />
      <aside role="dialog" aria-modal="true" aria-label={selected.nombre} className="relative z-10 flex h-full w-full max-w-2xl flex-col border-l bg-background shadow-2xl">
        <div className="flex items-start justify-between gap-4 border-b p-6"><h3 className="text-xl font-semibold">{selected.nombre}</h3><button type="button" onClick={() => setSelectedId(null)} className="rounded-xl border px-3 py-2 text-sm">Cerrar</button></div>
        <div className="flex-1 space-y-7 overflow-y-auto p-6"><CriterionMetric criterion={selected} year={year} />
          <p className="text-xs text-muted-foreground">Encuestas con respuesta válida al indicador: {selected.encuestas_validas.toLocaleString("es-MX")}</p>
          <section className="space-y-3"><h4 className="font-semibold">¿Qué podríamos hacer para mejorar Educación Continua?</h4><p className="text-xs text-muted-foreground">Análisis cualitativo con SIA · En construcción</p>
            {comments.length ? comments.map((row, i) => <p key={`${row.fila_origen}-${i}`} className="rounded-xl border p-4 text-sm leading-6">{row.texto}</p>) : <p className="text-sm text-muted-foreground">No hay comentarios para la población seleccionada.</p>}
          </section>
        </div>
      </aside>
    </div>}
  </div>
}

export function ContinuingEducation({ periodo, modalidad, servicio }: { periodo: string; modalidad: string; servicio: string }) {
  const [selection, setSelection] = useState<ContinuingSelection>({ carrera: "", modalidad_academica: "", participacion: "" })
  const key = JSON.stringify({ periodo, modalidad, servicio, ...selection })
  const [result, setResult] = useState<{ key: string; data: ContinuingCurrent | null; error: string | null } | null>(null)
  useEffect(() => {
    const controller = new AbortController()
    let active = true
    const scope: ContinuingScope = JSON.parse(key)
    fetchContinuingEducation(scope, controller.signal).then(data => { if (active) setResult({ key, data, error: null }) })
      .catch(error => { if (active) setResult({ key, data: null, error: String(error.message) }) })
    return () => { active = false; controller.abort() }
  }, [key])
  if (result?.key !== key) return <MetricSkeleton />
  if (result.error || !result.data) return <p role="alert" className="text-sm text-red-700">{result.error}</p>
  return <ContinuingEducationView current={result.data} year={periodo} scope={selection} onScopeChange={setSelection} />
}
