import { useEffect, useState } from "react"
import { GlobalSectionSummary, MetricComparison, MetricDistribution, MetricSkeleton, MetricState } from "@/components/common/CalidadMetrics"
import { fetchLanguageCenter } from "@/services/languageCenter"
import type { CategoryDistribution, LanguageCriterion, LanguageCurrent, LanguageScope, LanguageSelection } from "@/services/languageCenter"

function CriterionMetric({ criterion, year }: { criterion: LanguageCriterion; year: string }) {
  return <div className="space-y-4"><h4 className="font-semibold">{criterion.nombre}</h4>
    <p className="text-sm">Resultado actual: <strong>{criterion.indice?.toFixed(2) ?? "—"}</strong> · {criterion.n.toLocaleString("es-MX")} respuestas a criterios</p>
    <MetricComparison previous={null} current={criterion.indice} currentYear={year} showHistoricalState={false} />
    {criterion.indice !== null && <MetricDistribution favorable={criterion.favorable_pct} neutral={criterion.neutral_pct} unfavorable={criterion.desfavorable_pct} />}
  </div>
}

function ContextDistribution({ title, data }: { title: string; data: CategoryDistribution }) {
  return <section aria-label={title} className="space-y-3 rounded-xl border bg-background p-4"><h4 className="font-semibold">{title}</h4>
    <p className="text-xs text-muted-foreground">{data.n.toLocaleString("es-MX")} respuestas válidas · distribución entre quienes respondieron</p>
    {data.n ? data.valores.map(row => <div key={row.nombre} className="space-y-2 text-sm">
      <div className="flex justify-between gap-3"><span>{row.nombre}</span><span>{row.n.toLocaleString("es-MX")} · {row.porcentaje.toFixed(2)}%</span></div>
      <div aria-hidden="true" className="h-2 overflow-hidden rounded-full bg-muted"><div className="h-full rounded-full bg-primary" style={{ width: `${row.porcentaje}%` }} /></div>
    </div>) : <MetricState empty />}
  </section>
}

export function LanguageCenterView({ current, year, scope, onScopeChange }: {
  current: LanguageCurrent; year: string; scope: LanguageSelection; onScopeChange: (scope: LanguageSelection) => void;
}) {
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const selected = current.criterios.find(row => row.id === selectedId)
  const comments = current.comentarios.filter(row => row.texto_informativo)
  const select = (field: keyof LanguageSelection, title: string, globalLabel: string, values: string[]) => <label className="text-sm">{title}<select className="mt-1 block rounded-lg border bg-background p-2" value={scope[field]} onChange={event => onScopeChange({ ...scope, [field]: event.target.value })}><option value="">{globalLabel}</option>{values.map(name => <option key={name}>{name}</option>)}</select></label>
  return <div className="space-y-6">
    <GlobalSectionSummary title="Centro de Idiomas">
      <div className="flex flex-wrap gap-8">
        <div><p className="text-sm text-muted-foreground">Encuestas analizadas</p><p className="mt-1 text-3xl font-semibold">{current.encuestas_analizadas.toLocaleString("es-MX")}</p></div>
        <div><p className="text-sm text-muted-foreground">Respuestas a criterios</p><p className="mt-1 text-3xl font-semibold">{current.respuestas.toLocaleString("es-MX")}</p></div>
      </div>
      <p className="text-xs text-muted-foreground">Una encuesta puede aportar respuestas a varios criterios de Centro de Idiomas.</p>
      <div className="flex flex-wrap gap-4">
        {select("carrera", "Carrera", "Todas / Vista global", current.opciones.carreras)}
        {select("modalidad_academica", "Modalidad", "Todas las modalidades", current.opciones.modalidades)}
        {current.opciones.idiomas.length > 0 && select("idioma", "Idioma", "Todos los idiomas", current.opciones.idiomas)}
        {current.opciones.tipos_curso.length > 0 && select("tipo_curso", "Tipo de curso", "Todos los tipos de curso", current.opciones.tipos_curso)}
      </div>
      {Object.values(scope).some(Boolean) && <p className="text-xs text-muted-foreground">Población seleccionada: {scope.carrera || "Todas las carreras"} · {scope.modalidad_academica || "Todas las modalidades"} · {scope.idioma || "Todos los idiomas"} · {scope.tipo_curso || "Todos los tipos de curso"}</p>}
      <section aria-label="Uso del servicio" className="space-y-3 rounded-xl border bg-background p-4"><h4 className="font-semibold">Uso del servicio</h4>
        <p className="text-xs text-muted-foreground">Indicador descriptivo independiente · {current.uso.total_valido.toLocaleString("es-MX")} respuestas válidas</p>
        {current.uso.si_pct === null ? <MetricState empty /> : <>
          <div aria-hidden="true" className="flex h-2 overflow-hidden rounded-full bg-muted"><div className="bg-primary" style={{ width: `${current.uso.si_pct}%` }} /><div className="bg-slate-300" style={{ width: `${current.uso.no_pct ?? 0}%` }} /></div>
          <div className="flex flex-wrap gap-4 text-sm"><span>Sí: {current.uso.si_pct.toFixed(2)}% ({current.uso.si.toLocaleString("es-MX")})</span><span>No: {current.uso.no_pct?.toFixed(2)}% ({current.uso.no.toLocaleString("es-MX")})</span></div>
        </>}
      </section>
      <div className="grid items-start gap-4 lg:grid-cols-2"><ContextDistribution title="Idiomas" data={current.idiomas} /><ContextDistribution title="Tipos de curso" data={current.tipos_curso} /></div>
    </GlobalSectionSummary>
    <section aria-label="Criterios de experiencia" className="space-y-4 rounded-2xl border bg-card p-6 shadow-sm"><h3 className="font-semibold">Criterios de experiencia</h3>
      <div className="grid gap-4 lg:grid-cols-2">{current.criterios.map(criterion => <article key={criterion.id} className="space-y-4 rounded-xl border bg-background p-4">
        <CriterionMetric criterion={criterion} year={year} />
        <button type="button" className="text-sm font-medium text-primary hover:underline" onClick={() => setSelectedId(criterion.id)}>Ver detalle de {criterion.nombre}</button>
      </article>)}</div>
    </section>
    {selected && <div className="fixed inset-0 z-[130] flex justify-end bg-black/30">
      <button type="button" aria-label="Cerrar detalle del criterio" onClick={() => setSelectedId(null)} className="absolute inset-0" />
      <aside role="dialog" aria-modal="true" aria-label={selected.nombre} className="relative z-10 flex h-full w-full max-w-2xl flex-col border-l bg-background shadow-2xl">
        <div className="flex items-start justify-between gap-4 border-b p-6"><h3 className="text-xl font-semibold">{selected.nombre}</h3><button type="button" onClick={() => setSelectedId(null)} className="rounded-xl border px-3 py-2 text-sm">Cerrar</button></div>
        <div className="flex-1 space-y-7 overflow-y-auto p-6"><CriterionMetric criterion={selected} year={year} />
          <p className="text-xs text-muted-foreground">Encuestas con respuesta válida al criterio: {selected.encuestas_validas.toLocaleString("es-MX")}</p>
          <section className="space-y-3"><h4 className="font-semibold">Profesor actual · Dato descriptivo</h4>
            <p className="text-xs text-muted-foreground">{current.profesores.n.toLocaleString("es-MX")} respuestas. Nombres informados en la población seleccionada, en orden alfabético.</p>
            {current.profesores.nombres.length ? <ul className="space-y-1 text-sm">{current.profesores.nombres.map(name => <li key={name}>{name}</li>)}</ul> : <MetricState empty />}
          </section>
          <section className="space-y-4"><h4 className="font-semibold">Comentarios de Centro de Idiomas</h4><p className="text-xs text-muted-foreground">Análisis cualitativo con SIA · En construcción</p>
            {current.preguntas_comentarios.map(question => <div key={question.pregunta_id} className="space-y-3"><p className="text-sm font-medium">{question.nombre}</p>
              {comments.filter(row => row.pregunta_id === question.pregunta_id).length ? comments.filter(row => row.pregunta_id === question.pregunta_id).map((row, i) => <p key={`${row.fila_origen}-${i}`} className="rounded-xl border p-4 text-sm leading-6">{row.texto}</p>) : <p className="text-sm text-muted-foreground">No hay comentarios para esta pregunta bajo los filtros activos.</p>}
            </div>)}
          </section>
        </div>
      </aside>
    </div>}
  </div>
}

export function LanguageCenter({ periodo, modalidad, servicio }: { periodo: string; modalidad: string; servicio: string }) {
  const [selection, setSelection] = useState<LanguageSelection>({ carrera: "", modalidad_academica: "", idioma: "", tipo_curso: "" })
  const key = JSON.stringify({ periodo, modalidad, servicio, ...selection })
  const [result, setResult] = useState<{ key: string; data: LanguageCurrent | null; error: string | null } | null>(null)
  useEffect(() => {
    const controller = new AbortController()
    let active = true
    const scope: LanguageScope = JSON.parse(key)
    fetchLanguageCenter(scope, controller.signal).then(data => { if (active) setResult({ key, data, error: null }) })
      .catch(error => { if (active) setResult({ key, data: null, error: String(error.message) }) })
    return () => { active = false; controller.abort() }
  }, [key])
  if (result?.key !== key) return <MetricSkeleton />
  if (result.error || !result.data) return <p role="alert" className="text-sm text-red-700">{result.error}</p>
  return <LanguageCenterView current={result.data} year={periodo} scope={selection} onScopeChange={setSelection} />
}
