import { useEffect, useState } from "react"
import { getCalidadSummary } from "@/services/calidadService"
import { GlobalSectionSummary, MetricCard, MetricDelta, MetricSkeleton, MetricState } from "@/components/common/CalidadMetrics"
import { criteriaFor, currentPopulations, criterionId, populationKey, populationMetric } from "./directionModel"
import type { DirectionData, DirectionPopulation } from "./directionModel"

// Strict filter scope: no fallback to another modality/program or an unfiltered year.
export function useDirectionHistory(enabled: boolean, modalidad = "", servicio = "") {
  const scope = JSON.stringify([modalidad, servicio])
  const [result, setResult] = useState<{ scope: string; data: DirectionData | null; loading: boolean; error: boolean } | null>(null)
  useEffect(() => {
    if (!enabled) return
    let active = true
    const controller = new AbortController()
    getCalidadSummary({ periodo: "2025", modalidad, servicio, historico: true }, controller.signal).then(data => {
      if (active) setResult({ scope, data, loading: false, error: false })
    }).catch(() => {
      if (active) setResult({ scope, data: null, loading: false, error: true })
    })
    return () => { active = false; controller.abort() }
  }, [enabled, modalidad, servicio, scope])
  if (!enabled) return { data: null, loading: false, error: false }
  return result?.scope === scope ? result : { data: null, loading: true, error: false }
}

export function DirectionCriteria({ current, historical, population, year, loading = false, error = false }: {
  current: DirectionData; historical: DirectionData | null; population?: DirectionPopulation; year: string; loading?: boolean; error?: boolean;
}) {
  const previous = criteriaFor(year === "2026" ? historical : null, population)
  return <div className="grid gap-4 md:grid-cols-2">{criteriaFor(current, population).map((question, i) => <MetricCard
    key={question.pregunta_id} title={question.texto_corto} current={question.indice} previous={previous[i]!.indice}
    responses={question.n} currentYear={year} distribution={question} loading={loading && year === "2026"} comparisonError={error && year === "2026"}
  />)}</div>
}

export function DirectionGlobal({ current, historical, year, loading, error, filtered, assigned = false }: {
  current: DirectionData; historical: DirectionData | null; year: string; loading: boolean; error: boolean; filtered: boolean; assigned?: boolean;
}) {
  const questions = criteriaFor(current)
  const section = current.secciones.find(row => criterionId(row.seccion_id) === "ESC_02_DIRECCION")
  const previous = year === "2026" ? historical?.secciones.find(row => criterionId(row.seccion_id) === "ESC_02_DIRECCION")?.indice ?? null : null
  return <GlobalSectionSummary title="Dirección de carrera">
    {assigned ? <p className="text-sm text-muted-foreground">Resultados de los servicios asignados.</p>
      : filtered && <p className="text-sm text-muted-foreground">Vista global dentro del programa seleccionado en los filtros superiores.</p>}
    <MetricCard title="Resultado general de Dirección" current={section?.indice ?? null} previous={previous} responses={questions.reduce((sum, row) => sum + row.n, 0)} currentYear={year} loading={loading && year === "2026"} comparisonError={error && year === "2026"} />
    <DirectionCriteria current={current} historical={historical} year={year} loading={loading} error={error} />
  </GlobalSectionSummary>
}

export function DirectionCareerList({ current, historical, year, onSelect, loading = false, error = false, assigned = false }: {
  current: DirectionData; historical: DirectionData | null; year: string; onSelect: (population: DirectionPopulation) => void; loading?: boolean; error?: boolean; assigned?: boolean;
}) {
  const [career, setCareer] = useState("")
  const [modality, setModality] = useState("")
  const populations = currentPopulations(current)
  const careers = [...new Set(populations.map(row => row.carrera_base))]
  const modalities = [...new Set(populations.map(row => row.modalidad_academica))].filter(Boolean)
  // An obsolete local selection means global when the source changes.
  const selectedCareer = careers.includes(career) ? career : ""
  const selectedModality = modalities.includes(modality) ? modality : ""
  const visible = populations.filter(row => (!selectedCareer || row.carrera_base === selectedCareer) && (!selectedModality || row.modalidad_academica === selectedModality))
  const allLabel = assigned ? "Toda mi asignación" : "Todas / Vista global"
  return <section aria-label="Carreras de Dirección" className="space-y-5 rounded-2xl border bg-card p-6 shadow-sm">
    <div><h3 className="font-semibold">Desglose opcional por carrera y modalidad</h3>
      {assigned ? <p className="mt-1 text-sm text-muted-foreground">Abre una población asignada para consultar su resultado.</p>
        : <><p className="mt-1 text-sm text-muted-foreground">Todas las carreras del catálogo actual, incluidas las que aún no tienen respuestas de Dirección.</p><p className="mt-1 text-xs text-muted-foreground">Estos filtros acotan el listado. El resumen superior conserva su alcance global; abre una población para consultar su resultado.</p></>}
    </div>
    <div className="flex flex-wrap gap-4">
      <label className="space-y-1 text-sm">Carrera / programa<select className="block rounded-lg border bg-background p-2" value={selectedCareer} onChange={event => setCareer(event.target.value)}><option value="">{allLabel}</option>{careers.map(name => <option key={name}>{name}</option>)}</select></label>
      <label className="space-y-1 text-sm">Modalidad<select className="block rounded-lg border bg-background p-2" value={selectedModality} onChange={event => setModality(event.target.value)}><option value="">{allLabel}</option>{modalities.map(name => <option key={name}>{name}</option>)}</select></label>
    </div>
    <p className="text-xs text-muted-foreground">{visible.length} poblaciones · {new Set(visible.map(row => row.carrera_base)).size} carreras / programas</p>
    <div className="grid max-h-[520px] gap-3 overflow-y-auto pr-2 lg:grid-cols-2">{visible.map(row => {
      const previous = year === "2026" && !loading && !error ? populationMetric(historical, row)?.indice ?? null : null
      return <button key={populationKey(row)} type="button" onClick={() => onSelect(row)} className="space-y-3 rounded-xl border bg-background p-4 text-left transition hover:border-primary/40">
        <div className="flex flex-wrap items-baseline justify-between gap-2"><h4 className="font-semibold">{row.carrera_base}</h4><MetricDelta previous={previous} current={row.indice} /></div>
        <p className="text-xs text-muted-foreground">{row.modalidad_academica || "Modalidad no informada"}</p>
        {row.conflict ? <p className="text-sm text-amber-800">Identidad ambigua: revisar origen</p> : row.indice === null || row.encuestas === 0 ? <MetricState empty /> : <><p className="text-xl font-semibold">{row.indice.toFixed(2)}</p><p className="text-xs text-muted-foreground">{row.encuestas} encuestas con respuestas de Dirección</p></>}
        {loading ? <MetricSkeleton /> : error ? <p className="text-xs text-muted-foreground">No fue posible cargar el comparativo histórico</p> : previous !== null ? <p className="text-xs text-muted-foreground">2025 · {previous.toFixed(2)}</p> : row.indice !== null && <MetricState />}
        <p className="text-xs font-medium text-primary">Ver cuatro criterios →</p>
      </button>
    })}</div>
    {visible.length === 0 && <p className="text-sm text-muted-foreground">No hay carreras para esta combinación de filtros.</p>}
  </section>
}
