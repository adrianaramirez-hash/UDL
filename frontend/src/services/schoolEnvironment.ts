import type { CalidadComment } from "./calidadService"
import type { ServiceDistribution, ServiceScope } from "./institutionalServices"
export type { ServiceScope }

export type SchoolIncidence = {
  si: number
  no: number
  total_valido: number
  si_pct: number | null
  no_pct: number | null
}

export type SchoolIncidenceHistory = {
  comparable: boolean
  porcentaje_2025: number | null
  porcentaje_2026: number | null
  delta_pp: number | null
}

export type SchoolEnvironmentCriterion = ServiceDistribution & {
  id: string
  pregunta_id: string
  nombre: string
  indice: number | null
  n: number
  encuestas_validas: number
}
export type SchoolEnvironmentCurrent = {
  incidencia: SchoolIncidence
  indice: number | null
  respuestas: number
  encuestas_analizadas: number
  distribucion: ServiceDistribution
  criterios: SchoolEnvironmentCriterion[]
  comentarios: CalidadComment[]
  opciones: { carreras: string[]; modalidades: string[] }
  poblacion: { carrera: string | null; modalidad_academica: string | null }
}
export type SchoolEnvironmentHistory = {
  incidencia: SchoolIncidenceHistory
  indice_2025: number | null
  indice_2026: number | null
  ids_comparables: string[]
  criterios: { id: string; indice_2025: number | null; indice_2026: number | null }[]
  poblacion: SchoolEnvironmentCurrent["poblacion"]
}

export async function fetchSchoolEnvironment<T>(kind: "actual" | "historico", scope: ServiceScope, signal?: AbortSignal): Promise<T> {
  const params = new URLSearchParams()
  for (const [key, value] of Object.entries(scope)) if (value && (kind === "actual" || key !== "periodo")) params.set(key, value)
  const response = await fetch(`/api/encuesta-calidad/ambiente-escolar/${kind}?${params}`, { signal })
  if (!response.ok) throw new Error(kind === "historico" ? "No fue posible cargar el comparativo histórico" : "No fue posible cargar Ambiente escolar")
  return response.json() as Promise<T>
}
