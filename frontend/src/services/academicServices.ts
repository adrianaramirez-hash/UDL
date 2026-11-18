import type { CalidadComment } from "./calidadService"
import type { ServiceDistribution, ServiceScope } from "./institutionalServices"
export type { ServiceScope }

export type AcademicCriterion = ServiceDistribution & {
  id: string
  pregunta_id: string
  nombre: string
  indice: number | null
  n: number
  encuestas_validas: number
}
export type AcademicCurrent = {
  indice: number | null
  respuestas: number
  encuestas_analizadas: number
  distribucion: ServiceDistribution
  criterios: AcademicCriterion[]
  comentarios: CalidadComment[]
  opciones: { carreras: string[]; modalidades: string[] }
  poblacion: { carrera: string | null; modalidad_academica: string | null }
}
export type AcademicHistory = {
  indice_2025: number | null
  indice_2026: number | null
  ids_comparables: string[]
  criterios: { id: string; indice_2025: number | null; indice_2026: number | null }[]
  poblacion: AcademicCurrent["poblacion"]
}

export async function fetchAcademicServices<T>(kind: "actual" | "historico", scope: ServiceScope, signal?: AbortSignal): Promise<T> {
  const params = new URLSearchParams()
  for (const [key, value] of Object.entries(scope)) if (value && (kind === "actual" || key !== "periodo")) params.set(key, value)
  const response = await fetch(`/api/encuesta-calidad/servicios-academicos/${kind}?${params}`, { signal })
  if (!response.ok) throw new Error(kind === "historico" ? "No fue posible cargar el comparativo histórico" : "No fue posible cargar Servicios académicos")
  return response.json() as Promise<T>
}
