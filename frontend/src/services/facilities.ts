import type { CalidadComment } from "./calidadService"
import type { ServiceDistribution, ServiceScope } from "./institutionalServices"
export type { ServiceScope }

export type FacilitiesCriterion = ServiceDistribution & {
  id: string
  pregunta_id: string
  nombre: string
  indice: number | null
  n: number
  encuestas_validas: number
}
export type FacilitiesCurrent = {
  indice: number | null
  respuestas: number
  encuestas_analizadas: number
  distribucion: ServiceDistribution
  criterios: FacilitiesCriterion[]
  comentarios: CalidadComment[]
  opciones: { carreras: string[]; modalidades: string[] }
  poblacion: { carrera: string | null; modalidad_academica: string | null }
}
export type FacilitiesHistory = {
  indice_2025: number | null
  indice_2026: number | null
  ids_comparables: string[]
  criterios: { id: string; indice_2025: number | null; indice_2026: number | null }[]
  poblacion: FacilitiesCurrent["poblacion"]
}

export async function fetchFacilities<T>(kind: "actual" | "historico", scope: ServiceScope, signal?: AbortSignal): Promise<T> {
  const params = new URLSearchParams()
  for (const [key, value] of Object.entries(scope)) if (value && (kind === "actual" || key !== "periodo")) params.set(key, value)
  const response = await fetch(`http://127.0.0.1:8000/api/encuesta-calidad/instalaciones/${kind}?${params}`, { signal })
  if (!response.ok) throw new Error(kind === "historico" ? "No fue posible cargar el comparativo histórico" : "No fue posible cargar Instalaciones y equipo tecnológico")
  return response.json() as Promise<T>
}
