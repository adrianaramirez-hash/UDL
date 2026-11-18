import type { CalidadComment } from "./calidadService"

export type ServiceDistribution = {
  favorable_pct: number | null
  neutral_pct: number | null
  desfavorable_pct: number | null
}
export type InstitutionalService = ServiceDistribution & {
  id: string
  pregunta_id: string
  nombre: string
  indice: number | null
  n: number
}
export type ServiceScope = {
  periodo: string
  modalidad: string
  servicio: string
  carrera: string
  modalidad_academica: string
}
export type InstitutionalCurrent = {
  indice: number | null
  respuestas: number
  encuestas_con_evaluacion: number
  distribucion: ServiceDistribution
  servicios: InstitutionalService[]
  comentarios: CalidadComment[]
  opciones: { carreras: string[]; modalidades: string[] }
  poblacion: { carrera: string | null; modalidad_academica: string | null }
}
export type InstitutionalHistory = {
  indice_2025: number | null
  indice_2026: number | null
  criterios: string[]
  servicios: { id: string; indice_2025: number | null; indice_2026: number | null }[]
  poblacion: InstitutionalCurrent["poblacion"]
}

export async function fetchInstitutionalServices<T>(kind: "actual" | "historico", scope: ServiceScope, signal?: AbortSignal): Promise<T> {
  const params = new URLSearchParams()
  for (const [key, value] of Object.entries(scope)) if (value && (kind === "actual" || key !== "periodo")) params.set(key, value)
  const response = await fetch(`/api/encuesta-calidad/servicios-institucionales/${kind}?${params}`, { signal })
  if (!response.ok) throw new Error(kind === "historico" ? "No fue posible cargar el comparativo histórico" : "No fue posible cargar Servicios institucionales")
  return response.json() as Promise<T>
}
