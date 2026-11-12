import type { CalidadComment } from "./calidadService"
import type { ServiceScope, ServiceDistribution } from "./institutionalServices"

export type LanguageScope = ServiceScope & { idioma: string; tipo_curso: string }
export type LanguageSelection = Pick<LanguageScope, "carrera" | "modalidad_academica" | "idioma" | "tipo_curso">
export type LanguageCriterion = ServiceDistribution & {
  id: string; nombre: string; pregunta_id: string; indice: number | null; n: number;
  encuestas_validas: number; historico_disponible: false;
}
export type CategoryDistribution = { n: number; valores: { nombre: string; n: number; porcentaje: number }[] }
export type LanguageCurrent = {
  criterios: LanguageCriterion[]; respuestas: number; encuestas_analizadas: number;
  uso: { si: number; no: number; total_valido: number; si_pct: number | null; no_pct: number | null };
  idiomas: CategoryDistribution; tipos_curso: CategoryDistribution;
  profesores: { n: number; nombres: string[] };
  comentarios: CalidadComment[];
  preguntas_comentarios: { pregunta_id: string; nombre: string }[];
  opciones: { carreras: string[]; modalidades: string[]; idiomas: string[]; tipos_curso: string[] };
  poblacion: { carrera: string | null; modalidad_academica: string | null; idioma: string | null; tipo_curso: string | null };
}

export async function fetchLanguageCenter(scope: LanguageScope, signal?: AbortSignal): Promise<LanguageCurrent> {
  const params = new URLSearchParams()
  for (const [key, value] of Object.entries(scope)) if (value) params.set(key, value)
  const response = await fetch(`http://127.0.0.1:8000/api/encuesta-calidad/centro-idiomas/actual?${params}`, { signal })
  if (!response.ok) {
    const body = await response.json().catch(() => null)
    throw new Error(typeof body?.detail === "string" ? body.detail : "No fue posible cargar Centro de Idiomas")
  }
  return response.json() as Promise<LanguageCurrent>
}
