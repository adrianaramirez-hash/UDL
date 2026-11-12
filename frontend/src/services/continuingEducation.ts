import type { CalidadComment } from "./calidadService"
import type { ServiceScope, ServiceDistribution } from "./institutionalServices"

export type ContinuingScope = ServiceScope & { participacion: string }
export type ContinuingSelection = Pick<ContinuingScope, "carrera" | "modalidad_academica" | "participacion">
export type ContinuingBinary = { si: number; no: number; total_valido: number; si_pct: number | null; no_pct: number | null }
export type ContinuingCriterion = ServiceDistribution & {
  id: string; nombre: string; pregunta_id: string; tipo_dato: "LIKERT" | "BINARIA"; escala_id: string;
  resultado: number | null; unidad: "indice" | "porcentaje"; n: number;
  encuestas_validas: number; historico_disponible: false; binaria: ContinuingBinary | null;
}
export type ContinuingCurrent = {
  criterios: ContinuingCriterion[]; respuestas: number; encuestas_analizadas: number;
  participacion: ContinuingBinary;
  medios: { n: number; valores: { nombre: string; n: number; porcentaje: number }[] };
  comentarios: CalidadComment[];
  opciones: { carreras: string[]; modalidades: string[]; participacion: string[] };
  poblacion: { carrera: string | null; modalidad_academica: string | null; participacion: string | null };
}

export async function fetchContinuingEducation(scope: ContinuingScope, signal?: AbortSignal): Promise<ContinuingCurrent> {
  const params = new URLSearchParams()
  for (const [key, value] of Object.entries(scope)) if (value) params.set(key, value)
  const response = await fetch(`http://127.0.0.1:8000/api/encuesta-calidad/educacion-continua/actual?${params}`, { signal })
  if (!response.ok) {
    const body = await response.json().catch(() => null)
    throw new Error(typeof body?.detail === "string" ? body.detail : "No fue posible cargar Educación Continua")
  }
  return response.json() as Promise<ContinuingCurrent>
}
