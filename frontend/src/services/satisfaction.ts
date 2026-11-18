import type { CalidadComment } from "./calidadService"
import type { ServiceDistribution, ServiceScope } from "./institutionalServices"
export type { ServiceScope }

export type SatisfactionIndicator = ServiceDistribution & {
  id: string; nombre: string; pregunta_id: string; escala_id: string | null;
  indice: number | null; n: number; encuestas_validas: number;
}
export type RecommendationIndicator = {
  id: string; nombre: string; pregunta_id: string; escala_id: string | null;
  si: number; no: number; total_valido: number; encuestas_validas: number;
  si_pct: number | null; no_pct: number | null;
}
export type SatisfactionCurrent = {
  satisfaccion: SatisfactionIndicator;
  recomendacion: RecommendationIndicator;
  encuestas_analizadas: number;
  comentarios: CalidadComment[];
  opciones: { carreras: string[]; modalidades: string[] };
  poblacion: { carrera: string | null; modalidad_academica: string | null };
}
export type SatisfactionHistory = {
  satisfaccion: { id: string; comparable: boolean; indice_2025: number | null; indice_2026: number | null; delta: number | null };
  recomendacion: { id: string; comparable: boolean; porcentaje_2025: number | null; porcentaje_2026: number | null; delta_pp: number | null };
  poblacion: SatisfactionCurrent["poblacion"];
}

export async function fetchSatisfaction<T>(kind: "actual" | "historico", scope: ServiceScope, signal?: AbortSignal): Promise<T> {
  const params = new URLSearchParams()
  for (const [key, value] of Object.entries(scope)) if (value && (kind === "actual" || key !== "periodo")) params.set(key, value)
  const response = await fetch(`/api/encuesta-calidad/satisfaccion-recomendacion/${kind}?${params}`, { signal })
  if (!response.ok) throw new Error(kind === "historico" ? "No fue posible cargar el comparativo histórico" : "No fue posible cargar Satisfacción y recomendación")
  return response.json() as Promise<T>
}
