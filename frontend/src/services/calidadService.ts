export type CalidadSection = {
  seccion_id: string
  seccion_nombre: string
  indice: number | null
  preguntas_con_datos: number
}

export type CalidadSummary = {
  encuestas: number
  respuestas_normalizadas: number
  respuestas_kpi_validas: number
  preguntas_con_datos: number
  indice_general: number | null
  secciones_general: number
  secciones: CalidadSection[]
  qa_total: number
  fila_inicio: number
  fila_fin: number
  fuente_id: string
  periodo_id: string
  version_instrumento: string
}

const API_BASE_URL = "http://127.0.0.1:8000"

export async function getCalidadSummary(): Promise<CalidadSummary> {
  const response = await fetch(
    `${API_BASE_URL}/api/encuesta-calidad/preview/resumen`,
  )

  if (!response.ok) {
    throw new Error(
      "No fue posible obtener el resumen de Encuesta de Calidad.",
    )
  }

  return response.json()
}
