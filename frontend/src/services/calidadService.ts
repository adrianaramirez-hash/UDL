export type CalidadSection = {
  seccion_id: string
  seccion_nombre: string
  indice: number | null
  preguntas_con_datos: number
}

export type CalidadFilterSelection = {
  periodo?: string
  modalidad?: string
  servicio?: string
}

export type CalidadFilters = {
  periodos: string[]
  modalidades: string[]
  servicios: string[]
  seleccion: {
    periodo: string | null
    modalidad: string | null
    servicio: string | null
  }
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
  filtros: CalidadFilters
}

const API_BASE_URL = "http://127.0.0.1:8000"

export async function getCalidadSummary(
  filtros: CalidadFilterSelection = {},
): Promise<CalidadSummary> {
  const params = new URLSearchParams()

  if (filtros.periodo) {
    params.set("periodo", filtros.periodo)
  }

  if (filtros.modalidad) {
    params.set("modalidad", filtros.modalidad)
  }

  if (filtros.servicio) {
    params.set("servicio", filtros.servicio)
  }

  const query = params.toString()

  const url = query
    ? `${API_BASE_URL}/api/encuesta-calidad/preview/resumen?${query}`
    : `${API_BASE_URL}/api/encuesta-calidad/preview/resumen`

  const response = await fetch(url)

  if (!response.ok) {
    throw new Error(
      "No fue posible obtener el resumen de Encuesta de Calidad.",
    )
  }

  return response.json()
}
