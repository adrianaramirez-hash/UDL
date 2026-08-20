export type ObservationSummary = {
  observaciones: number
  consolidado: number
  en_proceso: number
  no_consolidado: number
  promedio: number
  casos_criticos: number
}

export type ObservationFilters = {
  corte?: string
  servicio?: string
  tipo?: string
}

export type ObservationFilterOptions = {
  cortes: string[]
  servicios: string[]
  tipos: string[]
}

const API_BASE_URL = "http://127.0.0.1:8000"

export async function getObservationSummary(
  filters: ObservationFilters = {},
): Promise<ObservationSummary> {
  const params = new URLSearchParams()

  if (
    filters.corte &&
    filters.corte !== "Todos los cortes"
  ) {
    params.set("corte", filters.corte)
  }

  if (
    filters.servicio &&
    filters.servicio !== "Todos los servicios"
  ) {
    params.set("servicio", filters.servicio)
  }

  if (
    filters.tipo &&
    filters.tipo !== "Todos los tipos"
  ) {
    params.set("tipo", filters.tipo)
  }

  const query = params.toString()

  const response = await fetch(
    `${API_BASE_URL}/api/observacion/resumen${
      query ? `?${query}` : ""
    }`,
  )

  if (!response.ok) {
    throw new Error(
      "No fue posible obtener el resumen de observación.",
    )
  }

  return response.json()
}

export async function getObservationFilters(): Promise<ObservationFilterOptions> {
  const response = await fetch(
    `${API_BASE_URL}/api/observacion/filtros`,
  )

  if (!response.ok) {
    throw new Error(
      "No fue posible obtener los filtros de observación.",
    )
  }

  return response.json()
}