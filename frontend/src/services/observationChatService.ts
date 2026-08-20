const API_BASE_URL = "http://127.0.0.1:8000"

export type ObservationChatContext = {
  ultimo_docente?: string
  ultimo_servicio?: string
  ultima_materia?: string
  ultimo_corte?: string
  ultimo_tipo?: string
  ultima_clasificacion?: string
  [key: string]: unknown
}

export type ObservationChatResponse = {
  respuesta: string
  tipo_respuesta: string
  filtros: Record<string, unknown>
  datos: Record<string, unknown>
  contexto: ObservationChatContext
}

export async function askObservationAdvisor(
  pregunta: string,
  contexto: ObservationChatContext = {},
): Promise<ObservationChatResponse> {
  const response = await fetch(
    `${API_BASE_URL}/api/observacion/chat`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        pregunta,
        contexto,
      }),
    },
  )

  if (!response.ok) {
    throw new Error(
      "No fue posible consultar al Asesor SIA.",
    )
  }

  return response.json()
}