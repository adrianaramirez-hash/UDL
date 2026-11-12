import type { CalidadFilterSelection } from "./calidadService"

export type ResultsReportScope = CalidadFilterSelection & { carrera?: string; modalidad_academica?: string; idioma?: string; tipo_curso?: string; participacion?: string }

export async function downloadResultsReportPdf(scope: ResultsReportScope, activeView = false): Promise<void> {
  // Read the controls already on screen; never change a control or dashboard state.
  const selection = { ...scope }
  if (activeView) {
    const fields: Record<string, "carrera" | "modalidad_academica" | "idioma" | "tipo_curso" | "participacion"> = { Carrera: "carrera", Modalidad: "modalidad_academica", Idioma: "idioma", "Tipo de curso": "tipo_curso", Participación: "participacion" }
    for (const label of document.querySelectorAll("section label")) {
      const select = label.querySelector("select")
      if (!select) continue
      for (const [title, key] of Object.entries(fields)) if (label.textContent?.trim().startsWith(title)) selection[key] = select.value
    }
  }
  const params = new URLSearchParams()
  for (const key of ["periodo", "modalidad", "servicio", "carrera", "modalidad_academica", "idioma", "tipo_curso", "participacion"] as const) {
    if (selection[key]) params.set(key, selection[key])
  }
  const response = await fetch(`http://127.0.0.1:8000/api/encuesta-calidad/informe-resultados.pdf?${params}`)
  if (!response.ok) {
    const error = await response.json().catch(() => null)
    throw new Error(error?.detail || "No fue posible generar el informe de resultados.")
  }
  if (!response.headers.get("Content-Type")?.includes("application/pdf")) throw new Error("La respuesta recibida no es un PDF.")
  const blob = await response.blob()
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement("a")
  anchor.href = url
  anchor.download = response.headers.get("Content-Disposition")?.match(/filename="([^"]+)"/)?.[1] || "Informe_de_resultados.pdf"
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
