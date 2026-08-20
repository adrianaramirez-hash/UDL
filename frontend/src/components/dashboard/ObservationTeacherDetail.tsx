import { useState } from "react"

import { ObservationTeacherSnapshot } from "@/components/dashboard/ObservationTeacherSnapshot"
import { ObservationTeacherTimeline } from "@/components/dashboard/ObservationTeacherTimeline"

type TeacherHistoryItem = {
  corte: string
  servicio: string
  grupo: string
  asignatura: string
  tipo: string
  puntaje: number
  clasificacion: string
}

type TeacherDetail = {
  docente: string
  observaciones: number
  promedio: number
  clasificacion: string
  historial: TeacherHistoryItem[]
  fortalezas: string[]
  areas_oportunidad: string[]
  recomendaciones: string[]
}

type ObservationTeacherDetailProps = {
  teacher: TeacherDetail
}

type TabKey = "resumen" | "historial" | "comentarios"

const API_BASE_URL = "http://127.0.0.1:8000"

const tabs: { key: TabKey; label: string }[] = [
  {
    key: "resumen",
    label: "Resumen",
  },
  {
    key: "historial",
    label: "Historial",
  },
  {
    key: "comentarios",
    label: "Comentarios",
  },
]

export function ObservationTeacherDetail({
  teacher,
}: ObservationTeacherDetailProps) {
  const [activeTab, setActiveTab] = useState<TabKey>("resumen")
  const [exporting, setExporting] = useState(false)

  async function handleExportPdf() {
    try {
      setExporting(true)

      const response = await fetch(
        `${API_BASE_URL}/api/observacion/docente/${encodeURIComponent(
          teacher.docente,
        )}/pdf`,
      )

      if (!response.ok) {
        throw new Error(
          "No fue posible generar el reporte PDF.",
        )
      }

      const blob = await response.blob()

      const url = window.URL.createObjectURL(blob)

      const nombreLimpio = teacher.docente
        .trim()
        .replace(/\s+/g, "_")

      const link = document.createElement("a")

      link.href = url
      link.download =
        `reporte_observacion_${nombreLimpio}.pdf`

      document.body.appendChild(link)

      link.click()

      document.body.removeChild(link)

      window.URL.revokeObjectURL(url)
    } catch (error) {
      console.error(error)

      window.alert(
        "No fue posible exportar el reporte PDF.",
      )
    } finally {
      setExporting(false)
    }
  }

  return (
    <div>
      <div className="border-b pb-5">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <p className="sia-eyebrow">
              DOCENTE
            </p>

            <h3 className="mt-2 text-3xl sia-heading">
              {teacher.docente}
            </h3>

            <p className="mt-2 text-sm text-muted-foreground">
              {teacher.observaciones} observación
              {teacher.observaciones === 1 ? "" : "es"} registrada
              {teacher.observaciones === 1 ? "" : "s"}
            </p>
          </div>

          <button
            type="button"
            onClick={handleExportPdf}
            disabled={exporting}
            className="inline-flex shrink-0 items-center justify-center gap-2 rounded-xl border bg-white px-4 py-2.5 text-sm font-medium text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <span aria-hidden="true">
              ↓
            </span>

            {exporting
              ? "Generando PDF..."
              : "Exportar PDF"}
          </button>
        </div>
      </div>

      <div className="mt-5 border-b">
        <div className="flex gap-1 overflow-x-auto">
          {tabs.map((tab) => {
            const isActive = activeTab === tab.key

            return (
              <button
                key={tab.key}
                type="button"
                onClick={() => setActiveTab(tab.key)}
                className={`whitespace-nowrap border-b-2 px-4 py-3 text-sm font-medium transition ${
                  isActive
                    ? "border-primary text-primary"
                    : "border-transparent text-slate-500 hover:text-slate-900"
                }`}
              >
                {tab.label}
              </button>
            )
          })}
        </div>
      </div>

      {activeTab === "resumen" && (
        <div>
          <ObservationTeacherSnapshot
            promedio={teacher.promedio}
            clasificacion={teacher.clasificacion}
            historial={teacher.historial}
          />

          <div className="mt-6 rounded-2xl border bg-slate-50/60 p-5">
            <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-400">
              LECTURA RÁPIDA
            </p>

            <p className="mt-3 text-sm leading-6 text-slate-700">
              El docente registra{" "}
              <span className="font-semibold">
                {teacher.observaciones}
              </span>{" "}
              observación
              {teacher.observaciones === 1 ? "" : "es"} con un promedio de{" "}
              <span className="font-semibold">
                {teacher.promedio.toFixed(1)}
              </span>{" "}
              puntos y clasificación{" "}
              <span className="font-semibold">
                {teacher.clasificacion}
              </span>
              .
            </p>
          </div>
        </div>
      )}

      {activeTab === "historial" && (
        <div>
          <ObservationTeacherTimeline
            historial={teacher.historial}
          />

          <div className="mt-8">
            <div>
              <p className="sia-eyebrow">
                DETALLE TABULAR
              </p>

              <h4 className="mt-2 text-lg font-semibold text-slate-900">
                Historial de observaciones
              </h4>
            </div>

            <div className="mt-4 overflow-x-auto rounded-xl border">
              <table className="w-full text-left">
                <thead className="bg-slate-50">
                  <tr>
                    <th className="px-4 py-3 text-xs text-slate-500">
                      Corte
                    </th>

                    <th className="px-4 py-3 text-xs text-slate-500">
                      Servicio
                    </th>

                    <th className="px-4 py-3 text-xs text-slate-500">
                      Grupo
                    </th>

                    <th className="px-4 py-3 text-xs text-slate-500">
                      Asignatura
                    </th>

                    <th className="px-4 py-3 text-xs text-slate-500">
                      Tipo
                    </th>

                    <th className="px-4 py-3 text-xs text-slate-500">
                      Puntaje
                    </th>

                    <th className="px-4 py-3 text-xs text-slate-500">
                      Clasificación
                    </th>
                  </tr>
                </thead>

                <tbody>
                  {teacher.historial.map((item, index) => (
                    <tr
                      key={`${item.corte}-${item.asignatura}-${index}`}
                      className="border-t"
                    >
                      <td className="px-4 py-3 text-sm">
                        {item.corte}
                      </td>

                      <td className="px-4 py-3 text-sm">
                        {item.servicio}
                      </td>

                      <td className="px-4 py-3 text-sm">
                        {item.grupo}
                      </td>

                      <td className="px-4 py-3 text-sm">
                        {item.asignatura}
                      </td>

                      <td className="px-4 py-3 text-sm">
                        {item.tipo}
                      </td>

                      <td className="px-4 py-3 text-sm font-semibold">
                        {item.puntaje.toFixed(1)}
                      </td>

                      <td className="px-4 py-3 text-sm">
                        {item.clasificacion}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {activeTab === "comentarios" && (
        <div className="mt-6 grid gap-4">
          <div className="rounded-xl border border-emerald-100 bg-emerald-50/50 p-5">
            <p className="text-xs font-semibold uppercase tracking-wide text-emerald-700">
              Fortalezas
            </p>

            {teacher.fortalezas.length > 0 ? (
              <ul className="mt-3 space-y-2">
                {teacher.fortalezas.map((item) => (
                  <li
                    key={item}
                    className="text-sm leading-6 text-slate-700"
                  >
                    • {item}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="mt-3 text-sm text-slate-600">
                Sin registro.
              </p>
            )}
          </div>

          <div className="rounded-xl border border-amber-100 bg-amber-50/60 p-5">
            <p className="text-xs font-semibold uppercase tracking-wide text-amber-700">
              Áreas de oportunidad
            </p>

            {teacher.areas_oportunidad.length > 0 ? (
              <ul className="mt-3 space-y-2">
                {teacher.areas_oportunidad.map((item) => (
                  <li
                    key={item}
                    className="text-sm leading-6 text-slate-700"
                  >
                    • {item}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="mt-3 text-sm text-slate-600">
                Sin registro.
              </p>
            )}
          </div>

          <div className="rounded-xl border border-sky-100 bg-sky-50/60 p-5">
            <p className="text-xs font-semibold uppercase tracking-wide text-sky-700">
              Recomendaciones
            </p>

            {teacher.recomendaciones.length > 0 ? (
              <ul className="mt-3 space-y-2">
                {teacher.recomendaciones.map((item) => (
                  <li
                    key={item}
                    className="text-sm leading-6 text-slate-700"
                  >
                    • {item}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="mt-3 text-sm text-slate-600">
                Sin registro.
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  )
}