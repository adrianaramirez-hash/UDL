import { useEffect, useState } from "react"

import { SIACard } from "@/components/cards/SIACard"
import { SIADrawer } from "@/components/common/SIADrawer"
import { ObservationTeacherDetail } from "@/components/dashboard/ObservationTeacherDetail"
import type {
  ObservationFilterState,
  ObservationPriorityFilter,
} from "@/pages/ObservacionClases"

type PriorityCase = {
  docente: string
  servicio: string
  tipo: string
  promedio: number
  observaciones: number
  clasificacion: string
}

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

type ObservationPriorityTableProps = {
  filters: ObservationFilterState
  priorityFilter: ObservationPriorityFilter
  onPriorityFilterChange: (
    value: ObservationPriorityFilter,
  ) => void
}

const API_BASE_URL = "http://127.0.0.1:8000"

export function ObservationPriorityTable({
  filters,
  priorityFilter,
  onPriorityFilterChange,
}: ObservationPriorityTableProps) {
  const [priorityCases, setPriorityCases] = useState<PriorityCase[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const [selectedTeacher, setSelectedTeacher] =
    useState<TeacherDetail | null>(null)

  const [detailLoading, setDetailLoading] = useState(false)
  const [detailError, setDetailError] = useState<string | null>(null)
  const [drawerOpen, setDrawerOpen] = useState(false)

  const [exportingPdf, setExportingPdf] = useState(false)

  useEffect(() => {
    async function cargarCasosPrioritarios() {
      try {
        setLoading(true)
        setError(null)

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
          `${API_BASE_URL}/api/observacion/casos-prioritarios${
            query ? `?${query}` : ""
          }`,
        )

        if (!response.ok) {
          throw new Error(
            "No fue posible obtener los casos prioritarios.",
          )
        }

        const data: PriorityCase[] = await response.json()

        setPriorityCases(data)
      } catch (err) {
        console.error(err)

        setError(
          "No fue posible cargar los casos prioritarios.",
        )
      } finally {
        setLoading(false)
      }
    }

    cargarCasosPrioritarios()
  }, [
    filters.corte,
    filters.servicio,
    filters.tipo,
  ])

  const filteredCases =
    priorityFilter === "Todos"
      ? priorityCases
      : priorityCases.filter(
          (item) =>
            item.clasificacion === priorityFilter,
        )

  async function analizarDocente(
    nombreDocente: string,
  ) {
    try {
      setDrawerOpen(true)
      setSelectedTeacher(null)
      setDetailLoading(true)
      setDetailError(null)

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
        `${API_BASE_URL}/api/observacion/docente/${encodeURIComponent(
          nombreDocente.trim(),
        )}${query ? `?${query}` : ""}`,
      )

      if (!response.ok) {
        throw new Error(
          "No fue posible obtener el detalle del docente.",
        )
      }

      const data: TeacherDetail =
        await response.json()

      setSelectedTeacher(data)
    } catch (err) {
      console.error(err)

      setDetailError(
        "No fue posible cargar el detalle del docente.",
      )
    } finally {
      setDetailLoading(false)
    }
  }

  async function exportarPdfDocente() {
    if (!selectedTeacher) {
      return
    }

    try {
      setExportingPdf(true)

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
        `${API_BASE_URL}/api/observacion/docente/${encodeURIComponent(
          selectedTeacher.docente.trim(),
        )}/pdf${query ? `?${query}` : ""}`,
      )

      if (!response.ok) {
        throw new Error(
          "No fue posible generar el reporte PDF.",
        )
      }

      const blob = await response.blob()

      const url =
        window.URL.createObjectURL(blob)

      const nombreLimpio =
        selectedTeacher.docente
          .trim()
          .replace(/\s+/g, "_")

      const link =
        document.createElement("a")

      link.href = url
      link.download =
        `reporte_observacion_${nombreLimpio}.pdf`

      document.body.appendChild(link)

      link.click()

      document.body.removeChild(link)

      window.URL.revokeObjectURL(url)
    } catch (err) {
      console.error(err)

      window.alert(
        "No fue posible exportar el reporte PDF.",
      )
    } finally {
      setExportingPdf(false)
    }
  }

  return (
    <>
      <section
        id="seguimiento-prioritario"
        className="mt-8 scroll-mt-24"
      >
        <div className="mb-5 flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="sia-eyebrow">
              SEGUIMIENTO PRIORITARIO
            </p>

            <h2 className="mt-2 text-3xl sia-heading">
              ¿Dónde debemos intervenir?
            </h2>

            <p className="mt-2 text-muted-foreground">
              Docentes clasificados como En proceso o No consolidado
              según los filtros seleccionados.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            {priorityFilter !== "Todos" && (
              <button
                type="button"
                onClick={() =>
                  onPriorityFilterChange("Todos")
                }
                className="rounded-xl border bg-white px-4 py-2 text-sm font-medium transition hover:bg-slate-50"
              >
                Mostrar todos
              </button>
            )}

            <div className="rounded-xl border bg-white px-4 py-2 text-sm">
              {loading
                ? "Cargando..."
                : `${filteredCases.length} casos`}
            </div>
          </div>
        </div>

        {priorityFilter !== "Todos" && (
          <div className="mb-4 flex items-center gap-2 rounded-xl border border-primary/10 bg-primary/5 px-4 py-3 text-sm">
            <span className="text-slate-500">
              Filtro activo:
            </span>

            <span className="font-semibold text-primary">
              {priorityFilter}
            </span>
          </div>
        )}

        {error && (
          <div className="mb-4 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
            {error}
          </div>
        )}

        <SIACard className="overflow-hidden">
          {loading ? (
            <div className="p-10 text-center text-sm text-muted-foreground">
              Cargando casos prioritarios...
            </div>
          ) : filteredCases.length === 0 ? (
            <div className="p-10 text-center">
              <p className="font-medium text-slate-800">
                No hay casos para este filtro.
              </p>

              <p className="mt-2 text-sm text-muted-foreground">
                No se encontraron docentes con clasificación{" "}
                <span className="font-medium">
                  {priorityFilter}
                </span>{" "}
                para los filtros seleccionados.
              </p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left">
                <thead className="border-b bg-slate-50/70">
                  <tr>
                    <th className="px-6 py-4 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">
                      Docente
                    </th>

                    <th className="px-6 py-4 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">
                      Servicio
                    </th>

                    <th className="px-6 py-4 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">
                      Tipo
                    </th>

                    <th className="px-6 py-4 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">
                      Promedio
                    </th>

                    <th className="px-6 py-4 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">
                      Observaciones
                    </th>

                    <th className="px-6 py-4 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">
                      Clasificación
                    </th>

                    <th className="px-6 py-4" />
                  </tr>
                </thead>

                <tbody>
                  {filteredCases.map((item) => {
                    const isCritical =
                      item.clasificacion ===
                      "No consolidado"

                    return (
                      <tr
                        key={`${item.docente}-${item.servicio}-${item.tipo}`}
                        className="border-b last:border-b-0 hover:bg-slate-50/60"
                      >
                        <td className="px-6 py-4 text-sm font-medium text-slate-900">
                          {item.docente.trim()}
                        </td>

                        <td className="px-6 py-4 text-sm text-slate-600">
                          {item.servicio}
                        </td>

                        <td className="px-6 py-4 text-sm text-slate-600">
                          {item.tipo}
                        </td>

                        <td className="px-6 py-4 text-sm font-semibold">
                          {item.promedio.toFixed(1)}
                        </td>

                        <td className="px-6 py-4 text-sm text-slate-600">
                          {item.observaciones}
                        </td>

                        <td className="px-6 py-4">
                          <span
                            className={`rounded-full px-2.5 py-1 text-xs font-medium ${
                              isCritical
                                ? "bg-red-50 text-red-600"
                                : "bg-amber-50 text-amber-700"
                            }`}
                          >
                            {item.clasificacion}
                          </span>
                        </td>

                        <td className="px-6 py-4 text-right">
                          <button
                            type="button"
                            onClick={() =>
                              analizarDocente(
                                item.docente,
                              )
                            }
                            className="text-sm font-medium text-primary"
                          >
                            Analizar →
                          </button>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </SIACard>
      </section>

      <SIADrawer
        open={drawerOpen}
        title="Detalle del docente"
        width="xl"
        onClose={() =>
          setDrawerOpen(false)
        }
      >
        {detailLoading && (
          <div className="rounded-xl border bg-white p-6 text-sm text-muted-foreground">
            Analizando información del docente...
          </div>
        )}

        {detailError && (
          <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
            {detailError}
          </div>
        )}

        {selectedTeacher &&
          !detailLoading && (
            <>
              <div className="mb-5 flex justify-end border-b pb-5">
                <button
                  type="button"
                  onClick={exportarPdfDocente}
                  disabled={exportingPdf}
                  className="inline-flex items-center justify-center gap-2 rounded-xl bg-primary px-5 py-2.5 text-sm font-medium text-white transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <span aria-hidden="true">
                    ↓
                  </span>

                  {exportingPdf
                    ? "Generando PDF..."
                    : "Exportar PDF"}
                </button>
              </div>

              <ObservationTeacherDetail
                teacher={selectedTeacher}
              />
            </>
          )}
      </SIADrawer>
    </>
  )
}