import { useEffect, useState } from "react"

import { MetricCard } from "@/components/cards/MetricCard"
import {
  getObservationFilters,
  getObservationSummary,
  type ObservationFilterOptions,
  type ObservationSummary,
} from "@/services/observationService"

import type {
  ObservationFilterState,
  ObservationPriorityFilter,
} from "@/pages/ObservacionClases"

type ObservationOverviewProps = {
  filters: ObservationFilterState
  onFiltersChange: (filters: ObservationFilterState) => void
  priorityFilter: ObservationPriorityFilter
  onPriorityFilterChange: (value: ObservationPriorityFilter) => void
}

const DEFAULT_FILTERS: ObservationFilterOptions = {
  cortes: [],
  servicios: [],
  tipos: [],
}

const API_BASE_URL = "http://127.0.0.1:8000"

export function ObservationOverview({
  filters,
  onFiltersChange,
  priorityFilter,
  onPriorityFilterChange,
}: ObservationOverviewProps) {
  const [summary, setSummary] = useState<ObservationSummary | null>(null)

  const [filterOptions, setFilterOptions] =
    useState<ObservationFilterOptions>(DEFAULT_FILTERS)

  const [loading, setLoading] = useState(true)
  const [filtersLoading, setFiltersLoading] = useState(true)
  const [exportingPdf, setExportingPdf] = useState(false)

  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    async function cargarFiltros() {
      try {
        setFiltersLoading(true)

        const data = await getObservationFilters()

        setFilterOptions(data)
      } catch (err) {
        console.error(err)

        setError(
          "No fue posible cargar los filtros.",
        )
      } finally {
        setFiltersLoading(false)
      }
    }

    cargarFiltros()
  }, [])

  useEffect(() => {
    async function cargarResumen() {
      try {
        setLoading(true)
        setError(null)

        const data = await getObservationSummary({
          corte: filters.corte,
          servicio: filters.servicio,
          tipo: filters.tipo,
        })

        setSummary(data)
      } catch (err) {
        console.error(err)

        setError(
          "No fue posible cargar los indicadores.",
        )
      } finally {
        setLoading(false)
      }
    }

    cargarResumen()
  }, [
    filters.corte,
    filters.servicio,
    filters.tipo,
  ])

  async function exportarReporteEjecutivo() {
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
        `${API_BASE_URL}/api/observacion/reporte-ejecutivo/pdf${
          query ? `?${query}` : ""
        }`,
      )

      if (!response.ok) {
        throw new Error(
          "No fue posible generar el reporte ejecutivo.",
        )
      }

      const blob = await response.blob()

      const url =
        window.URL.createObjectURL(blob)

      const link =
        document.createElement("a")

      link.href = url

      const partesNombre = [
        "reporte_ejecutivo_observacion",
      ]

      if (
        filters.corte &&
        filters.corte !== "Todos los cortes"
      ) {
        partesNombre.push(
          filters.corte
            .replace(/\//g, "-")
            .replace(/\s+/g, "_"),
        )
      }

      if (
        filters.servicio &&
        filters.servicio !== "Todos los servicios"
      ) {
        partesNombre.push(
          filters.servicio
            .replace(/[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑ]+/g, "_"),
        )
      }

      link.download =
        `${partesNombre.join("_")}.pdf`

      document.body.appendChild(link)

      link.click()

      document.body.removeChild(link)

      window.URL.revokeObjectURL(url)
    } catch (err) {
      console.error(err)

      window.alert(
        "No fue posible exportar el reporte ejecutivo.",
      )
    } finally {
      setExportingPdf(false)
    }
  }

  const observationMetrics = [
    {
      label: "Observaciones totales",
      value: loading
        ? "..."
        : String(summary?.observaciones ?? 0),
      status: "Corte seleccionado",
      tooltip:
        "Total de observaciones registradas considerando los filtros seleccionados.",
    },
    {
      label: "Consolidado",
      value: loading
        ? "..."
        : String(summary?.consolidado ?? 0),
      status: "Desempeño consolidado",
      tooltip:
        "Observaciones cuyo resultado se encuentra en nivel Consolidado.",
    },
    {
      label: "En proceso",
      value: loading
        ? "..."
        : String(summary?.en_proceso ?? 0),
      status: "Requiere seguimiento",
      tooltip:
        "Observaciones clasificadas como En proceso y que requieren seguimiento académico.",
      onClick: () =>
        onPriorityFilterChange("En proceso"),
      active:
        priorityFilter === "En proceso",
    },
    {
      label: "No consolidado",
      value: loading
        ? "..."
        : String(summary?.no_consolidado ?? 0),
      status: "Atención prioritaria",
      tooltip:
        "Observaciones clasificadas como No consolidado y que requieren atención prioritaria.",
      onClick: () =>
        onPriorityFilterChange("No consolidado"),
      active:
        priorityFilter === "No consolidado",
    },
  ]

  return (
    <section className="mt-8">
      <div className="flex flex-col gap-5 xl:flex-row xl:items-end xl:justify-between">
        <div>
          <p className="sia-eyebrow">
            RESUMEN EJECUTIVO
          </p>

          <h2 className="mt-2 text-3xl sia-heading">
            ¿Qué está pasando en las aulas?
          </h2>

          <p className="mt-2 max-w-3xl text-muted-foreground">
            El desempeño observado se analiza por cortes de 30 días.
            Los indicadores y visualizaciones se actualizan automáticamente
            según los filtros seleccionados.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <select
            value={filters.corte}
            onChange={(event) =>
              onFiltersChange({
                ...filters,
                corte: event.target.value,
              })
            }
            disabled={filtersLoading}
            className="rounded-xl border bg-white px-4 py-2.5 text-sm disabled:opacity-50"
          >
            <option value="Todos los cortes">
              Todos los cortes
            </option>

            {filterOptions.cortes.map((item) => (
              <option
                key={item}
                value={item}
              >
                {item}
              </option>
            ))}
          </select>

          <select
            value={filters.servicio}
            onChange={(event) =>
              onFiltersChange({
                ...filters,
                servicio: event.target.value,
              })
            }
            disabled={filtersLoading}
            className="rounded-xl border bg-white px-4 py-2.5 text-sm disabled:opacity-50"
          >
            <option value="Todos los servicios">
              Todos los servicios
            </option>

            {filterOptions.servicios.map((item) => (
              <option
                key={item}
                value={item}
              >
                {item}
              </option>
            ))}
          </select>

          <select
            value={filters.tipo}
            onChange={(event) =>
              onFiltersChange({
                ...filters,
                tipo: event.target.value,
              })
            }
            disabled={filtersLoading}
            className="rounded-xl border bg-white px-4 py-2.5 text-sm disabled:opacity-50"
          >
            <option value="Todos los tipos">
              Todos los tipos
            </option>

            {filterOptions.tipos.map((item) => (
              <option
                key={item}
                value={item}
              >
                {item}
              </option>
            ))}
          </select>

          <button
            type="button"
            onClick={exportarReporteEjecutivo}
            disabled={
              exportingPdf ||
              loading ||
              filtersLoading
            }
            className="inline-flex items-center justify-center gap-2 rounded-xl bg-primary px-4 py-2.5 text-sm font-medium text-white transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <span aria-hidden="true">
              ↓
            </span>

            {exportingPdf
              ? "Generando PDF..."
              : "Exportar reporte"}
          </button>
        </div>
      </div>

      {error && (
        <div className="mt-6 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
          {error}
        </div>
      )}

      <div className="mt-6 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {observationMetrics.map((metric) => (
          <MetricCard
            key={metric.label}
            label={metric.label}
            value={metric.value}
            status={metric.status}
            tooltip={metric.tooltip}
            onClick={metric.onClick}
            active={metric.active}
            loading={loading}
          />
        ))}
      </div>
    </section>
  )
}