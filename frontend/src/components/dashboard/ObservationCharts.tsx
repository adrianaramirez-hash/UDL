import { useEffect, useMemo, useState } from "react"

import {
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"

import { SIACard } from "@/components/cards/SIACard"
import type { ObservationFilterState } from "@/pages/ObservacionClases"

type TrendItem = {
  corte: string
  promedio: number
  observaciones: number
  consolidado: number
  en_proceso: number
  no_consolidado: number
  pct_consolidado: number
}

type ObservationChartsProps = {
  filters: ObservationFilterState
}

const API_BASE_URL = "http://127.0.0.1:8000"

export function ObservationCharts({
  filters,
}: ObservationChartsProps) {
  const [trendData, setTrendData] = useState<TrendItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    async function cargarTendencia() {
      try {
        setLoading(true)
        setError(null)

        const params = new URLSearchParams()

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
          `${API_BASE_URL}/api/observacion/tendencia${
            query ? `?${query}` : ""
          }`,
        )

        if (!response.ok) {
          throw new Error(
            "No fue posible obtener la tendencia de observaciones.",
          )
        }

        const data: TrendItem[] = await response.json()

        setTrendData(data)
      } catch (err) {
        console.error(err)

        setError(
          "No fue posible cargar la tendencia de observaciones.",
        )
      } finally {
        setLoading(false)
      }
    }

    cargarTendencia()
  }, [filters.servicio, filters.tipo])

  const filteredTrendData = useMemo(() => {
    if (
      !filters.corte ||
      filters.corte === "Todos los cortes"
    ) {
      return trendData
    }

    return trendData.filter(
      (item) => item.corte === filters.corte,
    )
  }, [trendData, filters.corte])

  const distributionData = useMemo(() => {
    const totals = filteredTrendData.reduce(
      (acc, item) => {
        acc.consolidado += item.consolidado
        acc.enProceso += item.en_proceso
        acc.noConsolidado += item.no_consolidado

        return acc
      },
      {
        consolidado: 0,
        enProceso: 0,
        noConsolidado: 0,
      },
    )

    return [
      {
        name: "Consolidado",
        value: totals.consolidado,
        color: "#0f766e",
      },
      {
        name: "En proceso",
        value: totals.enProceso,
        color: "#f59e0b",
      },
      {
        name: "No consolidado",
        value: totals.noConsolidado,
        color: "#ef4444",
      },
    ]
  }, [filteredTrendData])

  if (error) {
    return (
      <section className="mt-8">
        <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
          {error}
        </div>
      </section>
    )
  }

  return (
    <section className="mt-8 grid gap-6 xl:grid-cols-2">
      <SIACard className="p-6">
        <p className="sia-eyebrow">
          TENDENCIA
        </p>

        <h3 className="mt-2 text-2xl sia-heading">
          Evolución por corte
        </h3>

        <p className="mt-2 text-sm text-muted-foreground">
          Promedio obtenido en cada corte de 30 días.
        </p>

        <div className="mt-6 h-[320px]">
          {loading ? (
            <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
              Cargando tendencia...
            </div>
          ) : (
            <ResponsiveContainer
              width="100%"
              height="100%"
            >
              <LineChart data={filteredTrendData}>
                <CartesianGrid strokeDasharray="3 3" />

                <XAxis
                  dataKey="corte"
                  tick={{ fontSize: 11 }}
                  interval="preserveStartEnd"
                />

                <YAxis
                  domain={["dataMin - 5", "dataMax + 5"]}
                  tick={{ fontSize: 11 }}
                />

                <Tooltip />

                <Line
                  type="monotone"
                  dataKey="promedio"
                  stroke="#0f766e"
                  strokeWidth={3}
                  dot={{ r: 4 }}
                  activeDot={{ r: 6 }}
                />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
      </SIACard>

      <SIACard className="p-6">
        <p className="sia-eyebrow">
          DISTRIBUCIÓN
        </p>

        <h3 className="mt-2 text-2xl sia-heading">
          Resultado de observaciones
        </h3>

        <p className="mt-2 text-sm text-muted-foreground">
          Clasificación acumulada según los filtros seleccionados.
        </p>

        <div className="mt-6 h-[260px]">
          {loading ? (
            <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
              Cargando distribución...
            </div>
          ) : (
            <ResponsiveContainer
              width="100%"
              height="100%"
            >
              <PieChart>
                <Pie
                  data={distributionData}
                  dataKey="value"
                  nameKey="name"
                  innerRadius={65}
                  outerRadius={95}
                >
                  {distributionData.map((entry) => (
                    <Cell
                      key={entry.name}
                      fill={entry.color}
                    />
                  ))}
                </Pie>

                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          )}
        </div>

        <div className="mt-3 grid gap-2">
          {distributionData.map((item) => (
            <div
              key={item.name}
              className="flex items-center justify-between text-sm"
            >
              <div className="flex items-center gap-2">
                <span
                  className="h-2.5 w-2.5 rounded-full"
                  style={{
                    backgroundColor: item.color,
                  }}
                />

                <span className="text-slate-600">
                  {item.name}
                </span>
              </div>

              <span className="font-semibold text-slate-900">
                {item.value}
              </span>
            </div>
          ))}
        </div>
      </SIACard>
    </section>
  )
}