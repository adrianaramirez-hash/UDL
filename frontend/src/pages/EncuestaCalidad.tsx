import { useEffect, useState } from "react"
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"

import { Header } from "@/components/layout/Header"
import { Sidebar } from "@/components/layout/Sidebar"
import {
  getCalidadSummary,
  type CalidadSummary,
} from "@/services/calidadService"

function getNivel(indice: number | null) {
  if (indice === null) return "Sin datos"
  if (indice <= 60) return "Crítico"
  if (indice <= 75) return "Atención"
  if (indice <= 85) return "Adecuado"
  return "Fortaleza"
}

function getNivelClasses(indice: number | null) {
  const nivel = getNivel(indice)

  if (nivel === "Crítico") {
    return "border-red-200 bg-red-50 text-red-700"
  }

  if (nivel === "Atención") {
    return "border-amber-200 bg-amber-50 text-amber-700"
  }

  if (nivel === "Adecuado") {
    return "border-emerald-200 bg-emerald-50 text-emerald-700"
  }

  if (nivel === "Fortaleza") {
    return "border-emerald-300 bg-emerald-100 text-emerald-800"
  }

  return "border-slate-200 bg-slate-50 text-slate-600"
}

export default function EncuestaCalidad() {
  const [summary, setSummary] = useState<CalidadSummary | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    async function cargarDatos() {
      try {
        setLoading(true)
        setError(null)

        const data = await getCalidadSummary()

        setSummary(data)
      } catch (err) {
        setError(
          err instanceof Error
            ? err.message
            : "No fue posible cargar Encuesta de Calidad.",
        )
      } finally {
        setLoading(false)
      }
    }

    cargarDatos()
  }, [])

  const chartData =
    summary?.secciones
      .filter((item) => item.indice !== null)
      .map((item) => ({
        nombre: item.seccion_nombre,
        indice: item.indice,
      })) ?? []

  return (
    <div className="flex min-h-screen bg-background">
      <Sidebar />

      <main className="flex flex-1 flex-col">
        <Header />

        <div className="mx-auto w-full max-w-7xl px-8 py-8">
          <p className="sia-eyebrow">
            CALIDAD ACADÉMICA
          </p>

          <div className="mt-2 flex items-end justify-between gap-6">
            <div>
              <h1 className="text-5xl sia-heading">
                Encuesta de Calidad
              </h1>

              <p className="mt-3 max-w-3xl text-muted-foreground">
                Monitorea la percepción estudiantil e identifica
                fortalezas, alertas y áreas de oportunidad institucional.
              </p>
            </div>

            {summary && (
              <div className="text-right text-sm text-muted-foreground">
                <p>Periodo 2026</p>
                <p>
                  Filas {summary.fila_inicio}–{summary.fila_fin}
                </p>
              </div>
            )}
          </div>

          {loading && (
            <div className="mt-10 rounded-2xl border bg-card p-8">
              <p className="text-muted-foreground">
                Cargando resultados de Encuesta de Calidad...
              </p>
            </div>
          )}

          {error && (
            <div className="mt-10 rounded-2xl border border-destructive/30 bg-card p-8">
              <p className="font-medium">
                No fue posible cargar los resultados.
              </p>

              <p className="mt-2 text-sm text-muted-foreground">
                {error}
              </p>
            </div>
          )}

          {summary && !loading && (
            <>
              <div className="mt-10 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                <div className="rounded-2xl border bg-card p-6 shadow-sm">
                  <p className="text-sm text-muted-foreground">
                    Índice general
                  </p>

                  <p className="mt-3 text-4xl font-semibold tracking-tight">
                    {summary.indice_general?.toFixed(2) ?? "—"}
                  </p>

                  <div className="mt-3 flex items-center justify-between gap-3">
                    <p className="text-xs text-muted-foreground">
                      Escala institucional 0–100
                    </p>

                    <span className={`rounded-full border px-3 py-1 text-xs font-medium ${getNivelClasses(summary.indice_general)}`}>
                      {getNivel(summary.indice_general)}
                    </span>
                  </div>
                </div>

                <div className="rounded-2xl border bg-card p-6 shadow-sm">
                  <p className="text-sm text-muted-foreground">
                    Encuestas
                  </p>

                  <p className="mt-3 text-4xl font-semibold tracking-tight">
                    {summary.encuestas}
                  </p>

                  <p className="mt-2 text-xs text-muted-foreground">
                    Respuestas detectadas en el periodo
                  </p>
                </div>

                <div className="rounded-2xl border bg-card p-6 shadow-sm">
                  <p className="text-sm text-muted-foreground">
                    Respuestas KPI
                  </p>

                  <p className="mt-3 text-4xl font-semibold tracking-tight">
                    {summary.respuestas_kpi_validas.toLocaleString()}
                  </p>

                  <p className="mt-2 text-xs text-muted-foreground">
                    Respuestas válidas para análisis
                  </p>
                </div>

                <div className="rounded-2xl border bg-card p-6 shadow-sm">
                  <p className="text-sm text-muted-foreground">
                    Control de calidad
                  </p>

                  <p className="mt-3 text-4xl font-semibold tracking-tight">
                    {summary.qa_total}
                  </p>

                  <p className="mt-2 text-xs text-muted-foreground">
                    Incidencias detectadas por QA
                  </p>
                </div>
              </div>

              <div className="mt-6 rounded-2xl border bg-card p-6 shadow-sm">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">
                    RESULTADOS POR SECCIÓN
                  </p>

                  <h2 className="mt-1 text-2xl font-semibold tracking-tight">
                    Índice de calidad
                  </h2>

                  <p className="mt-1 text-sm text-muted-foreground">
                    Comparativo preliminar de las secciones con respuestas disponibles.
                  </p>
                </div>

                <div className="mt-8 h-[460px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={chartData}
                      layout="vertical"
                      margin={{
                        top: 0,
                        right: 30,
                        bottom: 0,
                        left: 40,
                      }}
                    >
                      <CartesianGrid
                        strokeDasharray="3 3"
                        horizontal={false}
                      />

                      <XAxis
                        type="number"
                        domain={[0, 100]}
                      />

                      <YAxis
                        type="category"
                        dataKey="nombre"
                        width={220}
                        tick={{
                          fontSize: 12,
                        }}
                      />

                      <Tooltip />

                      <Bar
                        dataKey="indice"
                        name="Índice"
                        radius={[0, 6, 6, 0]}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </>
          )}
        </div>
      </main>
    </div>
  )
}


