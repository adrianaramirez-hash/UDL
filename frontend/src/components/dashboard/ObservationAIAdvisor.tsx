import { useEffect, useState } from "react"

import { SIACard } from "@/components/cards/SIACard"
import { SIADrawer } from "@/components/common/SIADrawer"
import { ObservationChat } from "@/components/dashboard/ObservationChat"
import type { ObservationFilterState } from "@/pages/ObservacionClases"

type ObservationAIAdvisorProps = {
  filters: ObservationFilterState
}

type ObservationInsights = {
  resumen: string
  fortalezas: string[]
  alertas: string[]
  recomendaciones: string[]
}

const API_BASE_URL = "http://127.0.0.1:8000"

export function ObservationAIAdvisor({
  filters,
}: ObservationAIAdvisorProps) {
  const [insights, setInsights] =
    useState<ObservationInsights | null>(null)

  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const [chatOpen, setChatOpen] = useState(false)

  useEffect(() => {
    async function cargarInsights() {
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
          `${API_BASE_URL}/api/observacion/insights${
            query ? `?${query}` : ""
          }`,
        )

        if (!response.ok) {
          throw new Error(
            "No fue posible obtener el análisis del Asesor SIA.",
          )
        }

        const data: ObservationInsights =
          await response.json()

        setInsights(data)
      } catch (err) {
        console.error(err)

        setError(
          "No fue posible cargar el análisis del Asesor SIA.",
        )
      } finally {
        setLoading(false)
      }
    }

    cargarInsights()
  }, [
    filters.corte,
    filters.servicio,
    filters.tipo,
  ])

  return (
    <>
      <SIACard className="mt-8 p-7">
        <div className="flex items-start gap-4">
          <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-sky-100 text-lg text-primary">
            ✦
          </div>

          <div className="flex-1">
            <p className="sia-eyebrow">
              ASESOR SIA · OBSERVACIÓN DE CLASES
            </p>

            <p className="mt-1 text-sm text-muted-foreground">
              Análisis ejecutivo basado en los filtros seleccionados
            </p>

            {loading ? (
              <div className="mt-6 space-y-3">
                <div className="h-6 w-64 animate-pulse rounded bg-slate-200" />
                <div className="h-4 w-full animate-pulse rounded bg-slate-100" />
                <div className="h-4 w-5/6 animate-pulse rounded bg-slate-100" />
              </div>
            ) : error ? (
              <div className="mt-6 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
                {error}
              </div>
            ) : insights ? (
              <>
                <h2 className="mt-5 text-2xl sia-heading">
                  Lectura ejecutiva del periodo
                </h2>

                <p className="mt-3 max-w-5xl text-base leading-7 text-slate-700">
                  {insights.resumen}
                </p>

                <div className="mt-6 grid gap-4 md:grid-cols-2">
                  <div className="rounded-xl border border-emerald-100 bg-emerald-50/60 p-4">
                    <p className="text-xs font-semibold uppercase tracking-wide text-emerald-700">
                      Fortalezas
                    </p>

                    {insights.fortalezas.length > 0 ? (
                      <ul className="mt-3 space-y-2">
                        {insights.fortalezas.map((item) => (
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
                        Sin fortalezas destacadas para este filtro.
                      </p>
                    )}
                  </div>

                  <div className="rounded-xl border border-amber-100 bg-amber-50/70 p-4">
                    <p className="text-xs font-semibold uppercase tracking-wide text-amber-700">
                      Requiere atención
                    </p>

                    {insights.alertas.length > 0 ? (
                      <ul className="mt-3 space-y-2">
                        {insights.alertas.map((item) => (
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
                        No se detectaron alertas relevantes.
                      </p>
                    )}
                  </div>
                </div>

                <div className="mt-4 rounded-xl border bg-slate-50/60 p-4">
                  <p className="text-xs font-semibold uppercase tracking-wide text-slate-600">
                    Recomendaciones
                  </p>

                  {insights.recomendaciones.length > 0 ? (
                    <ul className="mt-3 space-y-2">
                      {insights.recomendaciones.map((item) => (
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
                      No hay recomendaciones adicionales para este filtro.
                    </p>
                  )}
                </div>

                <div className="mt-6 flex flex-wrap gap-3 border-t border-slate-100 pt-5">
                  <button
                    type="button"
                    onClick={() => setChatOpen(true)}
                    className="rounded-xl bg-primary px-5 py-2.5 text-sm font-medium text-white transition hover:opacity-90"
                  >
                    Preguntar al Asesor SIA
                  </button>

                  <button
                    type="button"
                    onClick={() => {
                      document
                        .getElementById("seguimiento-prioritario")
                        ?.scrollIntoView({
                          behavior: "smooth",
                          block: "start",
                        })
                    }}
                    className="rounded-xl border bg-white px-5 py-2.5 text-sm font-medium transition hover:bg-slate-50"
                  >
                    Ver casos prioritarios
                  </button>
                </div>
              </>
            ) : null}
          </div>
        </div>
      </SIACard>

      <SIADrawer
        open={chatOpen}
        title="Asesor SIA"
        width="lg"
        onClose={() => setChatOpen(false)}
      >
        <ObservationChat />
      </SIADrawer>
    </>
  )
}