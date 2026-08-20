import { useRef, useState } from "react"

import { ObservationAIAdvisor } from "@/components/dashboard/ObservationAIAdvisor"
import { ObservationCharts } from "@/components/dashboard/ObservationCharts"
import { ObservationOverview } from "@/components/dashboard/ObservationOverview"
import { ObservationPriorityTable } from "@/components/dashboard/ObservationPriorityTable"
import { Header } from "@/components/layout/Header"
import { Sidebar } from "@/components/layout/Sidebar"
import { ObservationChat } from "@/components/dashboard/ObservationChat"

export type ObservationFilterState = {
  corte: string
  servicio: string
  tipo: string
}

export type ObservationPriorityFilter =
  | "Todos"
  | "En proceso"
  | "No consolidado"

export default function ObservacionClases() {
  const [filters, setFilters] = useState<ObservationFilterState>({
    corte: "Todos los cortes",
    servicio: "Todos los servicios",
    tipo: "Todos los tipos",
  })

  const [priorityFilter, setPriorityFilter] =
    useState<ObservationPriorityFilter>("Todos")

  const prioritySectionRef = useRef<HTMLDivElement | null>(null)

  function handlePriorityFilter(
    value: ObservationPriorityFilter,
  ) {
    setPriorityFilter(value)

    window.setTimeout(() => {
      prioritySectionRef.current?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      })
    }, 50)
  }

  return (
    <div className="flex min-h-screen bg-background">
      <Sidebar />

      <main className="flex flex-1 flex-col">
        <Header />

        <div className="mx-auto w-full max-w-7xl px-8 py-8">
          <p className="sia-eyebrow">
            CALIDAD ACADÉMICA
          </p>

          <h1 className="mt-2 text-5xl sia-heading">
            Observación de Clases
          </h1>

          <p className="mt-3 max-w-3xl text-muted-foreground">
            Analiza el desempeño observado en aula, identifica tendencias,
            fortalezas y casos que requieren seguimiento.
          </p>

          <ObservationOverview
            filters={filters}
            onFiltersChange={setFilters}
            priorityFilter={priorityFilter}
            onPriorityFilterChange={handlePriorityFilter}
          />

          <ObservationAIAdvisor
            filters={filters}
          />

          <ObservationCharts
            filters={filters}
          />

          <div
            ref={prioritySectionRef}
            className="scroll-mt-24"
          >
            <ObservationPriorityTable
              filters={filters}
              priorityFilter={priorityFilter}
              onPriorityFilterChange={setPriorityFilter}
            />
          </div>
        </div>
      </main>
    </div>
  )
}