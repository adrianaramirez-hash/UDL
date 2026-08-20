import { MetricCard } from "@/components/cards/MetricCard"

const metrics = [
  {
    label: "Calidad académica",
    value: "87.4%",
    change: "+2.3",
    status: "Bueno",
  },
  {
    label: "Desempeño docente",
    value: "92.1%",
    change: "+0.8",
    status: "Excelente",
  },
  {
    label: "Retención estudiantil",
    value: "91.7%",
    change: "+1.4",
    status: "Bueno",
  },
  {
    label: "Cobertura de evaluación",
    value: "78.5%",
    change: "-1.2",
    status: "Atención",
  },
]

export function KPIGrid() {
  return (
    <section className="mb-10">
      <p className="sia-eyebrow">
        INDICADORES ESTRATÉGICOS
      </p>

      <h2 className="mt-2 text-3xl sia-heading">
        Métricas institucionales clave
      </h2>

      <div className="mt-6 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {metrics.map((metric) => (
          <MetricCard
            key={metric.label}
            label={metric.label}
            value={metric.value}
            change={metric.change}
            status={metric.status}
          />
        ))}
      </div>
    </section>
  )
}