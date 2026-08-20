import { useNavigate } from "react-router-dom"

import { AcademicAreaCard } from "@/components/cards/AcademicAreaCard"

const academicAreas = [
  {
    title: "Observación de Clases",
    value: "86%",
    status: "Bueno",
    description:
      "Seguimiento del desempeño observado en aula y cumplimiento de criterios institucionales.",
    trend: "+2.1 pts",
    path: "/observacion-clases",
  },
  {
    title: "Evaluación Docente",
    value: "92%",
    status: "Bueno",
    description:
      "Resultado global de la evaluación docente y percepción del desempeño académico.",
    trend: "+0.8 pts",
  },
  {
    title: "Capacitación Docente",
    value: "78%",
    status: "Atención",
    description:
      "Avance de participación y cumplimiento en programas de formación docente.",
    trend: "-1.4 pts",
  },
  {
    title: "Índice de Reprobación",
    value: "12%",
    status: "Atención",
    description:
      "Proporción de estudiantes con resultados no aprobatorios en el periodo actual.",
    trend: "+1.7 pts",
  },
  {
    title: "Bajas",
    value: "4.8%",
    status: "Bueno",
    description:
      "Seguimiento de bajas académicas y posibles señales de riesgo de permanencia.",
    trend: "-0.6 pts",
  },
]

export function AcademicAreaGrid() {
  const navigate = useNavigate()

  return (
    <section className="mt-10">
      <p className="sia-eyebrow">
        VISIÓN EJECUTIVA
      </p>

      <h2 className="mt-2 text-3xl sia-heading">
        Frentes de Calidad Académica
      </h2>

      <p className="mt-2 max-w-3xl text-muted-foreground">
        Una vista integrada para identificar fortalezas, riesgos y áreas que
        requieren seguimiento.
      </p>

      <div className="mt-6 grid gap-5 md:grid-cols-2 xl:grid-cols-3">
        {academicAreas.map((area) => (
          <div
            key={area.title}
            onClick={() => {
              if (area.path) {
                navigate(area.path)
              }
            }}
            className={area.path ? "cursor-pointer" : ""}
          >
            <AcademicAreaCard
              title={area.title}
              value={area.value}
              status={area.status}
              description={area.description}
              trend={area.trend}
            />
          </div>
        ))}
      </div>
    </section>
  )
}