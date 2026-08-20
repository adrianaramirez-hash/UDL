import { QuestionCard } from "@/components/cards/QuestionCard";

const questions = [
  {
    title: "¿Cómo está mi carrera?",
    description:
      "Consulta la salud general, tendencias, comparativos y principales indicadores."
  },
  {
    title: "¿Cómo están mis docentes?",
    description:
      "Observación de clases, evaluación docente y avance de capacitación."
  },
  {
    title: "¿Cómo están mis alumnos?",
    description:
      "Reprobación, bajas, permanencia y avance hacia la titulación."
  },
  {
    title: "¿Están aprendiendo?",
    description:
      "Resultados de CENEVAL y exámenes departamentales."
  },
  {
    title: "¿Qué opinan de nosotros?",
    description:
      "Encuesta de calidad y percepción institucional."
  },
  {
    title: "¿Qué debo hacer hoy?",
    description:
      "Acciones sugeridas por el Asesor SIA."
  },
];

export function QuestionGrid() {
  return (
    <section className="mb-10">

      <p className="sia-eyebrow">
        ÁREAS DE ANÁLISIS
      </p>

      <h2 className="mt-2 text-3xl sia-heading">
        ¿Qué necesitas resolver?
      </h2>

      <p className="mb-8 text-muted-foreground">
        Navega por preguntas, no por módulos.
      </p>

      <div className="grid gap-6 md:grid-cols-2 xl:grid-cols-3">

        {questions.map((item) => (
  <QuestionCard
    key={item.title}
    title={item.title}
    description={item.description}
  />
))}

      </div>

    </section>
  );
}