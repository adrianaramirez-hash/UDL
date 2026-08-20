import { SIACard } from "@/components/cards/SIACard"

export function AIAdvisor() {
  return (
    <SIACard className="mb-8 p-8">
      <div className="flex items-center gap-3">
        <div className="flex h-12 w-12 items-center justify-center rounded-full bg-sky-100 text-lg">
          ✦
        </div>

        <div>
          <p className="sia-eyebrow">
            ASISTENTE SIA
          </p>

          <p className="text-sm text-muted-foreground">
            Recomendación preparada para ti
          </p>
        </div>
      </div>

      <div className="mt-6 space-y-5">
        <p className="text-lg leading-8">
          Detecté tres temas que requieren seguimiento.
          La salud institucional permanece estable,
          aunque existen áreas que conviene revisar
          antes del siguiente corte.
        </p>

        <p className="text-muted-foreground">
          También encontré mejoras relevantes en desempeño
          docente y permanencia estudiantil.
        </p>
      </div>

      <div className="mt-8 flex gap-3">
        <button className="rounded-xl bg-primary px-5 py-2 text-sm font-medium text-white">
          Hablar con SIA
        </button>

        <button className="rounded-xl border px-5 py-2 text-sm">
          Ver análisis completo
        </button>
      </div>
    </SIACard>
  )
}