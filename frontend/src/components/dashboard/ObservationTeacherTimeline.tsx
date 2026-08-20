type TeacherHistoryItem = {
  corte: string
  servicio: string
  grupo: string
  asignatura: string
  tipo: string
  puntaje: number
  clasificacion: string
}

type ObservationTeacherTimelineProps = {
  historial: TeacherHistoryItem[]
}

export function ObservationTeacherTimeline({
  historial,
}: ObservationTeacherTimelineProps) {
  if (historial.length === 0) {
    return (
      <div className="rounded-xl border bg-slate-50 p-5 text-sm text-muted-foreground">
        No hay observaciones registradas.
      </div>
    )
  }

  return (
    <div className="mt-6">
      <div>
        <p className="sia-eyebrow">
          TRAYECTORIA
        </p>

        <h4 className="mt-2 text-lg font-semibold text-slate-900">
          Timeline de observaciones
        </h4>

        <p className="mt-1 text-sm text-muted-foreground">
          Secuencia de observaciones registradas para el docente.
        </p>
      </div>

      <div className="mt-5">
        {historial.map((item, index) => {
          const isLast = index === historial.length - 1

          const statusStyles =
            item.clasificacion === "Consolidado"
              ? {
                  dot: "bg-emerald-500",
                  badge: "bg-emerald-50 text-emerald-700",
                }
              : item.clasificacion === "No consolidado"
                ? {
                    dot: "bg-red-500",
                    badge: "bg-red-50 text-red-700",
                  }
                : {
                    dot: "bg-amber-500",
                    badge: "bg-amber-50 text-amber-700",
                  }

          return (
            <div
              key={`${item.corte}-${item.asignatura}-${index}`}
              className="relative flex gap-4"
            >
              <div className="flex flex-col items-center">
                <span
                  className={`mt-1 h-3.5 w-3.5 shrink-0 rounded-full ${statusStyles.dot}`}
                />

                {!isLast && (
                  <div className="my-1 h-full min-h-20 w-px bg-slate-200" />
                )}
              </div>

              <div
                className={`flex-1 ${
                  isLast ? "pb-0" : "pb-6"
                }`}
              >
                <div className="rounded-xl border bg-white p-4">
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div>
                      <p className="text-sm font-semibold text-slate-900">
                        {item.corte}
                      </p>

                      <p className="mt-1 text-sm text-slate-600">
                        {item.tipo}
                      </p>
                    </div>

                    <span
                      className={`rounded-full px-2.5 py-1 text-xs font-medium ${statusStyles.badge}`}
                    >
                      {item.clasificacion}
                    </span>
                  </div>

                  <div className="mt-4 grid gap-3 sm:grid-cols-2">
                    <div>
                      <p className="text-[10px] font-semibold uppercase tracking-wide text-slate-400">
                        Asignatura
                      </p>

                      <p className="mt-1 text-sm text-slate-700">
                        {item.asignatura || "Sin registro"}
                      </p>
                    </div>

                    <div>
                      <p className="text-[10px] font-semibold uppercase tracking-wide text-slate-400">
                        Grupo
                      </p>

                      <p className="mt-1 text-sm text-slate-700">
                        {item.grupo || "Sin registro"}
                      </p>
                    </div>
                  </div>

                  <div className="mt-4 flex items-center justify-between border-t pt-3">
                    <span className="text-xs text-muted-foreground">
                      {item.servicio}
                    </span>

                    <span className="text-sm font-semibold text-slate-900">
                      {item.puntaje.toFixed(1)} pts
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}