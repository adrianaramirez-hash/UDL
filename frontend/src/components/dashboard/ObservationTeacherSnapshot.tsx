import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"

type TeacherHistoryItem = {
  corte: string
  servicio: string
  grupo: string
  asignatura: string
  tipo: string
  puntaje: number
  clasificacion: string
}

type ObservationTeacherSnapshotProps = {
  promedio: number
  clasificacion: string
  historial: TeacherHistoryItem[]
}

export function ObservationTeacherSnapshot({
  promedio,
  clasificacion,
  historial,
}: ObservationTeacherSnapshotProps) {
  const statusStyles = {
    Consolidado: {
      badge: "bg-emerald-50 text-emerald-700 border-emerald-200",
      dot: "bg-emerald-500",
      label: "Desempeño consolidado",
    },
    "En proceso": {
      badge: "bg-amber-50 text-amber-700 border-amber-200",
      dot: "bg-amber-500",
      label: "Requiere seguimiento",
    },
    "No consolidado": {
      badge: "bg-red-50 text-red-700 border-red-200",
      dot: "bg-red-500",
      label: "Atención prioritaria",
    },
  }

  const currentStatus =
    statusStyles[
      clasificacion as keyof typeof statusStyles
    ] ?? statusStyles["En proceso"]

  const chartData = historial.map((item) => ({
    corte: item.corte,
    puntaje: item.puntaje,
  }))

  return (
    <div className="mt-6 grid gap-4 xl:grid-cols-[240px_1fr]">
      <div
        className={`rounded-2xl border p-5 ${currentStatus.badge}`}
      >
        <p className="text-[10px] font-semibold uppercase tracking-[0.16em]">
          Estado actual
        </p>

        <div className="mt-5 flex items-center gap-3">
          <span
            className={`h-4 w-4 shrink-0 rounded-full ${currentStatus.dot}`}
          />

          <div>
            <p className="text-xl font-semibold">
              {clasificacion}
            </p>

            <p className="mt-1 text-xs opacity-80">
              {currentStatus.label}
            </p>
          </div>
        </div>

        <div className="mt-6 border-t border-current/10 pt-4">
          <p className="text-xs opacity-70">
            Promedio
          </p>

          <p className="mt-1 text-4xl font-semibold">
            {promedio.toFixed(1)}
          </p>
        </div>
      </div>

      <div className="rounded-2xl border bg-white p-5">
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-400">
            EVOLUCIÓN
          </p>

          <h4 className="mt-1 text-lg font-semibold text-slate-900">
            Historial de desempeño
          </h4>

          <p className="mt-1 text-sm text-muted-foreground">
            Puntaje obtenido en sus observaciones registradas.
          </p>
        </div>

        {chartData.length > 1 ? (
          <div className="mt-5 h-[190px]">
            <ResponsiveContainer
              width="100%"
              height="100%"
            >
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />

                <XAxis
                  dataKey="corte"
                  tick={{ fontSize: 10 }}
                />

                <YAxis
                  domain={["dataMin - 5", "dataMax + 5"]}
                  tick={{ fontSize: 10 }}
                />

                <Tooltip />

                <Line
                  type="monotone"
                  dataKey="puntaje"
                  stroke="#0f766e"
                  strokeWidth={3}
                  dot={{ r: 4 }}
                  activeDot={{ r: 6 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="mt-5 flex min-h-[190px] items-center justify-center rounded-xl bg-slate-50 p-6 text-center">
            <div>
              <p className="text-sm font-medium text-slate-700">
                Aún no hay tendencia histórica
              </p>

              <p className="mt-2 text-sm text-muted-foreground">
                Se requiere más de una observación para identificar
                evolución.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}