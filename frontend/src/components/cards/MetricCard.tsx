import { SIACard } from "@/components/cards/SIACard"

type MetricCardProps = {
  label: string
  value: string
  change?: string
  status?: string
  tooltip?: string
  onClick?: () => void
  active?: boolean
  loading?: boolean
}

export function MetricCard({
  label,
  value,
  change,
  status,
  tooltip,
  onClick,
  active = false,
  loading = false,
}: MetricCardProps) {
  const isNegative = change?.startsWith("-")
  const isInteractive = Boolean(onClick)

  if (loading) {
    return (
      <SIACard className="p-6">
        <div className="animate-pulse">
          <div className="h-3 w-24 rounded bg-slate-200" />
          <div className="mt-4 h-8 w-16 rounded bg-slate-200" />
          <div className="mt-4 h-3 w-32 rounded bg-slate-100" />
        </div>
      </SIACard>
    )
  }

  const content = (
    <SIACard
      className={`relative p-6 transition ${
        isInteractive
          ? "cursor-pointer hover:-translate-y-0.5 hover:shadow-md"
          : ""
      } ${
        active
          ? "ring-2 ring-primary/20"
          : ""
      }`}
    >
      <div className="flex items-start justify-between gap-3">
        <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-400">
          {label}
        </p>

        {tooltip && (
          <div className="group relative z-20">
            <span
              tabIndex={0}
              aria-label={`Información sobre ${label}`}
              className="flex h-5 w-5 cursor-help items-center justify-center rounded-full border border-slate-300 bg-white text-[11px] font-bold text-slate-500 transition hover:border-primary hover:text-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
            >
              ?
            </span>

            <div className="pointer-events-none absolute right-0 top-7 z-50 hidden w-64 rounded-xl bg-slate-900 px-3 py-2 text-left text-xs font-normal leading-5 text-white shadow-xl group-hover:block group-focus-within:block">
              {tooltip}
            </div>
          </div>
        )}
      </div>

      <div className="mt-2 flex items-center gap-3">
        <span className="text-2xl font-semibold">
          {value}
        </span>

        {change && (
          <span
            className={`rounded-full px-2 py-1 text-[10px] font-medium ${
              isNegative
                ? "bg-red-50 text-red-600"
                : "bg-emerald-50 text-emerald-700"
            }`}
          >
            {change}
          </span>
        )}
      </div>

      {status && (
        <p className="mt-3 text-xs text-muted-foreground">
          {status}
        </p>
      )}

      {isInteractive && (
        <p className="mt-4 text-[11px] font-medium text-primary">
          Ver detalle →
        </p>
      )}
    </SIACard>
  )

  if (!isInteractive) {
    return content
  }

  return (
    <button
      type="button"
      onClick={onClick}
      className="w-full text-left"
    >
      {content}
    </button>
  )
}