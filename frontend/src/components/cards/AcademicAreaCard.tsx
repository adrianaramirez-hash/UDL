import { SIACard } from "@/components/cards/SIACard"

type AcademicAreaCardProps = {
  title: string
  value: string
  status: string
  description: string
  trend?: string
}

export function AcademicAreaCard({
  title,
  value,
  status,
  description,
  trend,
}: AcademicAreaCardProps) {
  const isAlert =
    status.toLowerCase().includes("atención") ||
    status.toLowerCase().includes("crítico")

  return (
    <SIACard interactive className="cursor-pointer p-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-400">
            ÁREA ACADÉMICA
          </p>

          <h3 className="mt-2 text-lg font-semibold text-slate-900">
            {title}
          </h3>
        </div>

        <span
          className={`rounded-full px-2.5 py-1 text-[10px] font-medium ${
            isAlert
              ? "bg-amber-50 text-amber-700"
              : "bg-emerald-50 text-emerald-700"
          }`}
        >
          {status}
        </span>
      </div>

      <div className="mt-6 flex items-end gap-3">
        <span className="sia-heading text-4xl">
          {value}
        </span>

        {trend && (
          <span className="mb-1 text-xs text-muted-foreground">
            {trend}
          </span>
        )}
      </div>

      <p className="mt-4 text-sm leading-6 text-muted-foreground">
        {description}
      </p>

      <div className="mt-6 border-t border-slate-100 pt-4">
        <span className="text-sm font-medium text-primary">
          Ver detalle →
        </span>
      </div>
    </SIACard>
  )
}