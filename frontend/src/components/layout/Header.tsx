export function Header() {
  return (
    <header className="sticky top-0 z-20 flex h-16 items-center justify-between border-b border-slate-200 bg-background/95 px-5 backdrop-blur md:px-8">
      <p className="text-xs text-slate-500">
        SIA / Centro de Decisiones
      </p>

      <div className="flex items-center gap-3">
        <button
          type="button"
          aria-label="Notificaciones"
          className="flex h-9 w-9 items-center justify-center rounded-full border border-slate-200 bg-white text-slate-600 transition hover:bg-slate-50"
        >
          ◇
        </button>

        <div
          className="flex h-9 w-9 items-center justify-center rounded-full bg-primary text-xs font-semibold text-primary-foreground"
          title="Dirección General"
        >
          DG
        </div>
      </div>
    </header>
  )
}