import { useAuth } from "@/auth/session"
import { LogoutButton } from "@/auth/ProtectedRoute"
import { Link } from "react-router-dom"

export function Header() {
  const { session } = useAuth()
  const user = session.status === "authenticated" ? session.user : null
  return (
    <header className="sticky top-0 z-20 flex h-16 items-center justify-between border-b border-slate-200 bg-background/95 px-5 backdrop-blur md:px-8">
      <Link to="/" className="text-xs text-slate-500 hover:underline">
        SIA / Inicio
      </Link>

      <div className="flex items-center gap-3">
        <button
          type="button"
          aria-label="Notificaciones"
          className="flex h-9 w-9 items-center justify-center rounded-full border border-slate-200 bg-white text-slate-600 transition hover:bg-slate-50"
        >
          ◇
        </button>

        {user && <>
          <div className="text-right text-xs">
            <p className="font-semibold">{user.nombre}</p>
            <p className="text-muted-foreground">{user.nombre_rol}</p>
          </div>
          <LogoutButton />
        </>}
      </div>
    </header>
  )
}
