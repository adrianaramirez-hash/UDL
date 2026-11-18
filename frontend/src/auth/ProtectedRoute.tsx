import { type ReactNode, useState } from "react"
import { Link, Navigate } from "react-router-dom"
import { useAuth } from "./session"

function AccessScreen({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <main className="mx-auto flex min-h-screen max-w-md flex-col justify-center gap-5 px-6">
      <h1 className="sia-heading text-3xl">{title}</h1>
      {children}
    </main>
  )
}

export function LogoutButton() {
  const { logout } = useAuth()
  const [pending, setPending] = useState(false)
  const [error, setError] = useState(false)
  async function submit() {
    setPending(true)
    setError(false)
    try {
      await logout()
    } catch {
      setError(true)
    } finally {
      setPending(false)
    }
  }
  return (
    <div>
      <button type="button" disabled={pending} onClick={() => void submit()}
        className="rounded-lg border bg-background px-3 py-2 text-sm hover:bg-muted">
        {pending ? "Cerrando sesión…" : "Cerrar sesión"}
      </button>
      {error && <p role="alert" className="text-sm text-red-700">No fue posible cerrar la sesión.</p>}
    </div>
  )
}

function AccessDenied({ active }: { active: boolean }) {
  return (
    <AccessScreen title="Acceso no autorizado">
      {active && <Link to="/" className="text-sm underline">Volver al inicio</Link>}
      <LogoutButton />
    </AccessScreen>
  )
}

export function LoginPage() {
  const { session } = useAuth()
  if (session.status === "loading") return <AccessScreen title="Verificando sesión…" />
  if (session.status === "authenticated") return <Navigate to="/" replace />
  if (session.status === "forbidden") return <AccessDenied active={false} />
  return (
    <AccessScreen title="Iniciar sesión">
      <p className="text-muted-foreground">Universidad de Londres</p>
      {session.status === "error" && <p role="alert">No se pudo verificar la sesión. Intenta nuevamente.</p>}
      <a href="/api/auth/login"
        className="w-fit rounded-lg bg-primary px-4 py-2 text-primary-foreground">
        Iniciar sesión con Google
      </a>
    </AccessScreen>
  )
}

export function ProtectedRoute({ permission, children }: { permission?: string; children: ReactNode }) {
  const { session } = useAuth()
  if (session.status === "loading") return <AccessScreen title="Verificando sesión…" />
  if (session.status === "error") return (
    <AccessScreen title="No se pudo verificar la sesión">
      <button type="button" onClick={() => window.location.reload()} className="w-fit underline">Reintentar</button>
    </AccessScreen>
  )
  if (session.status === "anonymous") return <Navigate to="/login" replace />
  if (session.status !== "authenticated") return <AccessDenied active={false} />
  if (permission && !session.user.permisos.includes(permission)) return <AccessDenied active />
  return children
}
