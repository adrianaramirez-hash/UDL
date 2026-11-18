import { createContext, useContext } from "react"

export type User = {
  nombre: string
  email: string
  rol: string
  nombre_rol: string
  modulos: string[]
  permisos: string[]
  alcance: { tipo: "global" | "asignado" | "sin_acceso"; servicios: string[] }
}

export type Session = { status: "loading" | "anonymous" | "forbidden" | "error" }
  | { status: "authenticated"; user: User }

export const AuthContext = createContext<{
  session: Session
  logout: () => Promise<void>
} | null>(null)

export function useAuth() {
  const auth = useContext(AuthContext)
  if (!auth) throw new Error("AuthProvider debe envolver la aplicación")
  return auth
}
