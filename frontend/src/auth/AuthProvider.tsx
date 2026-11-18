import { useEffect, useState, type ReactNode } from "react"
import { AuthContext, type Session, type User } from "./session"

export function AuthProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session>({ status: "loading" })

  useEffect(() => {
    const controller = new AbortController()
    async function loadSession() {
      try {
        const response = await fetch("/api/auth/me", { signal: controller.signal })
        if (controller.signal.aborted) return
        if (response.status === 401) setSession({ status: "anonymous" })
        else if (response.status === 403) setSession({ status: "forbidden" })
        else if (response.ok) {
          const user: User = await response.json()
          if (!controller.signal.aborted) setSession({ status: "authenticated", user })
        } else setSession({ status: "error" })
      } catch {
        if (!controller.signal.aborted) setSession({ status: "error" })
      }
    }
    void loadSession()
    return () => controller.abort()
  }, [])

  async function logout() {
    const response = await fetch("/api/auth/logout", { method: "POST" })
    if (!response.ok) throw new Error("No fue posible cerrar la sesión")
    setSession({ status: "anonymous" })
  }

  return <AuthContext.Provider value={{ session, logout }}>{children}</AuthContext.Provider>
}
