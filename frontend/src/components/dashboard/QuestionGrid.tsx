import { useAuth } from "@/auth/session"
import { QuestionCard } from "@/components/cards/QuestionCard"
import { moduleRoutes } from "@/router/modules"

export function QuestionGrid() {
  const { session } = useAuth()
  if (session.status !== "authenticated") return null

  const { permisos } = session.user
  const modules = Object.values(moduleRoutes).filter(item => permisos.includes(item.permission))
  if (modules.length === 0) return null

  return (
    <section className="mb-10" aria-labelledby="modules-title">
      <p className="sia-eyebrow">ÁREAS DE ANÁLISIS</p>
      <h2 id="modules-title" className="mb-8 mt-2 text-3xl sia-heading">
        ¿Qué necesitas resolver?
      </h2>
      <div className="grid gap-6 md:grid-cols-2 xl:grid-cols-3">
        {modules.map(({ path, title, description }) => (
          <QuestionCard key={path} to={path} title={title} description={description} />
        ))}
      </div>
    </section>
  )
}
