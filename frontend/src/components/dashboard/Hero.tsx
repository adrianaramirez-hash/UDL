import { useAuth } from "@/auth/session"

export function Hero({ institutional }: { institutional: boolean }) {
  const { session } = useAuth()
  const user = session.status === "authenticated" ? session.user : null
  const name = user?.nombre?.trim()
  const firstName = name === user?.email ? "" : name?.split(/\s+/, 1)[0]
  const role = user?.nombre_rol?.trim()

  return (
    <section className="mb-10">
      <p className="sia-eyebrow">
        {institutional ? "CENTRO DE DECISIONES" : "INICIO"}
      </p>

      <h1 className="mt-2 text-6xl sia-heading">
        {firstName ? `Buenos días, ${firstName}.` : "Buenos días."}
      </h1>

      {role && <p className="mt-2 text-lg font-medium text-muted-foreground">{role}</p>}

      <p className="mt-3 max-w-3xl text-lg text-muted-foreground">
        {institutional
          ? "El Asesor SIA preparó un resumen estratégico de la situación académica institucional."
          : user?.alcance.tipo === "asignado"
            ? "Consulta los resultados y herramientas disponibles para las áreas que tienes asignadas."
            : "Consulta las herramientas disponibles para tu cuenta."}
      </p>
    </section>
  )
}
