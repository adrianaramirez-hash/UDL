import { Header } from "@/components/layout/Header"
import { Sidebar } from "@/components/layout/Sidebar"

export default function CalidadAcademica() {
  return (
    <div className="flex min-h-screen bg-background">
      <Sidebar />

      <main className="flex flex-1 flex-col">
        <Header />

        <div className="mx-auto w-full max-w-7xl px-8 py-8">
          <p className="sia-eyebrow">CALIDAD ACADÉMICA</p>

          <h1 className="mt-2 text-5xl sia-heading">
            Calidad Académica
          </h1>

          <p className="mt-3 text-muted-foreground">
            Aquí integraremos observación de clases, evaluación docente,
            capacitación docente, índice de reprobación y bajas.
          </p>
        </div>
      </main>
    </div>
  )
}