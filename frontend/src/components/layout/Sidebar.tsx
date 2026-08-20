import { NavLink } from "react-router-dom";

const menuItems = [
  {
    title: "Calidad Académica",
    path: "/calidad-academica",
  },
  {
    title: "Evaluación del Aprendizaje",
    path: "/evaluacion-aprendizaje",
  },
  {
    title: "Titulación y Egreso",
    path: "/titulacion-egreso",
  },
  {
    title: "Informes Ejecutivos",
    path: "/informes",
  },
];

export function Sidebar() {
  return (
    <aside className="hidden w-[250px] shrink-0 border-r border-sidebar-border bg-sidebar lg:flex lg:flex-col">
      <div className="border-b border-sidebar-border px-6 py-6">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary text-sm font-bold text-primary-foreground">
            SIA
          </div>

          <div>
            <p className="text-sm font-semibold text-slate-900">
              SIA Intelligence
            </p>
            <p className="text-xs text-slate-500">
              Universidad de Londres
            </p>
          </div>
        </div>
      </div>

      <nav className="flex-1 px-4 py-6">

        <p className="mb-2 px-3 text-[10px] font-semibold tracking-[0.18em] text-slate-400">
          PRINCIPAL
        </p>

        <NavLink
          to="/"
          className={({ isActive }) =>
            `mb-6 flex w-full items-center rounded-xl px-4 py-3 text-left text-sm font-medium transition ${
              isActive
                ? "bg-sidebar-accent text-sidebar-accent-foreground"
                : "text-slate-600 hover:bg-slate-100"
            }`
          }
        >
          Centro de Decisiones
        </NavLink>

        <p className="mb-2 px-3 text-[10px] font-semibold tracking-[0.18em] text-slate-400">
          ÁREAS DE ANÁLISIS
        </p>

        <div className="space-y-1">
          {menuItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                `block rounded-lg px-4 py-2.5 text-sm transition ${
                  isActive
                    ? "bg-sidebar-accent font-medium text-sidebar-accent-foreground"
                    : "text-slate-600 hover:bg-slate-100 hover:text-slate-900"
                }`
              }
            >
              {item.title}
            </NavLink>
          ))}
        </div>

      </nav>

      <div className="border-t border-sidebar-border p-4">
        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <p className="text-xs text-slate-500">
            Sesión activa
          </p>

          <p className="mt-1 text-sm font-medium text-slate-800">
            Dirección General
          </p>
        </div>
      </div>
    </aside>
  );
}