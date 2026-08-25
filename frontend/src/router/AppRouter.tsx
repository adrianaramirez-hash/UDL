import { BrowserRouter, Route, Routes } from "react-router-dom"

import Dashboard from "@/pages/Dashboard"
import CalidadAcademica from "@/pages/CalidadAcademica"
import ObservacionClases from "@/pages/ObservacionClases"
import EncuestaCalidad from "@/pages/EncuestaCalidad"

export function AppRouter() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/calidad-academica" element={<CalidadAcademica />} />
        <Route path="/observacion-clases" element={<ObservacionClases />} />
        <Route path="/encuesta-calidad" element={<EncuestaCalidad />} />
      </Routes>
    </BrowserRouter>
  )
}
