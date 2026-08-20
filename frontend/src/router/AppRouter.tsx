import { BrowserRouter, Route, Routes } from "react-router-dom"

import Dashboard from "@/pages/Dashboard"
import CalidadAcademica from "@/pages/CalidadAcademica"
import ObservacionClases from "@/pages/ObservacionClases"

export function AppRouter() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/calidad-academica" element={<CalidadAcademica />} />
        <Route path="/observacion-clases" element={<ObservacionClases />} />
      </Routes>
    </BrowserRouter>
  )
}