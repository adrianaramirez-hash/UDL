import { BrowserRouter, Route, Routes } from "react-router-dom"

import { AuthProvider } from "@/auth/AuthProvider"
import { LoginPage, ProtectedRoute } from "@/auth/ProtectedRoute"
import Dashboard from "@/pages/Dashboard"
import CalidadAcademica from "@/pages/CalidadAcademica"
import ObservacionClases from "@/pages/ObservacionClases"
import EncuestaCalidad from "@/pages/EncuestaCalidad"
import { moduleRoutes } from "./modules"

export function AppRouter() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
          <Route path="/calidad-academica" element={<ProtectedRoute permission={moduleRoutes.observacion.permission}><CalidadAcademica /></ProtectedRoute>} />
          <Route path={moduleRoutes.observacion.path} element={<ProtectedRoute permission={moduleRoutes.observacion.permission}><ObservacionClases /></ProtectedRoute>} />
          <Route path={moduleRoutes.calidad.path} element={<ProtectedRoute permission={moduleRoutes.calidad.permission}><EncuestaCalidad /></ProtectedRoute>} />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  )
}
