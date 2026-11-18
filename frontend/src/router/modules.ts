export const moduleRoutes = {
  calidad: {
    title: "Encuesta de Calidad",
    description: "Consulta resultados, comparativos y percepción institucional.",
    path: "/encuesta-calidad",
    permission: "encuesta_calidad.ver",
  },
  observacion: {
    title: "Observación de Clases",
    description: "Consulta el seguimiento y los resultados de la observación docente.",
    path: "/observacion-clases",
    permission: "observacion_clases.ver",
  },
} as const
