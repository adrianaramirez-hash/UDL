export type CalidadFilterSelection = {
  periodo?: string
  modalidad?: string
  servicio?: string
  historico?: boolean
}

export type CalidadQuestion = {
  pregunta_id: string
  seccion_id: string
  seccion_nombre: string
  texto_corto: string
  tipo_dato: string
  persona_evaluada: string
  rol_evaluado: string
  indice: number | null
  n: number
  favorable_pct: number | null
  neutral_pct: number | null
  desfavorable_pct: number | null
  entra_seccion: boolean
  entra_general: boolean
}

export type CalidadSection = {
  seccion_id: string
  seccion_nombre: string
  indice: number | null
  preguntas_con_datos: number
  preguntas_formulario_total: number
  reactivos_indice_total: number
  entra_general: boolean
}

export type CalidadSectionCareerModality = {
  seccion_id: string
  seccion_nombre: string
  carrera_base: string
  modalidad_academica: string
  indice: number | null
  encuestas: number
  preguntas_con_datos: number
  tiene_datos: boolean
}

export type CalidadSectionQuestionByCareerModality = {
  seccion_id: string
  seccion_nombre: string
  carrera_base: string
  modalidad_academica: string
  pregunta_id: string
  texto_corto: string
  tipo_dato: string
  indice: number | null
  n: number
  favorable_pct: number | null
  neutral_pct: number | null
  desfavorable_pct: number | null
  entra_seccion: boolean
}

export type CalidadSectionComparisons = {
  por_carrera_modalidad: CalidadSectionCareerModality[]
  preguntas_por_carrera_modalidad: CalidadSectionQuestionByCareerModality[]
}

export type CalidadStaffCareerBase = {
  seccion_id: string
  carrera_base: string
  indice: number | null
  indice_mostrable: number | null
  personas_evaluadas: number
  evaluaciones: number
  modalidades: string[]
  requiere_modalidad: boolean
}

export type CalidadStaffCareerModality = {
  seccion_id: string
  carrera_base: string
  modalidad_academica: string
  indice: number | null
  personas_evaluadas: number
  personas_asignadas: number
  evaluaciones: number
  tiene_datos: boolean
}

export type CalidadStaffPersonByCareerModality = {
  seccion_id: string
  carrera_base: string
  modalidad_academica: string
  persona_evaluada: string
  rol_evaluado: string
  indice: number | null
  preguntas_con_datos: number
  evaluaciones: number
}

export type CalidadStaffQuestionByCareerModality = {
  seccion_id: string
  carrera_base: string
  modalidad_academica: string
  persona_evaluada: string
  rol_evaluado: string
  pregunta_id: string
  texto_corto: string
  tipo_dato: string
  indice: number | null
  n: number
  favorable_pct: number | null
  neutral_pct: number | null
  desfavorable_pct: number | null
}

export type CalidadStaffCareer = {
  seccion_id: string
  servicio_programa: string
  indice: number | null
  personas_evaluadas: number
  evaluaciones: number
}

export type CalidadStaffPersonByCareer = {
  seccion_id: string
  servicio_programa: string
  persona_evaluada: string
  rol_evaluado: string
  indice: number | null
  preguntas_con_datos: number
  evaluaciones: number
}

export type CalidadStaffQuestionByCareer = {
  seccion_id: string
  servicio_programa: string
  persona_evaluada: string
  rol_evaluado: string
  pregunta_id: string
  texto_corto: string
  tipo_dato: string
  indice: number | null
  n: number
  favorable_pct: number | null
  neutral_pct: number | null
  desfavorable_pct: number | null
}

export type CalidadStaffRole = {
  seccion_id: string
  rol_evaluado: string
  indice: number | null
  personas_evaluadas: number
  evaluaciones: number
}

export type CalidadDirectionCareerModality = {
  seccion_id: string
  carrera_base: string
  modalidad_academica: string
  indice: number | null
  encuestas: number
  encuestas_total: number
  encuestas_ruta: number
  preguntas_con_datos: number
  tiene_datos: boolean
  aplica_direccion: boolean
  estado: "con_datos" | "sin_datos" | "no_aplica"
}

export type CalidadDirectionQuestion = {
  seccion_id: string
  carrera_base: string
  modalidad_academica: string
  pregunta_id: string
  texto_corto: string
  indice: number | null
  n: number
  favorable_pct: number | null
  neutral_pct: number | null
  desfavorable_pct: number | null
}

export type CalidadDirectionComparisons = {
  por_carrera_modalidad: CalidadDirectionCareerModality[]
  preguntas_por_carrera_modalidad: CalidadDirectionQuestion[]
}

export type CalidadStaffComparisons = {
  asignaciones?: CalidadStaffPersonByCareerModality[]
  por_carrera_base: CalidadStaffCareerBase[]
  por_carrera_modalidad: CalidadStaffCareerModality[]
  personas_por_carrera_modalidad: CalidadStaffPersonByCareerModality[]
  preguntas_persona_por_carrera_modalidad: CalidadStaffQuestionByCareerModality[]

  por_carrera: CalidadStaffCareer[]
  personas_por_carrera: CalidadStaffPersonByCareer[]
  preguntas_persona_por_carrera: CalidadStaffQuestionByCareer[]
  por_rol: CalidadStaffRole[]
}

export type CalidadPerson = {
  seccion_id: string
  seccion_nombre: string
  persona_evaluada: string
  rol_evaluado: string
  indice: number | null
  preguntas_con_datos: number
  evaluaciones: number
}

export type CalidadProfileDistributionItem = {
  valor: string
  cantidad: number
  porcentaje: number
}

export type CalidadProfileDimension = {
  total_encuestas: number
  con_dato: number
  sin_dato: number
  cobertura_pct: number
  distribucion: CalidadProfileDistributionItem[]
}

export type CalidadProfile = {
  total_encuestas: number
  servicio_programa: CalidadProfileDimension
  modalidad_detalle: CalidadProfileDimension
  grado: CalidadProfileDimension
  ciclo_escolar: CalidadProfileDimension
  turno: CalidadProfileDimension
}

export type CalidadComment = {
  categoria_cualitativa: string
  incluir_resumen: boolean
  polaridad_base: string | null
  texto_informativo: boolean
  fila_origen: number | null
  servicio_programa: string
  seccion_id: string
  seccion_nombre: string
  pregunta_id: string
  texto_corto: string
  persona_evaluada: string
  rol_evaluado: string
  texto: string
}

export type CalidadQualitativeSummary = {
  seccion_id: string
  seccion_nombre: string
  total: number
  positivos: number
  oportunidades: number
  incidencias: number
  sin_clasificar: number
}

export type CalidadFilters = {
  periodos: string[]
  modalidades: string[]
  servicios: string[]
  seleccion: {
    periodo: string | null
    modalidad: string | null
    servicio: string | null
  }
}

export type CalidadSummary = {
  encuestas: number
  perfil_respuestas: CalidadProfile
  respuestas_normalizadas: number
  respuestas_kpi_validas: number
  preguntas_con_datos: number
  indice_general: number | null
  secciones_general: number
  secciones: CalidadSection[]
  preguntas: CalidadQuestion[]
  comentarios: CalidadComment[]
  resumen_cualitativo: CalidadQualitativeSummary[]
  comparativos_personal: CalidadStaffComparisons
  comparativos_direccion: CalidadDirectionComparisons
  comparativos_secciones: CalidadSectionComparisons
  personas: CalidadPerson[]
  qa_total: number
  fila_inicio: number
  fila_fin: number
  fuente_id: string
  periodo_id: string
  version_instrumento: string
  filtros: CalidadFilters
}


export async function getCalidadSummary(
  filtros: CalidadFilterSelection = {},
  signal?: AbortSignal,
): Promise<CalidadSummary> {
  const params = new URLSearchParams()

  if (filtros.periodo) {
    params.set("periodo", filtros.periodo)
  }

  if (filtros.modalidad) {
    params.set("modalidad", filtros.modalidad)
  }

  if (filtros.servicio) {
    params.set("servicio", filtros.servicio)
  }

  if (filtros.historico) {
    params.set("historico", "true")
  }

  const query = params.toString()

  const url = query
    ? `/api/encuesta-calidad/preview/resumen?${query}`
    : `/api/encuesta-calidad/preview/resumen`

  const response = await fetch(url, { signal })

  if (!response.ok) {
    throw new Error(
      "No fue posible obtener el resumen de Encuesta de Calidad.",
    )
  }

  return response.json()
}
