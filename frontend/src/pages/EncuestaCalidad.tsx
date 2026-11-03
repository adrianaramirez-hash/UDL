import { useEffect, useState } from "react"
import {
  Bar,
  BarChart,
  Cell,
  CartesianGrid,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"

import { Header } from "@/components/layout/Header"
import { Sidebar } from "@/components/layout/Sidebar"
import {
  getCalidadSummary,
  type CalidadFilterSelection,
  type CalidadQuestion,
  type CalidadSummary,
} from "@/services/calidadService"


type DirectionPopulationRef = {
  carrera_base: string
  modalidad_academica: string
  indice: number | null
  encuestas: number
  preguntas_con_datos: number
}

type DirectionDirector = {
  persona_evaluada: string
  rol_evaluado: string
  indice: number | null
  evaluaciones: number
  criterios_evaluados: number
  carreras_a_cargo: number
  poblaciones: DirectionPopulationRef[]
}

type DirectionDirectorQuestion = {
  seccion_id: string
  persona_evaluada: string
  rol_evaluado: string
  pregunta_id: string
  texto_corto: string
  indice: number | null
  n: number
  favorable_pct: number | null
  neutral_pct: number | null
  desfavorable_pct: number | null
}

type DirectionComparisonsExtended = CalidadSummary["comparativos_direccion"] & {
  por_director?: DirectionDirector[]
  preguntas_por_director?: DirectionDirectorQuestion[]
}

type CalidadSummaryWithExport = CalidadSummary & {
  respuestas_export?: Array<Record<string, unknown>>
}

type QualitativeSummaryItem = {
  seccion_id: string
  seccion_nombre: string
  total: number
  positivos: number
  oportunidades: number
  incidencias: number
  sin_clasificar: number
}

type InstitutionalServiceQuestion = CalidadQuestion & {
  historical_index?: number | null
}

function institutionalServiceDisplayName(question: Pick<CalidadQuestion, "texto_corto">) {
  const label = String(question.texto_corto ?? "").trim()
  return label.replace(/^Servicio institucional\s+\d+$/i, "Servicio institucional")
}

const INSTITUTIONAL_SERVICE_TOPIC_RULES: Array<{
  match: string[]
  keywords: string[]
}> = [
  {
    match: ["cafeteria"],
    keywords: ["cafeteria", "comida", "alimento", "alimentos", "menu", "precio", "precios", "comedor", "bebida", "bebidas"],
  },
  {
    match: ["wifi", "internet"],
    keywords: ["wifi", "wi fi", "internet", "red", "conexion", "conectividad", "senal"],
  },
  {
    match: ["sanitario", "bano"],
    keywords: ["sanitario", "sanitarios", "bano", "banos", "toilet", "limpieza de los banos"],
  },
  {
    match: ["biblioteca"],
    keywords: ["biblioteca", "libro", "libros", "acervo", "prestamo", "bibliotec"],
  },
  {
    match: ["laboratorio de computo", "computo"],
    keywords: ["laboratorio", "computo", "computadora", "computadoras", "pc", "equipo de computo"],
  },
  {
    match: ["salon", "salones"],
    keywords: ["salon", "salones", "aula", "aulas", "pizarron", "mesabanco", "silla", "sillas"],
  },
  {
    match: ["hyflex", "equipo didactico"],
    keywords: ["hyflex", "canon", "proyector", "pantalla", "television", "equipo didactico", "audio", "bocina"],
  },
  {
    match: ["instalaciones especificas"],
    keywords: ["instalaciones especificas", "taller", "talleres", "cocina", "foro", "cabina", "laboratorio especializado", "equipo especializado"],
  },
  {
    match: ["instalaciones del plantel", "instalaciones"],
    keywords: ["instalaciones", "plantel", "alumbrado", "pintura", "ventilacion", "mantenimiento", "edificio", "infraestructura"],
  },
  {
    match: ["aire libre"],
    keywords: ["aire libre", "patio", "terraza", "jardin", "areas verdes", "espacio exterior", "espacios exteriores"],
  },
]

function getInstitutionalServiceKeywords(label: string) {
  const normalized = normalizeAcademicLabel(label)
  const rules = INSTITUTIONAL_SERVICE_TOPIC_RULES.filter((rule) =>
    rule.match.some((term) => normalized.includes(normalizeAcademicLabel(term))),
  )

  const generic = normalized
    .split(/\s+/)
    .filter((token) => token.length >= 5)
    .filter((token) => !["equipo", "servicio", "plantel", "estudiantes", "clase", "clases"].includes(token))

  return Array.from(
    new Set([
      ...rules.flatMap((rule) => rule.keywords.map(normalizeAcademicLabel)),
      ...generic,
    ]),
  )
}

function commentsForInstitutionalServiceQuestion(
  question: CalidadQuestion,
  comments: CalidadSummary["comentarios"],
) {
  const keywords = getInstitutionalServiceKeywords(question.texto_corto || question.pregunta_id)
  if (keywords.length === 0) return []

  return comments.filter((comment) => {
    if (canonicalSectionId(comment.seccion_id) !== "ESC_03_SERVICIOS") return false
    if (!comment.incluir_resumen || !comment.texto_informativo) return false
    if (!isMeaningfulComment(comment.texto ?? "")) return false

    const text = normalizeAcademicLabel(comment.texto ?? "")
    return keywords.some((keyword) => keyword && text.includes(keyword))
  })
}

function findHistoricalInstitutionalQuestion(
  question: CalidadQuestion,
  historical: CalidadSummary | null,
) {
  return findHistoricalComparableQuestion(question, historical)
}

type FacilitiesQuestion = CalidadQuestion & {
  historical_index?: number | null
}

function facilitiesDisplayName(question: Pick<CalidadQuestion, "texto_corto" | "pregunta_id">) {
  const label = normalizeAcademicLabel(question.texto_corto || question.pregunta_id || "")
  if (label.includes("laboratorio") && label.includes("comput")) return "Laboratorio de cómputo"
  if (label.includes("salon") || label.includes("aula")) return "Salones de clase"
  if (label.includes("sanitario") || label.includes("bano")) return "Sanitarios"
  if (label.includes("hyflex") || label.includes("equipo didact")) return "Equipo didáctico HyFlex"
  if (label.includes("instalaciones especific")) return "Instalaciones específicas para el aprendizaje"
  if (label.includes("instalaciones") && label.includes("plantel")) return "Instalaciones del plantel"
  if (label.includes("biblioteca")) return "Biblioteca"
  if (label.includes("cafeteria")) return "Cafetería"
  if (label.includes("aire libre")) return "Espacios al aire libre"
  if (label.includes("wifi") || label.includes("internet")) return "WiFi para estudiantes"
  return question.texto_corto || "Instalación / equipo"
}

function facilitiesCommentsForQuestion(
  question: CalidadQuestion,
  comments: CalidadSummary["comentarios"],
) {
  const keywords = getInstitutionalServiceKeywords(question.texto_corto || question.pregunta_id)
  if (keywords.length === 0) return []
  return comments.filter((comment) => {
    const sectionName = normalizeAcademicLabel(comment.seccion_nombre || "")
    if (!(canonicalSectionId(comment.seccion_id) === "ESC_07_INSTALACIONES" || sectionName.includes("instalaciones y equipo"))) return false
    if (!comment.incluir_resumen || !isMeaningfulComment(comment.texto ?? "")) return false
    const text = normalizeAcademicLabel(comment.texto ?? "")
    return keywords.some((keyword) => keyword && text.includes(keyword))
  })
}

function findHistoricalFacilitiesQuestion(
  question: CalidadQuestion,
  historical: CalidadSummary | null,
) {
  return findHistoricalComparableQuestion(question, historical)
}

function facilitiesSuggestionComments(comments: CalidadSummary["comentarios"]) {
  return comments.filter((comment) => {
    const sectionName = normalizeAcademicLabel(comment.seccion_nombre || "")
    if (!(canonicalSectionId(comment.seccion_id) === "ESC_07_INSTALACIONES" || sectionName.includes("instalaciones y equipo"))) return false
    if (!comment.incluir_resumen || !isMeaningfulComment(comment.texto ?? "")) return false
    return !isPositiveNoSuggestion(comment.texto ?? "")
  })
}


type AcademicServiceQuestion = CalidadQuestion & {
  historical_index?: number | null
}

function academicServiceDisplayName(question: Pick<CalidadQuestion, "texto_corto" | "pregunta_id">) {
  const raw = String(question.texto_corto || question.pregunta_id || "Servicio académico").trim()
  return raw
    .replace(/^Servicio académico\s+\d+$/i, "Servicio académico")
    .replace(/^Servicio academico\s+\d+$/i, "Servicio académico")
}

const ACADEMIC_SERVICE_TOPIC_RULES: Array<{
  match: string[]
  keywords: string[]
}> = [
  {
    match: ["profesores", "asesores", "profesor", "asesor"],
    keywords: ["profesor", "profesores", "maestro", "maestros", "asesor", "asesores", "docente", "docentes"],
  },
  {
    match: ["plan de estudios", "plan estudios"],
    keywords: ["plan de estudios", "plan estudios", "materias", "asignaturas", "curricula", "curricular", "programa academico"],
  },
  {
    match: ["clases presenciales", "presenciales"],
    keywords: ["clase presencial", "clases presenciales", "presencial", "aula", "clases en salon"],
  },
  {
    match: ["zoom", "clases en zoom"],
    keywords: ["zoom", "clase en linea", "clases en linea", "videollamada", "sesion virtual"],
  },
  {
    match: ["practicas academicas", "practicas"],
    keywords: ["practica", "practicas", "practicas academicas", "visita", "visitas", "campo", "salida academica"],
  },
  {
    match: ["certificaciones", "certificacion"],
    keywords: ["certificacion", "certificaciones", "certificado", "certificados", "constancia", "constancias"],
  },
  {
    match: ["plataforma seac en general", "seac en general"],
    keywords: ["seac", "plataforma", "sistema seac", "portal seac"],
  },
  {
    match: ["apps moviles", "app movil", "aplicacion movil"],
    keywords: ["app", "apps", "aplicacion", "aplicaciones", "movil", "android", "iphone", "ios"],
  },
  {
    match: ["facilidad para ingresar", "ingresar"],
    keywords: ["ingresar", "acceso", "entrar", "login", "contrasena", "usuario", "inicio de sesion"],
  },
  {
    match: ["tramites administrativos", "tramites"],
    keywords: ["tramite", "tramites", "administrativo", "administrativos", "solicitud", "solicitudes", "documento", "documentos"],
  },
  {
    match: ["tareas y trabajos", "tareas", "trabajos"],
    keywords: ["tarea", "tareas", "trabajo", "trabajos", "entrega", "entregas", "actividad", "actividades"],
  },
  {
    match: ["aulas virtuales", "contenidos de las aulas virtuales", "contenidos"],
    keywords: ["aula virtual", "aulas virtuales", "contenido", "contenidos", "material", "materiales", "recurso", "recursos"],
  },
]

function getAcademicServiceKeywords(label: string) {
  const normalized = normalizeAcademicLabel(label)
  const rules = ACADEMIC_SERVICE_TOPIC_RULES.filter((rule) =>
    rule.match.some((term) => normalized.includes(normalizeAcademicLabel(term))),
  )

  const generic = normalized
    .split(/\s+/)
    .filter((token) => token.length >= 5)
    .filter((token) => !["servicio", "academico", "academicos", "general", "universidad"].includes(token))

  return Array.from(
    new Set([
      ...rules.flatMap((rule) => rule.keywords.map(normalizeAcademicLabel)),
      ...generic,
    ]),
  )
}

function commentsForAcademicServiceQuestion(
  question: CalidadQuestion,
  comments: CalidadSummary["comentarios"],
) {
  const keywords = getAcademicServiceKeywords(question.texto_corto || question.pregunta_id)
  if (keywords.length === 0) return []

  return comments.filter((comment) => {
    if (canonicalSectionId(comment.seccion_id) !== "ESC_04_ACADEMICOS") return false
    if (!comment.incluir_resumen || !comment.texto_informativo) return false
    if (!isMeaningfulComment(comment.texto ?? "")) return false

    const text = normalizeAcademicLabel(comment.texto ?? "")
    return keywords.some((keyword) => keyword && text.includes(keyword))
  })
}

function findHistoricalAcademicQuestion(
  question: CalidadQuestion,
  historical: CalidadSummary | null,
) {
  return findHistoricalComparableQuestion(question, historical)
}

type LanguageCenterQuestion = CalidadQuestion & {
  historical_index?: number | null
}

type ContinuingEducationQuestion = CalidadQuestion & {
  historical_index?: number | null
}

function languageCenterDisplayName(
  question: Pick<CalidadQuestion, "texto_corto" | "pregunta_id">,
) {
  const raw = String(
    question.texto_corto || question.pregunta_id || "Centro de Idiomas",
  ).trim()

  return raw
    .replace(/^Centro de Idiomas\s+\d+$/i, "Centro de Idiomas")
    .replace(/^Centro de Idiomas\s*[-–—:]\s*/i, "")
}

function findHistoricalLanguageCenterQuestion(
  question: CalidadQuestion,
  historical: CalidadSummary | null,
) {
  return findHistoricalComparableQuestion(question, historical)
}

function getLanguageCenterKeywords(label: string) {
  const normalized = normalizeAcademicLabel(label)

  if (
    normalized.includes("docente") ||
    normalized.includes("profesor")
  ) {
    return [
      "docente",
      "profesor",
      "maestro",
      "teacher",
      "clase",
      "ensenanza",
    ]
  }

  if (
    normalized.includes("oficina") ||
    normalized.includes("servicio")
  ) {
    return [
      "oficina",
      "atencion",
      "servicio",
      "tramite",
      "horario",
      "respuesta",
      "personal",
      "centro de idiomas",
    ]
  }

  if (normalized.includes("clase")) {
    return [
      "clase",
      "curso",
      "dinamica",
      "actividad",
      "material",
      "contenido",
      "horario",
      "ingles",
      "frances",
    ]
  }

  return normalized
    .split(/\s+/)
    .filter((token) => token.length >= 5)
}

function commentsForLanguageCenterQuestion(
  question: CalidadQuestion,
  comments: CalidadSummary["comentarios"],
) {
  const keywords = getLanguageCenterKeywords(
    question.texto_corto || question.pregunta_id,
  )

  return comments.filter((comment) => {
    if (canonicalSectionId(comment.seccion_id) !== "ESC_05_IDIOMAS") {
      return false
    }

    if (!comment.incluir_resumen) return false
    if (!isMeaningfulComment(comment.texto ?? "")) return false

    const normalizedText = normalizeAcademicLabel(comment.texto ?? "")
    return keywords.some(
      (keyword) =>
        keyword &&
        normalizedText.includes(normalizeAcademicLabel(keyword)),
    )
  })
}

function languageCenterCommentKind(
  comment: CalidadSummary["comentarios"][number],
) {
  const label = normalizeAcademicLabel(comment.texto_corto ?? "")
  const category = normalizeAcademicLabel(
    comment.categoria_cualitativa ?? "",
  )
  const text = normalizeAcademicLabel(comment.texto ?? "")

  // Primero interpretamos el sentido del texto. Esto evita que una respuesta
  // favorable escrita dentro de una pregunta de "mejora" se marque como queja.
  const explicitPositivePhrases = [
    "todo esta bien",
    "todo bien",
    "no hay queja",
    "no hay quejas",
    "ninguna queja",
    "sin quejas",
    "nada que mejorar",
    "ninguna sugerencia",
    "ningun problema",
    "no tengo problema",
    "no tengo problemas",
    "no hay algo que me desagrade",
    "no hay nada que me desagrade",
    "no hay nada que no me guste",
    "nada me desagrada",
    "nada me disgusta",
    "no me desagrada nada",
    "no me disgusta nada",
    "no tengo algo que me desagrade",
    "no tengo nada que me desagrade",
    "no tengo nada malo que decir",
    "no tengo comentarios negativos",
    "no se tardan",
    "no tardan",
    "contestan rapido",
    "responden rapido",
    "siempre contestan",
    "siempre responden",
    "atienden bien",
    "buena atencion",
    "me gusta",
    "muy bueno",
    "muy buena",
    "excelente servicio",
    "excelente atencion",
  ]

  if (explicitPositivePhrases.some((phrase) => text.includes(phrase))) {
    return "fortaleza"
  }

  const explicitImprovementPhrases = [
    "se tardan",
    "tardan mucho",
    "no contestan",
    "no responden",
    "casi no contestan",
    "casi no responden",
    "responden enojados",
    "contestan enojados",
    "mala atencion",
    "mal servicio",
    "deberian",
    "deberia",
    "falta",
    "mejorar",
    "pesimo",
    "malo",
  ]

  if (explicitImprovementPhrases.some((phrase) => text.includes(phrase))) {
    return "mejora"
  }

  if (
    label.includes("mas te gusta") ||
    category.includes("positivo") ||
    category.includes("reconocimiento")
  ) {
    return "fortaleza"
  }

  // Solo usamos el origen de la pregunta como respaldo cuando el texto no
  // expresa claramente una valoración positiva o negativa.
  if (
    label.includes("menos te gusta") ||
    label.includes("mejorar") ||
    category.includes("oportunidad") ||
    category.includes("incidencia")
  ) {
    return "mejora"
  }

  return "general"
}


function continuingEducationDisplayName(question: CalidadQuestion) {
  const label = normalizeAcademicLabel(question.texto_corto || question.pregunta_id)
  if (label.includes("calidad") && (label.includes("curso") || label.includes("diplomado"))) return "Calidad de cursos y diplomados"
  if (label.includes("tema") && (label.includes("formacion") || label.includes("profesional"))) return "Pertinencia de los temas"
  if (label.includes("horario") || label.includes("modalidad")) return "Horarios y modalidades"
  if (label.includes("recomendar")) return "Recomendación de programas"
  if (label.includes("conozco") && (label.includes("curso") || label.includes("taller") || label.includes("diplomado"))) return "Conocimiento de la oferta"
  if (label.includes("donde") && label.includes("oferta")) return "Ubicación de la oferta académica"
  if (label.includes("consultar") && label.includes("oferta")) return "Ubicación de la oferta académica"
  return question.texto_corto || "Indicador de Educación Continua"
}

function findHistoricalContinuingEducationQuestion(
  question: CalidadQuestion,
  historical: CalidadSummary | null,
) {
  return findHistoricalComparableQuestion(question, historical)
}

function isPositiveNoSuggestion(textValue: string) {
  const text = normalizeAcademicLabel(textValue)
  const phrases = [
    "todo esta bien", "todo bien", "no hay queja", "no hay quejas", "sin quejas",
    "nada que mejorar", "nada por mejorar", "ninguna sugerencia", "sin sugerencias",
    "ningun problema", "no tengo problema", "no tengo problemas", "ninguno",
    "ninguna", "nada", "esta bien asi", "me parece bien", "no cambiaria nada",
  ]
  return phrases.some((phrase) => text === phrase || text.includes(phrase))
}

function continuingEducationImprovementComments(comments: CalidadSummary["comentarios"]) {
  return comments.filter((comment) => {
    const sectionName = normalizeAcademicLabel(comment.seccion_nombre || "")
    if (!sectionName.includes("educacion continua")) return false
    if (!comment.incluir_resumen || !isMeaningfulComment(comment.texto ?? "")) return false
    const label = normalizeAcademicLabel(comment.texto_corto ?? "")
    if (!(label.includes("mejorar") || label.includes("suger"))) return false
    return !isPositiveNoSuggestion(comment.texto ?? "")
  })
}

function commentsForContinuingEducationQuestion(
  question: CalidadQuestion,
  comments: CalidadSummary["comentarios"],
) {
  const label = normalizeAcademicLabel(question.texto_corto || question.pregunta_id)
  const keywords: string[] = []
  if (label.includes("calidad") || label.includes("curso") || label.includes("diplomado")) keywords.push("calidad", "curso", "diplomado", "taller")
  if (label.includes("tema") || label.includes("formacion")) keywords.push("tema", "contenido", "formacion", "profesional")
  if (label.includes("horario") || label.includes("modalidad")) keywords.push("horario", "modalidad", "tiempo", "sabatino", "semana")
  if (label.includes("recomendar")) keywords.push("recomendar", "recomendaria", "recomendacion")
  if (label.includes("conozco") || label.includes("oferta")) keywords.push("oferta", "conocer", "informacion", "difusion", "publicidad", "correo", "redes", "pagina")
  if (keywords.length === 0) keywords.push(...label.split(/\s+/).filter((token) => token.length >= 5))
  return comments.filter((comment) => {
    const sectionName = normalizeAcademicLabel(comment.seccion_nombre || "")
    if (!sectionName.includes("educacion continua")) return false
    if (!comment.incluir_resumen || !isMeaningfulComment(comment.texto ?? "")) return false
    const text = normalizeAcademicLabel(comment.texto ?? "")
    return keywords.some((keyword) => text.includes(normalizeAcademicLabel(keyword)))
  })
}

function safeFilePart(value: string | null | undefined) {
  return String(value ?? "")
    .trim()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-zA-Z0-9_-]+/g, "_")
    .replace(/^_+|_+$/g, "")
}

function csvCell(value: unknown) {
  const text =
    value === null || value === undefined
      ? ""
      : String(value)

  return `"${text.replace(/"/g, '""')}"`
}

function downloadResponsesCsv(
  summary: CalidadSummaryWithExport,
  filters: CalidadFilterSelection,
) {
  const rows = summary.respuestas_export ?? []

  if (rows.length === 0) {
    window.alert(
      "No hay respuestas para descargar con los filtros actuales.",
    )
    return
  }

  const headers = Object.keys(rows[0])
  const csv = [
    headers.map(csvCell).join(","),
    ...rows.map((row) =>
      headers.map((header) => csvCell(row[header])).join(","),
    ),
  ].join("\r\n")

  // BOM UTF-8 para conservar acentos correctamente al abrir en Excel.
  const blob = new Blob(["\ufeff", csv], {
    type: "text/csv;charset=utf-8;",
  })

  const url = URL.createObjectURL(blob)
  const anchor = document.createElement("a")

  const parts = [
    "Encuesta_Calidad_Respuestas",
    safeFilePart(filters.periodo),
    safeFilePart(filters.modalidad),
    safeFilePart(filters.servicio),
  ].filter(Boolean)

  anchor.href = url
  anchor.download = `${parts.join("_")}.csv`
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  URL.revokeObjectURL(url)
}

function normalizePdfText(value: unknown) {
  return String(value ?? "")
    .replace(/[\u2010\u2011\u2012\u2013\u2014\u2212]/g, "-")
    .replace(/[\u2018\u2019]/g, "'")
    .replace(/[\u201c\u201d]/g, '"')
    .replace(/\u2026/g, "...")
    .replace(/[^\u0009\u000a\u000d\u0020-\u00ff]/g, "?")
}

function pdfHexText(value: unknown) {
  const text = normalizePdfText(value)
  let hex = ""

  for (const character of text) {
    const code = character.charCodeAt(0)
    const byte = code >= 0 && code <= 255 ? code : 63
    hex += byte.toString(16).padStart(2, "0")
  }

  return `<${hex}>`
}

function wrapPdfText(text: string, maxChars: number) {
  const clean = normalizePdfText(text).replace(/\s+/g, " ").trim()
  if (!clean) return [""]

  const words = clean.split(" ")
  const lines: string[] = []
  let current = ""

  for (const word of words) {
    if (!current) {
      current = word
      continue
    }

    if (`${current} ${word}`.length <= maxChars) {
      current = `${current} ${word}`
    } else {
      lines.push(current)
      current = word
    }
  }

  if (current) lines.push(current)
  return lines
}


function downloadQualityReportPdf(
  summary: CalidadSummary,
  filters: CalidadFilterSelection,
  historico2025: CalidadSummary | null,
  historico2026: CalidadSummary | null,
) {
  // El informe actual se concentra en la lectura visual de la selección.
  // Los históricos siguen disponibles en el dashboard, pero no se mezclan
  // en este documento para mantener una lectura clara y consistente.
  const PAGE_WIDTH = 595
  const PAGE_HEIGHT = 842
  const MARGIN = 46
  const CONTENT_WIDTH = PAGE_WIDTH - MARGIN * 2
  const FOOTER_TOP = 46
  const pages: string[][] = [[]]
  let pageIndex = 0
  let y = PAGE_HEIGHT - 58

  const currentPage = () => pages[pageIndex]
  const addRaw = (command: string) => currentPage().push(command)

  const estimatedTextWidth = (text: unknown, size: number) =>
    normalizePdfText(text).length * size * 0.49

  const addText = (
    text: unknown,
    x: number,
    yPosition: number,
    size = 10,
    bold = false,
    color: [number, number, number] = [0.12, 0.16, 0.22],
  ) => {
    const font = bold ? "F2" : "F1"
    addRaw(
      `BT /${font} ${size} Tf ${color[0]} ${color[1]} ${color[2]} rg ${x} ${yPosition} Td ${pdfHexText(text)} Tj ET`,
    )
  }

  const addTextRight = (
    text: unknown,
    right: number,
    yPosition: number,
    size = 10,
    bold = false,
    color: [number, number, number] = [0.12, 0.16, 0.22],
  ) => {
    const width = estimatedTextWidth(text, size)
    addText(text, Math.max(MARGIN, right - width), yPosition, size, bold, color)
  }

  const addContinuationHeader = () => {
    addText(
      "UNIVERSIDAD DE LONDRES",
      MARGIN,
      PAGE_HEIGHT - 34,
      8.5,
      true,
      [0.08, 0.16, 0.32],
    )
    addTextRight(
      "Encuesta de Calidad - Informe de resultados",
      PAGE_WIDTH - MARGIN,
      PAGE_HEIGHT - 34,
      8.2,
      false,
      [0.38, 0.43, 0.5],
    )
    addRaw(
      `0.86 0.88 0.91 RG 0.7 w ${MARGIN} ${PAGE_HEIGHT - 43} m ${PAGE_WIDTH - MARGIN} ${PAGE_HEIGHT - 43} l S`,
    )
  }

  const newPage = () => {
    pages.push([])
    pageIndex += 1
    y = PAGE_HEIGHT - 62
    addContinuationHeader()
    y -= 8
  }

  const ensureSpace = (needed: number) => {
    if (y - needed >= FOOTER_TOP + 14) return
    newPage()
  }

  const addParagraph = (
    text: unknown,
    options: {
      size?: number
      bold?: boolean
      color?: [number, number, number]
      indent?: number
      gapAfter?: number
      maxChars?: number
    } = {},
  ) => {
    const size = options.size ?? 9.5
    const indent = options.indent ?? 0
    const lineHeight = size * 1.45
    const maxChars =
      options.maxChars ??
      Math.max(24, Math.floor((CONTENT_WIDTH - indent) / (size * 0.52)))
    const lines = wrapPdfText(String(text ?? ""), maxChars)
    ensureSpace(lines.length * lineHeight + (options.gapAfter ?? 4))

    for (const line of lines) {
      addText(
        line,
        MARGIN + indent,
        y,
        size,
        options.bold ?? false,
        options.color,
      )
      y -= lineHeight
    }
    y -= options.gapAfter ?? 4
  }

  const addHeading = (text: string, size = 14) => {
    ensureSpace(size * 2.7)
    y -= 4
    addText(text, MARGIN, y, size, true, [0.08, 0.16, 0.32])
    y -= size * 1.55
  }

  type ChartItem = { label: string; value: number }

  const addHorizontalBarChart = (
    title: string,
    description: string,
    items: ChartItem[],
    options: { sortAscending?: boolean; maxRowsPerPage?: number } = {},
  ) => {
    const data = [...items]
      .filter((item) => Number.isFinite(item.value))
      .sort((a, b) =>
        options.sortAscending === false ? b.value - a.value : a.value - b.value,
      )

    if (data.length === 0) {
      addHeading(title, 13)
      addParagraph("No hay información cuantitativa disponible para esta selección.")
      return
    }

    const maxRows = options.maxRowsPerPage ?? 13
    const chunks: ChartItem[][] = []
    for (let i = 0; i < data.length; i += maxRows) {
      chunks.push(data.slice(i, i + maxRows))
    }

    chunks.forEach((chunk, chunkIndex) => {
      if (chunkIndex > 0) newPage()

      addHeading(
        chunkIndex === 0 ? title : `${title} (continuación)`,
        13,
      )
      if (chunkIndex === 0 && description) {
        addParagraph(description, {
          size: 8.6,
          color: [0.38, 0.43, 0.5],
          gapAfter: 9,
        })
      }

      const labelWidth = 205
      const chartX = MARGIN + labelWidth
      const chartWidth = CONTENT_WIDTH - labelWidth - 42
      const rowGap = 8

      chunk.forEach((item) => {
        const labelLines = wrapPdfText(item.label, 37).slice(0, 3)
        const rowHeight = Math.max(29, labelLines.length * 10 + 7)
        ensureSpace(rowHeight + rowGap)

        const rowTop = y
        let labelY = rowTop - 7
        labelLines.forEach((line, idx) => {
          addText(
            line,
            MARGIN,
            labelY - idx * 9.5,
            7.8,
            idx === 0,
            [0.18, 0.22, 0.28],
          )
        })

        const barY = rowTop - rowHeight + 9
        const barHeight = 8
        const value = Math.max(0, Math.min(100, item.value))
        const fillWidth = (value / 100) * chartWidth

        addRaw(`0.92 0.94 0.97 rg ${chartX} ${barY} ${chartWidth} ${barHeight} re f`)
        addRaw(`0.16 0.32 0.58 rg ${chartX} ${barY} ${fillWidth} ${barHeight} re f`)

        addTextRight(
          value.toFixed(1),
          PAGE_WIDTH - MARGIN,
          barY - 1,
          8.2,
          true,
          [0.08, 0.16, 0.32],
        )

        y = rowTop - rowHeight - rowGap
      })

      ensureSpace(22)
      addText("0", chartX, y + 4, 6.8, false, [0.48, 0.52, 0.58])
      addTextRight("100", chartX + chartWidth, y + 4, 6.8, false, [0.48, 0.52, 0.58])
      addRaw(`0.84 0.87 0.91 RG 0.5 w ${chartX} ${y + 13} m ${chartX + chartWidth} ${y + 13} l S`)
      y -= 14
    })
  }

  const normalizedComments = (sectionId: string) =>
    (summary.comentarios ?? [])
      .filter((comment) => canonicalSectionId(comment.seccion_id) === sectionId)
      .filter((comment) => comment.incluir_resumen)
      .filter((comment) => comment.texto_informativo)
      .filter((comment) => isMeaningfulComment(comment.texto))

  const summarizeSectionComments = (
    comments: Array<{ texto?: string; texto_corto?: string; polaridad_base?: string }>,
  ) => summarizeCommentSet(comments)

  const aggregateQuestionsForSection = (sectionId: string): ChartItem[] => {
    const groups = new Map<
      string,
      { label: string; weighted: number; weight: number; simple: number[] }
    >()

    for (const question of summary.preguntas ?? []) {
      if (canonicalSectionId(question.seccion_id) !== sectionId) continue
      if (question.indice === null || question.indice === undefined) continue

      const label = question.texto_corto?.trim() || question.pregunta_id
      const key = normalizeAcademicLabel(label || question.pregunta_id)
      if (!key) continue

      const current = groups.get(key) ?? {
        label,
        weighted: 0,
        weight: 0,
        simple: [],
      }
      const n = Number(question.n ?? 0)
      if (n > 0) {
        current.weighted += question.indice * n
        current.weight += n
      }
      current.simple.push(question.indice)
      groups.set(key, current)
    }

    return [...groups.values()].map((group) => ({
      label: group.label,
      value:
        group.weight > 0
          ? group.weighted / group.weight
          : group.simple.reduce((sum, value) => sum + value, 0) /
            Math.max(1, group.simple.length),
    }))
  }

  // Portada y selección.
  addRaw(`0.08 0.16 0.32 rg 0 ${PAGE_HEIGHT - 142} ${PAGE_WIDTH} 142 re f`)
  addText("UNIVERSIDAD DE LONDRES", MARGIN, PAGE_HEIGHT - 48, 9, true, [1, 1, 1])
  addText("Informe de resultados", MARGIN, PAGE_HEIGHT - 79, 22, true, [1, 1, 1])
  addText("Encuesta de Calidad", MARGIN, PAGE_HEIGHT - 105, 18, true, [0.86, 0.91, 1])
  addText(
    `Generado el ${new Date().toLocaleDateString("es-MX")}`,
    MARGIN,
    PAGE_HEIGHT - 126,
    8.5,
    false,
    [0.82, 0.86, 0.93],
  )

  y = PAGE_HEIGHT - 176
  addHeading("Selección analizada", 13)
  addParagraph(`Periodo: ${filters.periodo || "Todos"}`, { size: 9.5, bold: true, gapAfter: 1 })
  addParagraph(`Modalidad: ${filters.modalidad || "Todas"}`, { size: 9.5, bold: true, gapAfter: 1 })
  addParagraph(`Programa: ${filters.servicio || "Todos los programas"}`, { size: 9.5, bold: true, gapAfter: 10 })

  const sections = [...(summary.secciones ?? [])]
    .filter((section) => canonicalSectionId(section.seccion_id) !== "ESC_00_PERFIL")
    .filter((section) => section.indice !== null)

  addHorizontalBarChart(
    "Resultado general por sección",
    "Las secciones se muestran de menor a mayor resultado para facilitar la identificación de áreas que requieren atención.",
    sections.map((section) => ({
      label: section.seccion_nombre,
      value: section.indice ?? 0,
    })),
    { sortAscending: true, maxRowsPerPage: 14 },
  )

  // Una lectura por sección: gráfica de preguntas + síntesis cualitativa.
  for (const section of sections) {
    newPage()
    const sectionId = canonicalSectionId(section.seccion_id)

    addText(
      section.seccion_nombre,
      MARGIN,
      y,
      16,
      true,
      [0.08, 0.16, 0.32],
    )
    addTextRight(
      section.indice === null ? "-" : section.indice.toFixed(1),
      PAGE_WIDTH - MARGIN,
      y,
      16,
      true,
      [0.08, 0.16, 0.32],
    )
    y -= 24
    addParagraph(
      "Resultado de las preguntas de la sección, ordenadas de menor a mayor para identificar con claridad los puntos de menor desempeño.",
      { size: 8.6, color: [0.38, 0.43, 0.5], gapAfter: 6 },
    )

    const questionData = aggregateQuestionsForSection(sectionId)
    addHorizontalBarChart(
      "Preguntas de la sección",
      "",
      questionData,
      { sortAscending: true, maxRowsPerPage: 13 },
    )

    const section2025 = historico2025?.secciones.find(
      (item) => canonicalSectionId(item.seccion_id) === sectionId,
    )
    const section2026 = historico2026?.secciones.find(
      (item) => canonicalSectionId(item.seccion_id) === sectionId,
    )
    if (section2025?.indice !== null && section2025?.indice !== undefined &&
        section2026?.indice !== null && section2026?.indice !== undefined) {
      const sectionDelta = section2026.indice - section2025.indice
      addHeading("Comparativo histórico de la sección", 12)
      addParagraph(
        `2025 ${section2025.indice.toFixed(2)} | 2026 ${section2026.indice.toFixed(2)} | Variación ${sectionDelta >= 0 ? "+" : ""}${sectionDelta.toFixed(2)} puntos`,
        { size: 9, bold: true, gapAfter: 6 },
      )
    } else {
      addHeading("Comparativo histórico de la sección", 12)
      addParagraph(
        "Sin antecedente comparable válido en 2025 para esta sección.",
        { size: 8.7, color: [0.38, 0.43, 0.5], gapAfter: 6 },
      )
    }

    if (sectionId === "ESC_03_SERVICIOS") {
      const serviceQuestions = (summary.preguntas ?? [])
        .filter((question) =>
          canonicalSectionId(question.seccion_id) === "ESC_03_SERVICIOS" &&
          question.entra_seccion &&
          question.indice !== null,
        )
        .sort((a, b) => (a.indice ?? 999) - (b.indice ?? 999))

      addHeading("Comparativo 2025 vs 2026", 12)
      serviceQuestions.forEach((question) => {
        const previous = findHistoricalInstitutionalQuestion(question, historico2025)
        const currentValue = question.indice
        const previousValue = previous?.indice ?? null
        const delta =
          currentValue !== null && previousValue !== null
            ? currentValue - previousValue
            : null
        addParagraph(
          `${institutionalServiceDisplayName(question)}: 2026 ${currentValue?.toFixed(1) ?? "-"} | 2025 ${previousValue?.toFixed(1) ?? "Sin comparable"}${delta !== null ? ` | Variación ${delta >= 0 ? "+" : ""}${delta.toFixed(1)}` : ""}`,
          { size: 8.6, gapAfter: 2 },
        )
      })

      addHeading("Comentarios vinculados por servicio", 12)
      serviceQuestions.forEach((question) => {
        const related = commentsForInstitutionalServiceQuestion(
          question,
          summary.comentarios ?? [],
        )
        if (related.length === 0) return
        addParagraph(
          `${institutionalServiceDisplayName(question)} · ${related.length} comentario${related.length === 1 ? "" : "s"} relacionado${related.length === 1 ? "" : "s"}`,
          { size: 8.8, bold: true, gapAfter: 2 },
        )
        related.slice(0, 3).forEach((comment) => {
          addParagraph(`• ${comment.texto}`, { size: 8.1, indent: 8, gapAfter: 2 })
        })
      })
    }


    if (sectionId === "ESC_04_ACADEMICOS") {
      const academicQuestions = (summary.preguntas ?? [])
        .filter((question) =>
          canonicalSectionId(question.seccion_id) === "ESC_04_ACADEMICOS" &&
          question.entra_seccion &&
          question.indice !== null,
        )
        .sort((a, b) => (a.indice ?? 999) - (b.indice ?? 999))

      addHeading("Comparativo 2025 vs 2026", 12)
      academicQuestions.forEach((question) => {
        const previous = findHistoricalAcademicQuestion(question, historico2025)
        const currentValue = question.indice
        const previousValue = previous?.indice ?? null
        const delta =
          currentValue !== null && previousValue !== null
            ? currentValue - previousValue
            : null
        addParagraph(
          `${academicServiceDisplayName(question)}: 2026 ${currentValue?.toFixed(1) ?? "-"} | 2025 ${previousValue?.toFixed(1) ?? "Sin comparable"}${delta !== null ? ` | Variación ${delta >= 0 ? "+" : ""}${delta.toFixed(1)}` : ""}`,
          { size: 8.6, gapAfter: 2 },
        )
      })

      addHeading("Comentarios vinculados por servicio académico", 12)
      academicQuestions.forEach((question) => {
        const related = commentsForAcademicServiceQuestion(
          question,
          summary.comentarios ?? [],
        )
        if (related.length === 0) return
        addParagraph(
          `${academicServiceDisplayName(question)} · ${related.length} comentario${related.length === 1 ? "" : "s"} relacionado${related.length === 1 ? "" : "s"}`,
          { size: 8.8, bold: true, gapAfter: 2 },
        )
        related.slice(0, 3).forEach((comment) => {
          addParagraph(`• ${comment.texto}`, { size: 8.1, indent: 8, gapAfter: 2 })
        })
      })
    }

    addHeading("Resumen de comentarios", 12)
    addParagraph(
      summarizeSectionComments(normalizedComments(sectionId)),
      { size: 9, gapAfter: 8 },
    )
  }

  pages.forEach((commands, index) => {
    commands.push(
      `0.86 0.88 0.91 RG 0.6 w ${MARGIN} 35 m ${PAGE_WIDTH - MARGIN} 35 l S`,
    )
    commands.push(
      `BT /F1 7.5 Tf 0.45 0.49 0.55 rg ${MARGIN} 22 Td ${pdfHexText("Universidad de Londres - Encuesta de Calidad")} Tj ET`,
    )
    const pageText = `Página ${index + 1} de ${pages.length}`
    const pageX = PAGE_WIDTH - MARGIN - estimatedTextWidth(pageText, 7.5)
    commands.push(
      `BT /F1 7.5 Tf 0.45 0.49 0.55 rg ${pageX} 22 Td ${pdfHexText(pageText)} Tj ET`,
    )
  })

  const objects: string[] = []
  objects[1] = "<< /Type /Catalog /Pages 2 0 R >>"
  objects[3] = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
  objects[4] = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>"

  const pageObjectIds: number[] = []
  let nextId = 5

  pages.forEach((commands) => {
    const pageObjectId = nextId++
    const contentObjectId = nextId++
    pageObjectIds.push(pageObjectId)

    const stream = commands.join("\n")
    objects[contentObjectId] = `<< /Length ${stream.length} >>\nstream\n${stream}\nendstream`
    objects[pageObjectId] =
      `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${PAGE_WIDTH} ${PAGE_HEIGHT}] ` +
      `/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents ${contentObjectId} 0 R >>`
  })

  objects[2] = `<< /Type /Pages /Kids [${pageObjectIds
    .map((id) => `${id} 0 R`)
    .join(" ")}] /Count ${pageObjectIds.length} >>`

  let pdf = "%PDF-1.4\n"
  const offsets: number[] = [0]

  for (let id = 1; id < objects.length; id += 1) {
    if (!objects[id]) continue
    offsets[id] = pdf.length
    pdf += `${id} 0 obj\n${objects[id]}\nendobj\n`
  }

  const xrefOffset = pdf.length
  pdf += `xref\n0 ${objects.length}\n`
  pdf += "0000000000 65535 f \n"

  for (let id = 1; id < objects.length; id += 1) {
    const offset = offsets[id] ?? 0
    pdf += `${String(offset).padStart(10, "0")} 00000 n \n`
  }

  pdf += `trailer\n<< /Size ${objects.length} /Root 1 0 R >>\nstartxref\n${xrefOffset}\n%%EOF`

  const bytes = new Uint8Array(pdf.length)
  for (let index = 0; index < pdf.length; index += 1) {
    bytes[index] = pdf.charCodeAt(index) & 0xff
  }

  const blob = new Blob([bytes], { type: "application/pdf" })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement("a")

  const fileParts = [
    "Informe_Encuesta_Calidad",
    safeFilePart(filters.periodo),
    safeFilePart(filters.modalidad),
    safeFilePart(filters.servicio),
  ].filter(Boolean)

  anchor.href = url
  anchor.download = `${fileParts.join("_")}.pdf`
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  URL.revokeObjectURL(url)
}

function getNivel(indice: number | null) {
  if (indice === null) return "Sin datos"
  if (indice <= 60) return "Crítico"
  if (indice <= 75) return "Atención"
  if (indice <= 85) return "Adecuado"
  return "Fortaleza"
}

function getNivelClasses(indice: number | null) {
  const nivel = getNivel(indice)

  if (nivel === "Crítico") {
    return "border-red-200 bg-red-50 text-red-700"
  }

  if (nivel === "Atención") {
    return "border-amber-200 bg-amber-50 text-amber-700"
  }

  if (nivel === "Adecuado") {
    return "border-emerald-200 bg-emerald-50 text-emerald-700"
  }

  if (nivel === "Fortaleza") {
    return "border-emerald-300 bg-emerald-100 text-emerald-800"
  }

  return "border-slate-200 bg-slate-50 text-slate-600"
}

function getBarColor(indice: number | null) {
  if (indice === null) return "#94a3b8"
  if (indice <= 60) return "#dc2626"
  if (indice <= 75) return "#d97706"
  if (indice <= 85) return "#059669"
  return "#047857"
}

function isMeaningfulComment(text: string) {
  const normalized = text
    .trim()
    .toLocaleLowerCase("es")
    .replace(/\s+/g, " ")

  if (!normalized) return false

  const compact = normalized.replace(
    /[\s.,;:!?¿¡'"()\[\]{}\-_/\\]+/g,
    "",
  )

  if (compact.length < 4) return false

  const withoutObservation = new Set([
    "ninguno",
    "ninguna",
    "nada",
    "sincomentario",
    "sincomentarios",
    "noaplica",
    "ninguncomentario",
    "ninguncomentario",
    "noninguno",
    "noninguna",
  ])

  return !withoutObservation.has(compact)
}

type CommentForSummary = {
  texto?: string
  polaridad_base?: string
}

function summarizeCommentSet(comments: CommentForSummary[]) {
  const useful = comments
    .map((comment) => String(comment.texto ?? "").trim())
    .filter((text) => isMeaningfulComment(text))

  if (useful.length === 0) {
    return "No se registraron comentarios sustantivos para esta selección."
  }

  const normalize = (value: string) =>
    value
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "")
      .toLocaleLowerCase("es")

  const normalizedComments = useful.map(normalize)

  const themes = [
    {
      label: "mayor presencia y disponibilidad cotidiana",
      action:
        "que la persona responsable tenga mayor presencia cotidiana y sea más fácil localizarla cuando el alumnado necesita atención",
      keys: [
        "presente", "presencia", "diario", "diaria", "todos los dias",
        "cada dia", "venir", "asistir", "oficina", "encontrar", "localizar",
        "disponible", "disponibilidad", "horario",
      ],
    },
    {
      label: "seguimiento y solución de solicitudes",
      action:
        "dar seguimiento más constante a las solicitudes y comunicar con claridad cuándo quedan resueltas",
      keys: [
        "seguimiento", "peticion", "solicitud", "resolver", "resuelva", "solucion",
        "pendiente", "tramite", "respuesta",
      ],
    },
    {
      label: "comunicación con el alumnado",
      action:
        "mejorar la comunicación con el alumnado, especialmente en avisos, respuestas e información de seguimiento",
      keys: [
        "comunicacion", "informacion", "aviso", "avisar", "explica", "claridad",
        "responde", "contesta", "mensaje",
      ],
    },
    {
      label: "trato y empatía",
      action:
        "mantener un trato cercano, respetuoso y empático en la atención al alumnado",
      keys: ["amable", "empati", "trato", "respeto", "actitud", "atento", "atenta"],
    },
    {
      label: "tiempos de atención",
      action:
        "reducir los tiempos de respuesta y agilizar la atención de asuntos pendientes",
      keys: ["tarda", "demora", "tiempo", "rapido", "agil", "espera"],
    },
  ]

  const scored = themes
    .map((theme) => {
      const matchedComments = normalizedComments.filter((comment) =>
        theme.keys.some((key) => comment.includes(key)),
      ).length
      return { ...theme, matchedComments }
    })
    .filter((theme) => theme.matchedComments > 0)
    .sort((a, b) => b.matchedComments - a.matchedComments)

  if (scored.length === 0) {
    const sample = useful[0].replace(/\s+/g, " ").trim()
    const shortened = sample.length > 170 ? `${sample.slice(0, 167)}...` : sample
    return `Los comentarios se concentran en observaciones específicas sobre la atención recibida. Una idea representativa es: “${shortened}”. Conviene revisar los comentarios originales para convertir estas observaciones en acciones puntuales.`
  }

  const principal = scored[0]
  const ratio = principal.matchedComments / useful.length
  const intro =
    ratio >= 0.5
      ? "La mayor parte de los comentarios sugiere"
      : ratio >= 0.3
        ? "Una tendencia recurrente en los comentarios sugiere"
        : "Entre los comentarios aparece de forma recurrente la necesidad de"

  let result = `${intro} ${principal.action}.`

  if (scored[1] && scored[1].matchedComments >= 2) {
    result += ` Como segundo tema se repite ${scored[1].label}.`
  }

  return result
}

function getStaffInterpretation(
  indice: number | null,
  usefulComments: number,
) {
  const commentsNote =
    usefulComments > 0
      ? `Hay ${usefulComments} ${
          usefulComments === 1 ? "comentario sustantivo" : "comentarios sustantivos"
        } para complementar la lectura.`
      : "No hay comentarios sustantivos registrados; la lectura se apoya en las evaluaciones cuantitativas."

  if (indice === null) {
    return `No hay información suficiente para interpretar el resultado de esta persona. ${commentsNote}`
  }

  if (indice >= 85) {
    return `La percepción general es favorable y se ubica como fortaleza. Conviene identificar en los comentarios qué prácticas de atención, disponibilidad, empatía o seguimiento vale la pena conservar. ${commentsNote}`
  }

  if (indice >= 75) {
    return `La percepción general es adecuada, aunque todavía hay margen de mejora. Los comentarios ayudan a ubicar situaciones puntuales que pueden elevar la experiencia de atención. ${commentsNote}`
  }

  if (indice >= 60) {
    return `El resultado requiere atención. Conviene revisar los criterios con menor promedio y contrastarlos con los comentarios para precisar si las oportunidades se concentran en disponibilidad, atención, empatía o seguimiento. ${commentsNote}`
  }

  return `El resultado se encuentra en nivel crítico y requiere seguimiento prioritario. Los comentarios deben revisarse junto con los criterios de menor promedio para identificar acciones concretas de mejora. ${commentsNote}`
}

function normalizeAcademicLabel(value: string) {
  return value
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLocaleLowerCase("es")
    .replace(/licenciatura ejecutiva|licenciatura escolarizada|licenciatura|ejecutiva|escolarizada/g, " ")
    .replace(/[^a-z0-9]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
}

function canonicalSectionId(value: string | null | undefined) {
  return (value ?? "").replace(/^ESC25_/, "ESC_")
}

function canonicalQuestionId(value: string | null | undefined) {
  return (value ?? "").replace(/^ESC25_/, "ESC_")
}

function findHistoricalComparableQuestion(
  question: CalidadQuestion,
  historical: CalidadSummary | null,
) {
  if (!historical) return null

  const sectionId = canonicalSectionId(question.seccion_id)
  const questionId = canonicalQuestionId(question.pregunta_id)
  const sectionName = normalizeAcademicLabel(question.seccion_nombre ?? "")
  const candidates = historical.preguntas.filter((item) => {
    const sameId = canonicalSectionId(item.seccion_id) === sectionId
    const sameName =
      sectionName.length > 0 &&
      normalizeAcademicLabel(item.seccion_nombre ?? "") === sectionName

    return (sameId || sameName) && item.indice !== null
  })

  const exactById = candidates.find(
    (item) => canonicalQuestionId(item.pregunta_id) === questionId,
  )
  if (exactById) return exactById

  const label = normalizeAcademicLabel(
    question.texto_corto || question.pregunta_id,
  )
  const exactByLabel = candidates.find(
    (item) =>
      normalizeAcademicLabel(item.texto_corto || item.pregunta_id) === label,
  )
  if (exactByLabel) return exactByLabel

  const stop = new Set([
    "como", "para", "cada", "este", "esta", "estos", "estas",
    "general", "servicio", "servicios", "universidad", "estudiantes",
    "considero", "ofrecidos", "ofrece", "solo", "utilizo",
  ])
  const tokens = new Set(
    label
      .split(/\s+/)
      .filter((token) => token.length >= 4 && !stop.has(token)),
  )

  let best: CalidadQuestion | null = null
  let bestScore = 0
  for (const candidate of candidates) {
    const candidateLabel = normalizeAcademicLabel(
      candidate.texto_corto || candidate.pregunta_id,
    )
    const candidateTokens = new Set(
      candidateLabel
        .split(/\s+/)
        .filter((token) => token.length >= 4 && !stop.has(token)),
    )
    const common = [...tokens].filter((token) => candidateTokens.has(token)).length
    const union = new Set([...tokens, ...candidateTokens]).size
    const score = union > 0 ? common / union : 0
    if (score > bestScore) {
      best = candidate
      bestScore = score
    }
  }

  return bestScore >= 0.3 ? best : null
}

function getDirectionInterpretation(
  indice: number | null,
  usefulComments: number,
) {
  const commentsNote =
    usefulComments > 0
      ? `Se localizaron ${usefulComments} ${
          usefulComments === 1 ? "comentario sustantivo" : "comentarios sustantivos"
        } para contextualizar el resultado.`
      : "No hay comentarios sustantivos asociados a esta selección; la lectura se apoya en los indicadores cuantitativos."

  if (indice === null) {
    return `No hay información suficiente para interpretar esta selección. ${commentsNote}`
  }

  if (indice >= 85) {
    return `La percepción de la Dirección se ubica como fortaleza. Conviene conservar las prácticas mejor valoradas y revisar los comentarios para identificar qué elementos de disponibilidad, atención, solución de problemas y seguimiento explican el resultado. ${commentsNote}`
  }

  if (indice >= 75) {
    return `La percepción de la Dirección es adecuada, con margen de mejora. Conviene revisar el criterio con menor promedio y contrastarlo con los comentarios para ubicar acciones puntuales. ${commentsNote}`
  }

  if (indice >= 60) {
    return `El resultado requiere atención. La prioridad es identificar si la principal oportunidad está en disponibilidad, atención a necesidades, solución de problemas o seguimiento al desempeño docente, y contrastarla con los comentarios. ${commentsNote}`
  }

  return `La percepción de la Dirección se encuentra en nivel crítico y requiere seguimiento prioritario. Deben revisarse los criterios de menor promedio y los comentarios asociados antes de definir acciones de mejora. ${commentsNote}`
}

function QuestionResultCard({
  question,
}: {
  question: CalidadQuestion
}) {
  const favorable =
    question.favorable_pct ?? 0
  const neutral =
    question.neutral_pct ?? 0
  const desfavorable =
    question.desfavorable_pct ?? 0
  const esBinaria =
    question.tipo_dato === "BINARIA"

  return (
    <div className="rounded-xl border bg-background p-5">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div className="max-w-4xl">
          <div className="flex flex-wrap items-center gap-2">
            <span className="rounded-full border px-2.5 py-1 text-xs text-muted-foreground">
              {esBinaria ? "Binaria" : "Likert"}
            </span>

            <span className="text-xs text-muted-foreground">
              n = {question.n}
            </span>
          </div>

          <h4 className="mt-3 text-sm font-semibold">
            {question.texto_corto}
          </h4>
        </div>

        <div className="min-w-[120px] text-left lg:text-right">
          <p className="text-xs text-muted-foreground">
            {esBinaria
              ? "Tasa favorable"
              : "Promedio"}
          </p>

          <p className="mt-1 text-xl font-semibold">
            {question.indice?.toFixed(2) ?? "?"}
          </p>
        </div>
      </div>

      <div className="mt-4">
        <div className="flex h-2.5 w-full overflow-hidden rounded-full bg-muted">
          <div
            className="bg-emerald-600"
            style={{
              width: `${favorable}%`,
            }}
          />

          {!esBinaria && (
            <div
              className="bg-amber-400"
              style={{
                width: `${neutral}%`,
              }}
            />
          )}

          <div
            className="bg-red-500"
            style={{
              width: `${desfavorable}%`,
            }}
          />
        </div>

        <div className="mt-2 flex flex-wrap gap-x-5 gap-y-1 text-xs">
          <span className="text-emerald-700">
            Favorable {favorable.toFixed(1)}%
          </span>

          {!esBinaria && (
            <span className="text-amber-700">
              Neutral {neutral.toFixed(1)}%
            </span>
          )}

          <span className="text-red-700">
            Desfavorable {desfavorable.toFixed(1)}%
          </span>
        </div>
      </div>
    </div>
  )
}

export default function EncuestaCalidad() {
  const [summary, setSummary] = useState<CalidadSummary | null>(null)

  const [historico2025, setHistorico2025] =
    useState<CalidadSummary | null>(null)

  const [historico2026, setHistorico2026] =
    useState<CalidadSummary | null>(null)

  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState("resumen")

  const [selectedSectionId, setSelectedSectionId] =
    useState("")

  const [
    selectedSectionCareer,
    setSelectedSectionCareer,
  ] = useState("")

  const [
    selectedSectionModality,
    setSelectedSectionModality,
  ] = useState("")

  const [assistantOpen, setAssistantOpen] =
    useState(false)
  const [siaQuestion, setSiaQuestion] = useState("")
  const [siaAnswer, setSiaAnswer] = useState("")

  const [institutionalServiceDrawerOpen, setInstitutionalServiceDrawerOpen] =
    useState(false)
  const [selectedInstitutionalServiceQuestionId, setSelectedInstitutionalServiceQuestionId] =
    useState("")

  const [academicServiceDrawerOpen, setAcademicServiceDrawerOpen] =
    useState(false)
  const [selectedAcademicServiceQuestionId, setSelectedAcademicServiceQuestionId] =
    useState("")

  const [languageCenterDrawerOpen, setLanguageCenterDrawerOpen] =
    useState(false)
  const [selectedLanguageCenterQuestionId, setSelectedLanguageCenterQuestionId] =
    useState("")

  const [continuingEducationDrawerOpen, setContinuingEducationDrawerOpen] =
    useState(false)
  const [selectedContinuingEducationQuestionId, setSelectedContinuingEducationQuestionId] =
    useState("")

  const [facilitiesDrawerOpen, setFacilitiesDrawerOpen] = useState(false)
  const [selectedFacilitiesQuestionId, setSelectedFacilitiesQuestionId] = useState("")

  const [staffDrawerOpen, setStaffDrawerOpen] =
    useState(false)

  const [staffPersonDrawerOpen, setStaffPersonDrawerOpen] =
    useState(false)

  const [staffView, setStaffView] = useState<
    "carrera" | "persona"
  >("carrera")

  const [
    selectedStaffOverviewPersonKey,
    setSelectedStaffOverviewPersonKey,
  ] = useState("")

  const [
    selectedDirectionCareer,
    setSelectedDirectionCareer,
  ] = useState("")

  const [
    selectedDirectionModality,
    setSelectedDirectionModality,
  ] = useState("")


  const [directionView, setDirectionView] = useState<
    "carrera" | "director"
  >("carrera")

  const [directionCareerDrawerOpen, setDirectionCareerDrawerOpen] =
    useState(false)

  const [directionDirectorDrawerOpen, setDirectionDirectorDrawerOpen] =
    useState(false)

  const [selectedDirectionDirectorKey, setSelectedDirectionDirectorKey] =
    useState("")

  const [
    selectedStaffCareer,
    setSelectedStaffCareer,
  ] = useState("")

  const [
    selectedStaffModality,
    setSelectedStaffModality,
  ] = useState("")

  const [
    selectedStaffPersonKey,
    setSelectedStaffPersonKey,
  ] = useState("")

  const [filters, setFilters] =
    useState<CalidadFilterSelection>({
      periodo: "2026",
      modalidad: "Escolarizadas",
      servicio: "",
    })

  useEffect(() => {
    async function cargarDatos() {
      try {
        setLoading(true)
        setError(null)

        const data = await getCalidadSummary(filters)

        setSummary(data)
      } catch (err) {
        setError(
          err instanceof Error
            ? err.message
            : "No fue posible cargar Encuesta de Calidad.",
        )
      } finally {
        setLoading(false)
      }
    }

    cargarDatos()
  }, [
    filters.periodo,
    filters.modalidad,
    filters.servicio,
  ])

  useEffect(() => {
    async function cargarHistorico(
      periodo: "2025" | "2026",
    ): Promise<CalidadSummary> {
      const buildUrl = (withFilters: boolean) => {
        const params = new URLSearchParams()
        params.set("periodo", periodo)
        params.set("historico", "true")

        if (withFilters) {
          if (filters.modalidad) {
            params.set("modalidad", filters.modalidad)
          }
          if (filters.servicio) {
            params.set("servicio", filters.servicio)
          }
        }

        return `http://127.0.0.1:8000/api/encuesta-calidad/preview/resumen?${params.toString()}`
      }

      const request = async (withFilters: boolean) => {
        const response = await fetch(buildUrl(withFilters))
        if (!response.ok) {
          throw new Error(
            `No fue posible cargar el histórico ${periodo}.`,
          )
        }
        return (await response.json()) as CalidadSummary
      }

      const filtered = await request(true)

      // Algunos valores de modalidad/servicio cambiaron entre 2025 y 2026.
      // Si el filtro actual deja el histórico sin reactivos, recuperamos el
      // periodo completo para no perder comparables válidos.
      if (
        filtered.encuestas > 0 &&
        (filtered.preguntas?.length ?? 0) > 0
      ) {
        return filtered
      }

      return request(false)
    }

    async function cargarComparativosHistoricos() {
      try {
        const [data2025, data2026] = await Promise.all([
          cargarHistorico("2025"),
          cargarHistorico("2026"),
        ])

        setHistorico2025(data2025)
        setHistorico2026(data2026)
      } catch (err) {
        console.error(
          "No fue posible cargar los comparativos históricos.",
          err,
        )
        setHistorico2025(null)
        setHistorico2026(null)
      }
    }

    cargarComparativosHistoricos()
  }, [filters.modalidad, filters.servicio])

  const moduleTabs = [
    {
      id: "resumen",
      label: "Resumen general",
    },
    {
      id: "secciones",
      label: "Por sección",
    },
    {
      id: "comparativos",
      label: "Comparativos",
    },
    {
      id: "comentarios",
      label: "Preguntas abiertas",
    },
  ]

  const availableSections =
    (summary?.secciones ?? []).filter(
      (section) => canonicalSectionId(section.seccion_id) !== "ESC_00_PERFIL",
    )

  const requestedSectionId = canonicalSectionId(selectedSectionId)

  const effectiveSectionId =
    availableSections.some(
      (section) =>
        canonicalSectionId(section.seccion_id) === requestedSectionId,
    )
      ? requestedSectionId
      : canonicalSectionId(availableSections[0]?.seccion_id ?? "")

  const selectedSection =
    availableSections.find(
      (section) =>
        canonicalSectionId(section.seccion_id) === effectiveSectionId,
    ) ?? null

  const selectedSectionQuestions =
    summary?.preguntas.filter(
      (question) =>
        canonicalSectionId(question.seccion_id) === effectiveSectionId,
    ) ?? []

  const sectionCareerModalities =
    (
      summary?.comparativos_secciones
        ?.por_carrera_modalidad.filter(
          (item) =>
            canonicalSectionId(item.seccion_id) ===
              effectiveSectionId &&
            item.indice !== null,
        ) ?? []
    ).sort((a, b) => {
      const diferencia =
        (b.indice ?? -1) -
        (a.indice ?? -1)

      if (diferencia !== 0) {
        return diferencia
      }

      const carrera =
        a.carrera_base.localeCompare(
          b.carrera_base,
          "es",
        )

      if (carrera !== 0) {
        return carrera
      }

      return a.modalidad_academica.localeCompare(
        b.modalidad_academica,
        "es",
      )
    })

  const selectedSectionPopulation =
    sectionCareerModalities.find(
      (item) =>
        item.carrera_base ===
          selectedSectionCareer &&
        item.modalidad_academica ===
          selectedSectionModality,
    ) ??
    sectionCareerModalities[0] ??
    null

  const effectiveSectionCareer =
    selectedSectionPopulation?.carrera_base ??
    ""

  const effectiveSectionModality =
    selectedSectionPopulation
      ?.modalidad_academica ?? ""

  const selectedSectionCareerQuestions =
    summary?.comparativos_secciones
      ?.preguntas_por_carrera_modalidad.filter(
        (item) =>
          item.seccion_id ===
            effectiveSectionId &&
          item.carrera_base ===
            effectiveSectionCareer &&
          item.modalidad_academica ===
            effectiveSectionModality,
      ) ?? []

  const selectedSectionPeople =
    summary?.personas.filter(
      (person) =>
        canonicalSectionId(person.seccion_id) ===
        effectiveSectionId,
    ) ?? []

  const staffSummary = (
    summary?.comparativos_personal as
      | (CalidadSummary["comparativos_personal"] & {
          resumen?: {
            evaluaciones: number
            criterios_evaluados: number
            personas_evaluadas: number
          }
        })
      | undefined
  )?.resumen

  const staffCareerModalitiesForSection =
    (
      summary?.comparativos_personal?.por_carrera_modalidad.filter(
        (item) =>
          canonicalSectionId(item.seccion_id) === effectiveSectionId,
      ) ?? []
    ).sort((a, b) => {
      const aTieneDatos =
        a.tiene_datos &&
        a.indice !== null

      const bTieneDatos =
        b.tiene_datos &&
        b.indice !== null

      if (aTieneDatos !== bTieneDatos) {
        return aTieneDatos ? -1 : 1
      }

      if (aTieneDatos && bTieneDatos) {
        const diferencia =
          (b.indice ?? -1) -
          (a.indice ?? -1)

        if (diferencia !== 0) {
          return diferencia
        }
      }

      const carrera =
        a.carrera_base.localeCompare(
          b.carrera_base,
          "es",
        )

      if (carrera !== 0) {
        return carrera
      }

      return a.modalidad_academica.localeCompare(
        b.modalidad_academica,
        "es",
      )
    })

  const selectedStaffPopulation =
    staffCareerModalitiesForSection.find(
      (item) =>
        item.carrera_base === selectedStaffCareer &&
        item.modalidad_academica ===
          selectedStaffModality,
    ) ??
    staffCareerModalitiesForSection[0] ??
    null

  const effectiveStaffCareer =
    selectedStaffPopulation?.carrera_base ?? ""

  const effectiveStaffModality =
    selectedStaffPopulation
      ?.modalidad_academica ?? ""

  const selectedStaffModalitySummary =
    selectedStaffPopulation

  const staffPeopleForCareer =
    summary?.comparativos_personal
      ?.personas_por_carrera_modalidad.filter(
        (item) =>
          canonicalSectionId(item.seccion_id) === effectiveSectionId &&
          item.carrera_base ===
            effectiveStaffCareer &&
          item.modalidad_academica ===
            effectiveStaffModality,
      ) ?? []

  const makeStaffPersonKey = (
    persona: string,
    rol: string,
  ) => `${persona}|||${rol}`

  const effectiveStaffOverviewPersonKey =
    selectedSectionPeople.some(
      (item) =>
        makeStaffPersonKey(
          item.persona_evaluada,
          item.rol_evaluado,
        ) === selectedStaffOverviewPersonKey,
    )
      ? selectedStaffOverviewPersonKey
      : ""

  const effectiveStaffOverviewPerson =
    selectedSectionPeople.find(
      (item) =>
        makeStaffPersonKey(
          item.persona_evaluada,
          item.rol_evaluado,
        ) === effectiveStaffOverviewPersonKey,
    ) ?? null

  const staffOverviewQuestions =
    summary?.preguntas.filter(
      (item) =>
        canonicalSectionId(item.seccion_id) === effectiveSectionId &&
        item.persona_evaluada ===
          effectiveStaffOverviewPerson?.persona_evaluada &&
        item.rol_evaluado ===
          effectiveStaffOverviewPerson?.rol_evaluado &&
        item.entra_seccion &&
        item.indice !== null,
    ) ?? []

  const staffOverviewAssignments = Array.from(
    new Map(
      (summary?.comparativos_personal
        ?.personas_por_carrera_modalidad ?? [])
        .filter(
          (item) =>
            Boolean(effectiveStaffOverviewPerson) &&
            normalizeAcademicLabel(item.persona_evaluada) ===
              normalizeAcademicLabel(
                effectiveStaffOverviewPerson?.persona_evaluada ?? "",
              ),
        )
        .map((item) => [
          `${item.carrera_base}|||${item.modalidad_academica}`,
          {
            carrera_base: item.carrera_base,
            modalidad_academica: item.modalidad_academica,
          },
        ]),
    ).values(),
  )

  const staffOverviewComments =
    summary?.comentarios.filter((comment) => {
      if (!effectiveStaffOverviewPerson || !comment.incluir_resumen) {
        return false
      }

      if (canonicalSectionId(comment.seccion_id) !== effectiveSectionId) {
        return false
      }

      const selectedPerson = normalizeAcademicLabel(
        effectiveStaffOverviewPerson.persona_evaluada,
      )
      const commentPerson = normalizeAcademicLabel(
        comment.persona_evaluada ?? "",
      )

      // La relación nominal tiene prioridad y no depende de que el rol
      // esté escrito exactamente igual en todos los años del formulario.
      if (commentPerson) {
        return commentPerson === selectedPerson
      }

      // Formularios antiguos pueden conservar el comentario y la carrera,
      // pero no persona_evaluada. En ese caso usamos las carreras asignadas.
      const service = normalizeAcademicLabel(
        comment.servicio_programa ?? "",
      )

      return staffOverviewAssignments.some((assignment) => {
        const career = normalizeAcademicLabel(assignment.carrera_base)
        return Boolean(
          service && career &&
            (service.includes(career) || career.includes(service)),
        )
      })
    }) ?? []

  const staffOverviewMeaningfulComments =
    staffOverviewComments.filter((comment) =>
      isMeaningfulComment(comment.texto ?? ""),
    )

  const staffOverviewCommentSummary = summarizeCommentSet(
    staffOverviewMeaningfulComments,
  )

  const staffOverviewInterpretation =
    getStaffInterpretation(
      effectiveStaffOverviewPerson?.indice ?? null,
      staffOverviewMeaningfulComments.length,
    )

  const effectiveStaffPersonKey =
    staffPeopleForCareer.some(
      (item) =>
        makeStaffPersonKey(
          item.persona_evaluada,
          item.rol_evaluado,
        ) === selectedStaffPersonKey,
    )
      ? selectedStaffPersonKey
      : staffPeopleForCareer[0]
        ? makeStaffPersonKey(
            staffPeopleForCareer[0]
              .persona_evaluada,
            staffPeopleForCareer[0]
              .rol_evaluado,
          )
        : ""

  const effectiveStaffPerson =
    staffPeopleForCareer.find(
      (item) =>
        makeStaffPersonKey(
          item.persona_evaluada,
          item.rol_evaluado,
        ) === effectiveStaffPersonKey,
    ) ?? null

  const staffQuestionsByCareer =
    summary?.comparativos_personal
      ?.preguntas_persona_por_carrera_modalidad.filter(
        (item) =>
          canonicalSectionId(item.seccion_id) === effectiveSectionId &&
          item.carrera_base ===
            effectiveStaffCareer &&
          item.modalidad_academica ===
            effectiveStaffModality &&
          item.persona_evaluada ===
            effectiveStaffPerson?.persona_evaluada &&
          item.rol_evaluado ===
            effectiveStaffPerson?.rol_evaluado,
      ) ?? []

  const staffUniqueCriteriaCount = new Set(
    (
      summary?.comparativos_personal
        ?.preguntas_persona_por_carrera_modalidad ?? []
    )
      .filter(
        (item) => canonicalSectionId(item.seccion_id) === "ESC_01_PERSONAL",
      )
      .map((item) => item.pregunta_id)
      .filter(Boolean),
  ).size

  const directionCareerModalities =
    (
      summary?.comparativos_direccion
        ?.por_carrera_modalidad ?? []
    ).sort((a, b) => {
      const aConDatos = a.estado === "con_datos" && a.indice !== null
      const bConDatos = b.estado === "con_datos" && b.indice !== null

      if (aConDatos !== bConDatos) {
        return aConDatos ? -1 : 1
      }

      if (aConDatos && bConDatos) {
        const indiceA = a.indice ?? -1
        const indiceB = b.indice ?? -1
        if (indiceB !== indiceA) {
          return indiceB - indiceA
        }
      }

      const carrera =
        a.carrera_base.localeCompare(
          b.carrera_base,
          "es",
        )

      if (carrera !== 0) {
        return carrera
      }

      return a.modalidad_academica.localeCompare(
        b.modalidad_academica,
        "es",
      )
    })

  const selectedDirectionPopulation =
    directionCareerModalities.find(
      (item) =>
        item.carrera_base ===
          selectedDirectionCareer &&
        item.modalidad_academica ===
          selectedDirectionModality,
    ) ??
    directionCareerModalities[0] ??
    null

  const effectiveDirectionCareer =
    selectedDirectionPopulation
      ?.carrera_base ?? ""

  const effectiveDirectionModality =
    selectedDirectionPopulation
      ?.modalidad_academica ?? ""

  const directionQuestions =
    summary?.comparativos_direccion
      ?.preguntas_por_carrera_modalidad.filter(
        (item) =>
          item.carrera_base ===
            effectiveDirectionCareer &&
          item.modalidad_academica ===
            effectiveDirectionModality,
      ) ?? []

  const historicalDirectionQuestionsForCareer =
    historico2025?.comparativos_direccion
      ?.preguntas_por_carrera_modalidad.filter(
        (item) =>
          normalizeAcademicLabel(item.carrera_base) ===
            normalizeAcademicLabel(effectiveDirectionCareer) &&
          normalizeAcademicLabel(item.modalidad_academica) ===
            normalizeAcademicLabel(effectiveDirectionModality),
      ) ?? []

  const findHistoricalDirectionQuestion = (question: {
    pregunta_id: string
    texto_corto: string
  }) => {
    const canonicalId = canonicalQuestionId(question.pregunta_id)
    const exactById = historicalDirectionQuestionsForCareer.find(
      (item) => canonicalQuestionId(item.pregunta_id) === canonicalId,
    )
    if (exactById) return exactById

    const label = normalizeAcademicLabel(question.texto_corto || question.pregunta_id)
    return historicalDirectionQuestionsForCareer.find(
      (item) =>
        normalizeAcademicLabel(item.texto_corto || item.pregunta_id) === label,
    ) ?? null
  }


  const directionComparisons =
    summary?.comparativos_direccion as
      | DirectionComparisonsExtended
      | undefined

  const directionDirectors =
    directionComparisons?.por_director ?? []

  const makeDirectionDirectorKey = (
    persona: string,
    rol: string,
  ) => `${persona}|||${rol}`

  const effectiveDirectionDirectorKey =
    directionDirectors.some(
      (item) =>
        makeDirectionDirectorKey(
          item.persona_evaluada,
          item.rol_evaluado,
        ) === selectedDirectionDirectorKey,
    )
      ? selectedDirectionDirectorKey
      : ""

  const effectiveDirectionDirector =
    directionDirectors.find(
      (item) =>
        makeDirectionDirectorKey(
          item.persona_evaluada,
          item.rol_evaluado,
        ) === effectiveDirectionDirectorKey,
    ) ?? null

  const directionDirectorQuestions =
    (directionComparisons?.preguntas_por_director ?? []).filter(
      (item) =>
        item.persona_evaluada ===
          effectiveDirectionDirector?.persona_evaluada &&
        item.rol_evaluado ===
          effectiveDirectionDirector?.rol_evaluado,
    )

  const directionSectionComments =
    summary?.comentarios.filter(
      (comment) =>
        canonicalSectionId(comment.seccion_id) === "ESC_02_DIRECCION" &&
        isMeaningfulComment(comment.texto),
    ) ?? []

  const directionCommentsForCareer =
    directionSectionComments.filter((comment) => {
      const service = normalizeAcademicLabel(
        comment.servicio_programa ?? "",
      )
      const career = normalizeAcademicLabel(
        effectiveDirectionCareer,
      )

      return Boolean(
        service &&
          career &&
          (service.includes(career) || career.includes(service)),
      )
    })

  const directionCommentsForDirector =
    directionSectionComments.filter((comment) => {
      const commentPerson = normalizeAcademicLabel(
        comment.persona_evaluada ?? "",
      )
      const directorPerson = normalizeAcademicLabel(
        effectiveDirectionDirector?.persona_evaluada ?? "",
      )

      // Cuando el comentario ya viene ligado nominalmente al director,
      // esa es la relacion mas precisa y tiene prioridad.
      if (commentPerson && directorPerson) {
        return commentPerson === directorPerson
      }

      // Compatibilidad con formularios historicos donde el comentario
      // no guarda persona_evaluada y solo conserva la carrera/programa.
      const service = normalizeAcademicLabel(
        comment.servicio_programa ?? "",
      )

      return (effectiveDirectionDirector?.poblaciones ?? []).some(
        (population) => {
          const career = normalizeAcademicLabel(
            population.carrera_base,
          )
          return Boolean(
            service &&
              career &&
              (service.includes(career) || career.includes(service)),
          )
        },
      )
    })

  const directionCareerCommentSummary = summarizeCommentSet(
    directionCommentsForCareer,
  )

  const directionDirectorCommentSummary = summarizeCommentSet(
    directionCommentsForDirector,
  )

  const selectedSectionGeneralQuestions =
    selectedSectionQuestions.filter(
      (question) =>
        !question.persona_evaluada?.trim(),
    )

  const selectedSectionIndexQuestions =
    selectedSectionGeneralQuestions.filter(
      (question) =>
        question.entra_seccion,
    )

  const selectedSectionComplementaryQuestions =
    selectedSectionGeneralQuestions.filter(
      (question) =>
        !question.entra_seccion,
    )

  const selectedSectionNameKey = normalizeAcademicLabel(
    selectedSection?.seccion_nombre ?? "",
  )

  const selectedSection2025 =
    historico2025?.secciones.find((section) => {
      const sameId = canonicalSectionId(section.seccion_id) === effectiveSectionId
      const sameName =
        selectedSectionNameKey.length > 0 &&
        normalizeAcademicLabel(section.seccion_nombre ?? "") === selectedSectionNameKey
      return sameId || sameName
    }) ?? null

  const selectedSection2026 =
    historico2026?.secciones.find((section) => {
      const sameId = canonicalSectionId(section.seccion_id) === effectiveSectionId
      const sameName =
        selectedSectionNameKey.length > 0 &&
        normalizeAcademicLabel(section.seccion_nombre ?? "") === selectedSectionNameKey
      return sameId || sameName
    }) ?? null

  const selectedSectionHistoricalDelta =
    selectedSection2025?.indice !== null &&
    selectedSection2025?.indice !== undefined &&
    selectedSection2026?.indice !== null &&
    selectedSection2026?.indice !== undefined
      ? selectedSection2026.indice - selectedSection2025.indice
      : null

  const selectedSectionComparableQuestions =
    selectedSectionIndexQuestions.filter(
      (question) =>
        Boolean(findHistoricalComparableQuestion(question, historico2025)),
    ).length

  const selectedSectionComments =
    summary?.comentarios.filter(
      (comment) =>
        canonicalSectionId(comment.seccion_id) === effectiveSectionId &&
        comment.incluir_resumen &&
        comment.texto_informativo,
    ) ?? []

  const selectedSectionQualitativeSummary =
    summary?.resumen_cualitativo.find(
      (item) =>
        canonicalSectionId(item.seccion_id) === effectiveSectionId,
    ) ?? null

  const selectedSectionCommentSamples =
    selectedSectionComments.slice(0, 4)

  const institutionalServiceQuestions: InstitutionalServiceQuestion[] =
    selectedSectionIndexQuestions
      .filter((question) =>
        canonicalSectionId(question.seccion_id) === "ESC_03_SERVICIOS" &&
        question.indice !== null,
      )
      .map((question) => ({
        ...question,
        historical_index:
          findHistoricalInstitutionalQuestion(question, historico2025)?.indice ?? null,
      }))
      .sort((a, b) => (a.indice ?? 999) - (b.indice ?? 999))

  const selectedInstitutionalServiceQuestion =
    institutionalServiceQuestions.find(
      (question) => question.pregunta_id === selectedInstitutionalServiceQuestionId,
    ) ?? institutionalServiceQuestions[0] ?? null

  const selectedInstitutionalServiceComments =
    selectedInstitutionalServiceQuestion
      ? commentsForInstitutionalServiceQuestion(
          selectedInstitutionalServiceQuestion,
          summary?.comentarios ?? [],
        )
      : []

  const institutionalServiceDelta =
    selectedInstitutionalServiceQuestion?.indice !== null &&
    selectedInstitutionalServiceQuestion?.historical_index !== null &&
    selectedInstitutionalServiceQuestion?.historical_index !== undefined
      ? (selectedInstitutionalServiceQuestion?.indice ?? 0) -
        (selectedInstitutionalServiceQuestion?.historical_index ?? 0)
      : null


  const academicServiceQuestions: AcademicServiceQuestion[] =
    selectedSectionIndexQuestions
      .filter((question) =>
        canonicalSectionId(question.seccion_id) === "ESC_04_ACADEMICOS" &&
        question.indice !== null,
      )
      .map((question) => ({
        ...question,
        historical_index:
          findHistoricalAcademicQuestion(question, historico2025)?.indice ?? null,
      }))
      .sort((a, b) => (a.indice ?? 999) - (b.indice ?? 999))

  const selectedAcademicServiceQuestion =
    academicServiceQuestions.find(
      (question) => question.pregunta_id === selectedAcademicServiceQuestionId,
    ) ?? academicServiceQuestions[0] ?? null

  const selectedAcademicServiceComments =
    selectedAcademicServiceQuestion
      ? commentsForAcademicServiceQuestion(
          selectedAcademicServiceQuestion,
          summary?.comentarios ?? [],
        )
      : []

  const academicServiceDelta =
    selectedAcademicServiceQuestion?.indice !== null &&
    selectedAcademicServiceQuestion?.historical_index !== null &&
    selectedAcademicServiceQuestion?.historical_index !== undefined
      ? (selectedAcademicServiceQuestion?.indice ?? 0) -
        (selectedAcademicServiceQuestion?.historical_index ?? 0)
      : null

  const languageCenterQuestions: LanguageCenterQuestion[] =
    selectedSectionIndexQuestions
      .filter(
        (question) =>
          canonicalSectionId(question.seccion_id) === "ESC_05_IDIOMAS" &&
          question.indice !== null,
      )
      .map((question) => ({
        ...question,
        historical_index:
          findHistoricalLanguageCenterQuestion(
            question,
            historico2025,
          )?.indice ?? null,
      }))
      .sort((a, b) => (a.indice ?? 999) - (b.indice ?? 999))

  const selectedLanguageCenterQuestion =
    languageCenterQuestions.find(
      (question) =>
        question.pregunta_id === selectedLanguageCenterQuestionId,
    ) ??
    languageCenterQuestions[0] ??
    null

  const selectedLanguageCenterComments =
    selectedLanguageCenterQuestion
      ? commentsForLanguageCenterQuestion(
          selectedLanguageCenterQuestion,
          summary?.comentarios ?? [],
        )
      : []

  const languageCenterDelta =
    selectedLanguageCenterQuestion?.indice !== null &&
    selectedLanguageCenterQuestion?.historical_index !== null &&
    selectedLanguageCenterQuestion?.historical_index !== undefined
      ? (selectedLanguageCenterQuestion?.indice ?? 0) -
        (selectedLanguageCenterQuestion?.historical_index ?? 0)
      : null

  const languageCenterAllComments =
    summary?.comentarios.filter(
      (comment) =>
        canonicalSectionId(comment.seccion_id) === "ESC_05_IDIOMAS" &&
        comment.incluir_resumen &&
        isMeaningfulComment(comment.texto ?? ""),
    ) ?? []

  const languageCenterStrengths =
    languageCenterAllComments.filter(
      (comment) => languageCenterCommentKind(comment) === "fortaleza",
    )

  const languageCenterOpportunities =
    languageCenterAllComments.filter(
      (comment) => languageCenterCommentKind(comment) === "mejora",
    )

  const languageCenterGeneralComments =
    languageCenterAllComments.filter(
      (comment) => languageCenterCommentKind(comment) === "general",
    )

  const isContinuingEducationSection =
    normalizeAcademicLabel(selectedSection?.seccion_nombre ?? "").includes("educacion continua")

  const continuingEducationQuestions: ContinuingEducationQuestion[] =
    isContinuingEducationSection
      ? selectedSectionGeneralQuestions
          .filter((question) => question.indice !== null)
          .map((question) => ({
            ...question,
            historical_index:
              findHistoricalContinuingEducationQuestion(question, historico2025)?.indice ?? null,
          }))
          .sort((a, b) => (a.indice ?? 999) - (b.indice ?? 999))
      : []

  const selectedContinuingEducationQuestion =
    continuingEducationQuestions.find(
      (question) => question.pregunta_id === selectedContinuingEducationQuestionId,
    ) ?? continuingEducationQuestions[0] ?? null

  const selectedContinuingEducationComments =
    selectedContinuingEducationQuestion
      ? commentsForContinuingEducationQuestion(
          selectedContinuingEducationQuestion,
          summary?.comentarios ?? [],
        )
      : []

  const continuingEducationDelta =
    selectedContinuingEducationQuestion?.indice !== null &&
    selectedContinuingEducationQuestion?.historical_index !== null &&
    selectedContinuingEducationQuestion?.historical_index !== undefined
      ? (selectedContinuingEducationQuestion?.indice ?? 0) -
        (selectedContinuingEducationQuestion?.historical_index ?? 0)
      : null

  const continuingEducationImprovementList =
    continuingEducationImprovementComments(summary?.comentarios ?? [])

  const isFacilitiesSection =
    effectiveSectionId === "ESC_07_INSTALACIONES" ||
    normalizeAcademicLabel(selectedSection?.seccion_nombre ?? "").includes("instalaciones y equipo")

  const facilitiesQuestions: FacilitiesQuestion[] =
    isFacilitiesSection
      ? selectedSectionGeneralQuestions
          .filter((question) => question.indice !== null)
          .map((question) => ({
            ...question,
            historical_index: findHistoricalFacilitiesQuestion(question, historico2025)?.indice ?? null,
          }))
          .sort((a, b) => (a.indice ?? 999) - (b.indice ?? 999))
      : []

  const selectedFacilitiesQuestion =
    facilitiesQuestions.find((question) => question.pregunta_id === selectedFacilitiesQuestionId) ??
    facilitiesQuestions[0] ?? null

  const selectedFacilitiesComments = selectedFacilitiesQuestion
    ? facilitiesCommentsForQuestion(selectedFacilitiesQuestion, summary?.comentarios ?? [])
    : []

  const facilitiesDelta =
    selectedFacilitiesQuestion?.indice !== null &&
    selectedFacilitiesQuestion?.historical_index !== null &&
    selectedFacilitiesQuestion?.historical_index !== undefined
      ? (selectedFacilitiesQuestion?.indice ?? 0) - (selectedFacilitiesQuestion?.historical_index ?? 0)
      : null

  const facilitiesSuggestions = facilitiesSuggestionComments(summary?.comentarios ?? [])

  const selectedSectionEntersGeneral =
    selectedSection?.entra_general ?? false

  const chartData =
    summary?.secciones
      .filter((item) => item.indice !== null)
      .map((item) => ({
        nombre:
          canonicalSectionId(item.seccion_id) === "ESC_00_PERFIL"
            ? "Perfil de respuestas"
            : item.seccion_nombre,
        indice: item.indice,
      })) ?? []

  const askSia = () => {
    const question = normalizeAcademicLabel(siaQuestion)
    if (!question) {
      setSiaAnswer("Escribe una pregunta sobre los resultados actuales.")
      return
    }

    if (question.includes("coment") || question.includes("opinion") || question.includes("sugier")) {
      if (directionDirectorDrawerOpen && effectiveDirectionDirector) {
        setSiaAnswer(
          `${effectiveDirectionDirector.persona_evaluada}: ${directionDirectorCommentSummary}`,
        )
        return
      }
      if (staffPersonDrawerOpen && effectiveStaffOverviewPerson) {
        setSiaAnswer(
          `${effectiveStaffOverviewPerson.persona_evaluada}: ${staffOverviewCommentSummary}`,
        )
        return
      }
      const sectionComments =
        summary?.comentarios.filter(
          (comment) =>
            canonicalSectionId(comment.seccion_id) === effectiveSectionId &&
            comment.incluir_resumen &&
            isMeaningfulComment(comment.texto ?? ""),
        ) ?? []
      setSiaAnswer(summarizeCommentSet(sectionComments))
      return
    }

    if (question.includes("indice general") || question.includes("resultado general")) {
      setSiaAnswer(
        summary?.indice_general !== null && summary?.indice_general !== undefined
          ? `El índice general de la selección actual es ${summary.indice_general.toFixed(2)} sobre 100.`
          : "La selección actual no tiene un índice general calculable.",
      )
      return
    }

    if (question.includes("director") && directionDirectors.length > 0) {
      const conDatos = directionDirectors
        .filter((item) => item.indice !== null)
        .sort((a, b) => (b.indice ?? -1) - (a.indice ?? -1))
      if (question.includes("mejor") && conDatos[0]) {
        setSiaAnswer(
          `El director con mayor resultado en la selección es ${conDatos[0].persona_evaluada}, con ${conDatos[0].indice?.toFixed(2)}.`
        )
        return
      }
      if ((question.includes("menor") || question.includes("peor")) && conDatos.length > 0) {
        const item = conDatos[conDatos.length - 1]
        setSiaAnswer(
          `El director con menor resultado en la selección es ${item.persona_evaluada}, con ${item.indice?.toFixed(2)}.`
        )
        return
      }
      setSiaAnswer(
        `Hay ${directionDirectors.length} responsables de Dirección en la selección actual. Puedes preguntarme por el mejor resultado, el menor resultado o por sus comentarios.`
      )
      return
    }

    if (question.includes("asistente") || question.includes("coordinador") || question.includes("personal")) {
      const people = selectedSectionPeople
        .filter((item) => item.indice !== null)
        .sort((a, b) => (b.indice ?? -1) - (a.indice ?? -1))
      if (question.includes("mejor") && people[0]) {
        setSiaAnswer(
          `La persona con mayor resultado en esta sección es ${people[0].persona_evaluada}, con ${people[0].indice?.toFixed(2)}.`
        )
        return
      }
      if ((question.includes("menor") || question.includes("peor")) && people.length > 0) {
        const item = people[people.length - 1]
        setSiaAnswer(
          `La persona con menor resultado en esta sección es ${item.persona_evaluada}, con ${item.indice?.toFixed(2)}.`
        )
        return
      }
    }

    setSiaAnswer(
      `Estoy trabajando con ${filters.periodo || "todos los periodos"}, ${filters.modalidad || "todas las modalidades"}${filters.servicio ? ` y ${filters.servicio}` : ""}. Puedes preguntarme por el índice general, el mejor o menor director, asistentes/coordinadores o un resumen de comentarios.`
    )
  }

  return (
    <div className="flex min-h-screen bg-background">
      <Sidebar />

      <main className="flex flex-1 flex-col">
        <Header />

        <div className="mx-auto w-full max-w-7xl px-8 py-8">
          <p className="sia-eyebrow">
            CALIDAD ACADÉMICA
          </p>

          <div className="mt-2 flex items-end justify-between gap-6">
            <div>
              <h1 className="text-5xl sia-heading">
                Encuesta de Calidad
              </h1>

              <p className="mt-3 max-w-3xl text-muted-foreground">
                Monitorea la percepción estudiantil e identifica
                fortalezas, alertas y áreas de oportunidad institucional.
              </p>
            </div>

            {summary && (
              <div className="text-right text-sm text-muted-foreground">
                <p>Periodo {summary.filtros.seleccion.periodo}</p>
                <p>
                  Filas {summary.fila_inicio}–{summary.fila_fin}
                </p>
              </div>
            )}
          </div>

          {summary && (
            <div className="mt-8 rounded-2xl border bg-card p-5 shadow-sm">
              <div className="flex flex-col gap-4 xl:flex-row xl:items-end">
                <div className="min-w-[180px] flex-1">
                  <label className="mb-2 block text-xs font-medium text-muted-foreground">
                    Periodo
                  </label>

                  <select
                    value={filters.periodo ?? ""}
                    onChange={(event) =>
                      setFilters((current) => ({
                        ...current,
                        periodo: event.target.value,
                        modalidad: "Escolarizadas",
                        servicio: "",
                      }))
                    }
                    className="w-full rounded-xl border bg-background px-4 py-2.5 text-sm outline-none transition focus:border-primary/40 focus:ring-2 focus:ring-primary/10"
                  >
                    {summary.filtros.periodos.map((periodo) => (
                      <option key={periodo} value={periodo}>
                        {periodo}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="min-w-[220px] flex-1">
                  <label className="mb-2 block text-xs font-medium text-muted-foreground">
                    Modalidad
                  </label>

                  <select
                    value={filters.modalidad ?? ""}
                    onChange={(event) =>
                      setFilters((current) => ({
                        ...current,
                        modalidad: event.target.value,
                        servicio: "",
                      }))
                    }
                    className="w-full rounded-xl border bg-background px-4 py-2.5 text-sm outline-none transition focus:border-primary/40 focus:ring-2 focus:ring-primary/10"
                  >
                    {summary.filtros.modalidades.map((modalidad) => (
                      <option key={modalidad} value={modalidad}>
                        {modalidad}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="min-w-[280px] flex-[1.5]">
                  <label className="mb-2 block text-xs font-medium text-muted-foreground">
                    {filters.modalidad === "Prepa"
                    ? "Grado"
                    : filters.modalidad === "Virtuales"
                      ? "Programa"
                      : "Servicio / programa"}
                  </label>

                  <select
                    value={filters.servicio ?? ""}
                    onChange={(event) =>
                      setFilters((current) => ({
                        ...current,
                        servicio: event.target.value,
                      }))
                    }
                    className="w-full rounded-xl border bg-background px-4 py-2.5 text-sm outline-none transition focus:border-primary/40 focus:ring-2 focus:ring-primary/10"
                  >
                    <option value="">
                      {filters.modalidad === "Prepa"
                        ? "Todos los grados"
                        : filters.modalidad === "Virtuales"
                          ? "Todos los programas"
                          : "Todos los servicios"}
                    </option>

                    {summary.filtros.servicios.map((servicio) => (
                      <option key={servicio} value={servicio}>
                        {servicio}
                      </option>
                    ))}
                  </select>
                </div>

                <button
                  type="button"
                  onClick={() =>
                    setFilters({
                      periodo: "2026",
                      modalidad: "Escolarizadas",
                      servicio: "",
                    })
                  }
                  className="rounded-xl border bg-background px-5 py-2.5 text-sm font-medium transition hover:bg-muted"
                >
                  Limpiar filtros
                </button>

              <div className="ml-auto flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={() =>
                    downloadQualityReportPdf(
                      summary,
                      filters,
                      historico2025,
                      historico2026,
                    )
                  }
                  className="rounded-xl border bg-background px-5 py-2.5 text-sm font-medium transition hover:bg-muted"
                >
                  Descargar informe
                </button>

                <button
                  type="button"
                  onClick={() =>
                    downloadResponsesCsv(
                      summary as CalidadSummaryWithExport,
                      filters,
                    )
                  }
                  className="rounded-xl bg-primary px-5 py-2.5 text-sm font-medium text-primary-foreground transition hover:opacity-95"
                >
                  Descargar respuestas
                </button>
              </div>
              </div>
            </div>
          )}

          {summary && (
            <div className="mt-6 border-b">
              <div className="flex flex-wrap gap-1">
                {moduleTabs.map((tab) => (
                  <button
                    key={tab.id}
                    type="button"
                    onClick={() => setActiveTab(tab.id)}
                    className={`border-b-2 px-4 py-3 text-sm font-medium transition ${
                      activeTab === tab.id
                        ? "border-primary text-foreground"
                        : "border-transparent text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>
            </div>
          )}

          {loading && (
            <div className="mt-10 rounded-2xl border bg-card p-8">
              <p className="text-muted-foreground">
                Cargando resultados de Encuesta de Calidad...
              </p>
            </div>
          )}

          {error && (
            <div className="mt-10 rounded-2xl border border-destructive/30 bg-card p-8">
              <p className="font-medium">
                No fue posible cargar los resultados.
              </p>

              <p className="mt-2 text-sm text-muted-foreground">
                {error}
              </p>
            </div>
          )}

          {summary && !loading && activeTab === "resumen" && (
            <>
              <div className="mt-10 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                <div className="rounded-2xl border bg-card p-6 shadow-sm">
                  <p className="text-sm text-muted-foreground">
                    Índice general
                  </p>

                  <p className="mt-3 text-4xl font-semibold tracking-tight">
                    {summary.indice_general?.toFixed(2) ?? "—"}
                  </p>

                  <div className="mt-3 flex items-center justify-between gap-3">
                    <p className="text-xs text-muted-foreground">
                      Escala institucional 0–100
                    </p>

                    <span className={`rounded-full border px-3 py-1 text-xs font-medium ${getNivelClasses(summary.indice_general)}`}>
                      {getNivel(summary.indice_general)}
                    </span>
                  </div>
                </div>

                <div className="rounded-2xl border bg-card p-6 shadow-sm">
                  <p className="text-sm text-muted-foreground">
                    Encuestas
                  </p>

                  <p className="mt-3 text-4xl font-semibold tracking-tight">
                    {summary.encuestas}
                  </p>

                  <p className="mt-2 text-xs text-muted-foreground">
                    Respuestas detectadas en el periodo
                  </p>
                </div>

                <div className="rounded-2xl border bg-card p-6 shadow-sm">
                  <p className="text-sm text-muted-foreground">
                    Respuestas KPI
                  </p>

                  <p className="mt-3 text-4xl font-semibold tracking-tight">
                    {summary.respuestas_kpi_validas.toLocaleString()}
                  </p>

                  <p className="mt-2 text-xs text-muted-foreground">
                    Respuestas válidas para análisis
                  </p>
                </div>

                <div className="rounded-2xl border bg-card p-6 shadow-sm">
                  <p className="text-sm text-muted-foreground">
                    Control de calidad
                  </p>

                  <p className="mt-3 text-4xl font-semibold tracking-tight">
                    {summary.qa_total}
                  </p>

                  <p className="mt-2 text-xs text-muted-foreground">
                    Incidencias detectadas por QA
                  </p>
                </div>
              </div>

              <div className="mt-6 rounded-2xl border bg-card p-6 shadow-sm">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">
                    RESULTADOS POR SECCIÓN
                  </p>

                  <h2 className="mt-1 text-2xl font-semibold tracking-tight">
                    Índice de calidad
                  </h2>

                  <p className="mt-1 text-sm text-muted-foreground">
                    Comparativo preliminar de las secciones con respuestas disponibles.
                  </p>
                </div>

                <div className="mt-4 flex flex-wrap gap-2 text-xs">
                  <span className="rounded-full border border-red-200 bg-red-50 px-3 py-1 text-red-700">
                    Crítico · 0–60
                  </span>

                  <span className="rounded-full border border-amber-200 bg-amber-50 px-3 py-1 text-amber-700">
                    Atención · &gt;60–75
                  </span>

                  <span className="rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-emerald-700">
                    Adecuado · &gt;75–85
                  </span>

                  <span className="rounded-full border border-emerald-300 bg-emerald-100 px-3 py-1 text-emerald-800">
                    Fortaleza · &gt;85–100
                  </span>
                </div>

                <div className="mt-6 h-[460px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={chartData}
                      layout="vertical"
                      margin={{
                        top: 0,
                        right: 30,
                        bottom: 0,
                        left: 40,
                      }}
                    >
                      <CartesianGrid
                        strokeDasharray="3 3"
                        horizontal={false}
                      />

                      <XAxis
                        type="number"
                        domain={[0, 100]}
                      />

                      <YAxis
                        type="category"
                        dataKey="nombre"
                        width={220}
                        tick={{
                          fontSize: 12,
                        }}
                      />

                      <Tooltip />

                      <Bar
                        dataKey="indice"
                        name="Índice"
                        radius={[0, 6, 6, 0]}
                      >
                        {chartData.map((item) => (
                          <Cell
                            key={item.nombre}
                            fill={getBarColor(item.indice)}
                          />
                        ))}

                        <LabelList
                          dataKey="indice"
                          position="right"
                          formatter={(value) =>
                            Number(value).toFixed(2)
                          }
                          className="fill-foreground text-xs font-medium"
                        />
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </>
          )}

          {summary && !loading && activeTab === "secciones" && (
            <div className="mt-8 space-y-6">
              <div className="rounded-2xl border bg-card p-6 shadow-sm">
                <div className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
                  <div>
                    <p className="sia-eyebrow">
                      ANÁLISIS DETALLADO
                    </p>

                    <h2 className="mt-2 text-2xl font-semibold">
                      Resultados por sección
                    </h2>

                    <p className="mt-2 text-sm text-muted-foreground">
                      Consulta el desempeño de cada sección y el
                      detalle de las preguntas que la integran.
                    </p>
                  </div>

                  <div className="w-full lg:max-w-md">
                    <label className="mb-2 block text-xs font-medium text-muted-foreground">
                      Sección
                    </label>

                    <select
                      value={effectiveSectionId}
                      onChange={(event) =>
                        setSelectedSectionId(event.target.value)
                      }
                      className="w-full rounded-xl border bg-background px-4 py-2.5 text-sm outline-none transition focus:border-primary/40 focus:ring-2 focus:ring-primary/10"
                    >
                      {availableSections.map((section) => (
                        <option
                          key={section.seccion_id}
                          value={canonicalSectionId(section.seccion_id)}
                        >
                          {canonicalSectionId(section.seccion_id) === "ESC_00_PERFIL"
                            ? "Perfil de respuestas"
                            : section.seccion_nombre}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>

              {effectiveSectionId === "ESC_00_PERFIL" && (
                <div className="space-y-6">
                  <div className="grid gap-4 md:grid-cols-3">
                    <div className="rounded-2xl border bg-card p-6 shadow-sm">
                      <p className="text-sm text-muted-foreground">
                        Encuestas en la selección
                      </p>

                      <p className="mt-3 text-4xl font-semibold tracking-tight">
                        {summary?.encuestas.toLocaleString() ?? "0"}
                      </p>

                      <p className="mt-2 text-xs text-muted-foreground">
                        Respuestas consideradas con los filtros actuales
                      </p>
                    </div>

                    <div className="rounded-2xl border bg-card p-6 shadow-sm">
                      <p className="text-sm text-muted-foreground">
                        Tipo de análisis
                      </p>

                      <p className="mt-3 text-2xl font-semibold tracking-tight">
                        Descriptivo
                      </p>

                      <p className="mt-2 text-xs text-muted-foreground">
                        Permite caracterizar y segmentar las respuestas
                      </p>
                    </div>

                    <div className="rounded-2xl border bg-card p-6 shadow-sm">
                      <p className="text-sm text-muted-foreground">
                        Participación en el Índice
                      </p>

                      <p className="mt-3 text-2xl font-semibold tracking-tight">
                        No aplica
                      </p>

                      <p className="mt-2 text-xs text-muted-foreground">
                        Los datos de perfil no modifican el Índice General
                      </p>
                    </div>
                  </div>

                  {summary?.perfil_respuestas && (
                    <div className="grid gap-6 xl:grid-cols-2">
                      <div className="rounded-2xl border bg-card p-6 shadow-sm">
                        <div className="flex items-start justify-between gap-4">
                          <div>
                            <p className="sia-eyebrow">
                              DISTRIBUCIÓN
                            </p>

                            <h3 className="mt-2 text-xl font-semibold">
                              Servicio / programa
                            </h3>

                            <p className="mt-1 text-sm text-muted-foreground">
                              Distribución de las encuestas por programa académico.
                            </p>
                          </div>

                          <div className="text-right">
                            <p className="text-2xl font-semibold">
                              {
                                summary.perfil_respuestas.servicio_programa
                                  .con_dato
                              }
                            </p>
                            <p className="text-xs text-muted-foreground">
                              con dato
                            </p>
                          </div>
                        </div>

                        <div className="mt-6 space-y-4">
                          {summary.perfil_respuestas.servicio_programa
                            .distribucion
                            .slice(0, 12)
                            .map((item) => {
                              const ancho = item.porcentaje

                              return (
                                <div key={item.valor}>
                                  <div className="flex items-start justify-between gap-4 text-sm">
                                    <span className="max-w-[75%] font-medium">
                                      {item.valor}
                                    </span>

                                    <span className="shrink-0 text-muted-foreground">
                                      {item.cantidad} ?{" "}
                                      {item.porcentaje.toFixed(1)}%
                                    </span>
                                  </div>

                                  <div className="mt-2 h-2 overflow-hidden rounded-full bg-muted">
                                    <div
                                      className="h-full rounded-full bg-primary"
                                      style={{
                                        width: `${ancho}%`,
                                      }}
                                    />
                                  </div>
                                </div>
                              )
                            })}
                        </div>

                        {summary.perfil_respuestas.servicio_programa
                          .distribucion.length > 12 && (
                          <p className="mt-5 text-xs text-muted-foreground">
                            Se muestran los 12 programas con mayor número
                            de respuestas.
                          </p>
                        )}
                      </div>

                      <div className="rounded-2xl border bg-card p-6 shadow-sm">
                        <div className="flex items-start justify-between gap-4">
                          <div>
                            <p className="sia-eyebrow">
                              SEGMENTACIÓN
                            </p>

                            <h3 className="mt-2 text-xl font-semibold">
                              Modalidad reportada
                            </h3>

                            <p className="mt-1 text-sm text-muted-foreground">
                              Distribución entre las encuestas que contienen
                              modalidad explícita.
                            </p>
                          </div>

                          <div className="text-right">
                            <p className="text-2xl font-semibold">
                              {summary.perfil_respuestas.modalidad_detalle
                                .cobertura_pct.toFixed(1)}
                              %
                            </p>
                            <p className="text-xs text-muted-foreground">
                              cobertura
                            </p>
                          </div>
                        </div>

                        <div className="mt-5 rounded-xl border bg-muted/30 px-4 py-3">
                          <p className="text-sm font-medium">
                            Cobertura del dato
                          </p>

                          <p className="mt-1 text-sm text-muted-foreground">
                            {
                              summary.perfil_respuestas.modalidad_detalle
                                .con_dato
                            }{" "}
                            de{" "}
                            {
                              summary.perfil_respuestas.modalidad_detalle
                                .total_encuestas
                            }{" "}
                            encuestas incluyen modalidad explícita.
                          </p>
                        </div>

                        {summary.perfil_respuestas.modalidad_detalle
                          .distribucion.length > 0 ? (
                          <div className="mt-6 space-y-4">
                            {summary.perfil_respuestas.modalidad_detalle
                              .distribucion.map((item) => (
                                <div
                                  key={item.valor}
                                  className="rounded-xl border p-4"
                                >
                                  <div className="flex items-center justify-between gap-4">
                                    <div>
                                      <p className="font-medium">
                                        {item.valor}
                                      </p>

                                      <p className="mt-1 text-xs text-muted-foreground">
                                        {item.cantidad} respuestas con dato
                                      </p>
                                    </div>

                                    <p className="text-2xl font-semibold">
                                      {item.porcentaje.toFixed(1)}%
                                    </p>
                                  </div>
                                </div>
                              ))}
                          </div>
                        ) : (
                          <p className="mt-6 text-sm text-muted-foreground">
                            No hay datos de modalidad disponibles para esta
                            selección.
                          </p>
                        )}

                        {summary.perfil_respuestas.modalidad_detalle
                          .cobertura_pct < 50 && (
                          <div className="mt-5 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3">
                            <p className="text-sm font-medium text-amber-900">
                              Cobertura limitada
                            </p>

                            <p className="mt-1 text-xs text-amber-800">
                              Esta distribución describe solo las encuestas
                              con modalidad explícita; no debe interpretarse
                              como representativa del total.
                            </p>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {selectedSection &&
                effectiveSectionId !== "ESC_00_PERFIL" && (
                <div className="grid gap-4 md:grid-cols-3">
                  <div className="rounded-2xl border bg-card p-6 shadow-sm">
                    <p className="text-sm text-muted-foreground">
                      Promedio de la sección
                    </p>

                    <p className="mt-3 text-4xl font-semibold tracking-tight">
                      {selectedSection.indice?.toFixed(2) ?? "—"}
                    </p>

                    <div className="mt-3">
                      <span
                        className={`rounded-full border px-3 py-1 text-xs font-medium ${getNivelClasses(
                          selectedSection.indice,
                        )}`}
                      >
                        {getNivel(selectedSection.indice)}
                      </span>
                    </div>
                  </div>

                  <div className="rounded-2xl border bg-card p-6 shadow-sm">
                    <p className="text-sm text-muted-foreground">
                      {effectiveSectionId === "ESC_01_PERSONAL"
                        ? "Criterios evaluados"
                        : "Reactivos evaluados"}
                    </p>

                    <p className="mt-3 text-4xl font-semibold tracking-tight">
                      {effectiveSectionId === "ESC_01_PERSONAL"
                        ? staffSummary?.criterios_evaluados ?? staffUniqueCriteriaCount
                        : selectedSection.preguntas_con_datos}
                    </p>

                    <p className="mt-2 text-xs text-muted-foreground">
                      {effectiveSectionId === "ESC_01_PERSONAL"
                        ? "Criterios únicos considerados en la evaluación"
                        : `${selectedSection.preguntas_con_datos} de ${selectedSection.reactivos_indice_total} reactivos cuentan actualmente con respuestas`}
                    </p>
                  </div>

                  <div className="rounded-2xl border bg-card p-6 shadow-sm">
                    <p className="text-sm text-muted-foreground">
                      {effectiveSectionId === "ESC_01_PERSONAL"
                        ? "Evaluaciones"
                        : "Respuestas registradas"}
                    </p>

                    <p className="mt-3 text-4xl font-semibold tracking-tight">
                      {effectiveSectionId === "ESC_01_PERSONAL"
                        ? (staffSummary?.evaluaciones ?? 0).toLocaleString()
                        : selectedSectionQuestions
                            .reduce(
                              (total, question) =>
                                total + question.n,
                              0,
                            )
                            .toLocaleString()}
                    </p>

                    <p className="mt-2 text-xs text-muted-foreground">
                      {effectiveSectionId === "ESC_01_PERSONAL"
                        ? `${staffSummary?.personas_evaluadas ?? selectedSectionPeople.length} personas cuentan con evaluación válida`
                        : "Acumuladas en los indicadores de esta sección"}
                    </p>
                  </div>
                </div>
              )}

              {selectedSection && effectiveSectionId !== "ESC_00_PERFIL" && (
                <div className="rounded-2xl border bg-card p-5 shadow-sm">
                  <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
                    <div>
                      <p className="sia-eyebrow">COMPARATIVO HISTÓRICO</p>
                      <h3 className="mt-2 text-lg font-semibold">
                        Evolución de la sección · 2025 vs 2026
                      </h3>
                      <p className="mt-1 text-xs text-muted-foreground">
                        Se compara únicamente información metodológicamente equivalente.
                      </p>
                    </div>

                    {selectedSection2025?.indice !== null &&
                    selectedSection2025?.indice !== undefined &&
                    selectedSection2026?.indice !== null &&
                    selectedSection2026?.indice !== undefined ? (
                      <div className="grid min-w-0 gap-3 sm:grid-cols-4 lg:min-w-[560px]">
                        <div className="rounded-xl border bg-background px-4 py-3">
                          <p className="text-[11px] text-muted-foreground">2025</p>
                          <p className="mt-1 text-xl font-semibold">{selectedSection2025.indice.toFixed(2)}</p>
                        </div>
                        <div className="rounded-xl border bg-background px-4 py-3">
                          <p className="text-[11px] text-muted-foreground">2026</p>
                          <p className="mt-1 text-xl font-semibold">{selectedSection2026.indice.toFixed(2)}</p>
                        </div>
                        <div className="rounded-xl border bg-background px-4 py-3">
                          <p className="text-[11px] text-muted-foreground">Variación</p>
                          <p className={`mt-1 text-xl font-semibold ${
                            selectedSectionHistoricalDelta === null
                              ? ""
                              : selectedSectionHistoricalDelta >= 0
                                ? "text-emerald-700"
                                : "text-red-700"
                          }`}>
                            {selectedSectionHistoricalDelta === null
                              ? "—"
                              : `${selectedSectionHistoricalDelta >= 0 ? "+" : ""}${selectedSectionHistoricalDelta.toFixed(2)}`}
                          </p>
                        </div>
                        <div className="rounded-xl border bg-background px-4 py-3">
                          <p className="text-[11px] text-muted-foreground">Reactivos comparables</p>
                          <p className="mt-1 text-xl font-semibold">{selectedSectionComparableQuestions}</p>
                        </div>
                      </div>
                    ) : (
                      <div className="rounded-xl border bg-muted/20 px-4 py-3 lg:max-w-sm">
                        <p className="text-sm font-medium">Sin antecedente comparable en 2025</p>
                        <p className="mt-1 text-xs leading-5 text-muted-foreground">
                          Esta sección no cuenta con un índice equivalente válido en el instrumento 2025. No se genera una comparación artificial.
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {effectiveSectionId === "ESC_01_PERSONAL" ? (
                <div className="space-y-6">
                  <div className="rounded-2xl border bg-card p-6 shadow-sm">
                    <div className="flex flex-col gap-5 lg:flex-row lg:items-start lg:justify-between">
                      <div>
                        <p className="sia-eyebrow">
                          EXPERIENCIA DE ATENCIÓN
                        </p>

                        <h3 className="mt-2 text-xl font-semibold">
                          Asistentes, coordinadores y responsables
                        </h3>

                        <p className="mt-2 max-w-3xl text-sm text-muted-foreground">
                          Consulta el resultado por carrera o directamente por persona.
                          El rol se muestra como contexto junto al nombre, sin generar
                          una vista adicional que duplique información.
                        </p>
                      </div>

                      <div className="inline-flex rounded-xl border bg-muted/20 p-1">
                        {[
                          ["carrera", "Por carrera"],
                          ["persona", "Por persona"],
                        ].map(([id, label]) => (
                          <button
                            key={id}
                            type="button"
                            onClick={() =>
                              setStaffView(
                                id as
                                  | "carrera"
                                  | "persona",
                              )
                            }
                            className={`rounded-lg px-4 py-2 text-sm font-medium transition ${
                              staffView === id
                                ? "bg-background shadow-sm"
                                : "text-muted-foreground hover:text-foreground"
                            }`}
                          >
                            {label}
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>

                  {staffView === "carrera" && (
                    <div className="rounded-2xl border bg-card p-6 shadow-sm">
                      <div>
                        <p className="text-sm font-semibold">
                          Resultado por carrera y modalidad
                        </p>

                        <p className="mt-1 max-w-3xl text-xs text-muted-foreground">
                          Selecciona una población para abrir su detalle. Las
                          modalidades se mantienen separadas para no mezclar muestras.
                        </p>
                      </div>

                      <div className="mt-5 grid max-h-[520px] gap-3 overflow-y-auto pr-2 lg:grid-cols-2">
                        {staffCareerModalitiesForSection.map((career) => {
                          const tieneDatos =
                            career.tiene_datos && career.indice !== null

                          return (
                            <button
                              key={`${career.carrera_base}|||${career.modalidad_academica}`}
                              type="button"
                              onClick={() => {
                                setSelectedStaffCareer(career.carrera_base)
                                setSelectedStaffModality(
                                  career.modalidad_academica,
                                )
                                setSelectedStaffPersonKey("")
                                setStaffDrawerOpen(true)
                              }}
                              className="rounded-xl border bg-background p-4 text-left transition hover:border-primary/40 hover:bg-muted/20"
                            >
                              <div className="flex items-start justify-between gap-4">
                                <div className="min-w-0 flex-1">
                                  <p className="font-semibold">
                                    {career.carrera_base}
                                  </p>

                                  <div className="mt-2 flex flex-wrap items-center gap-2">
                                    <span className="rounded-full border px-2.5 py-1 text-[10px] font-medium text-muted-foreground">
                                      {career.modalidad_academica}
                                    </span>

                                    <span className="text-xs text-muted-foreground">
                                      {career.evaluaciones.toLocaleString()} evaluaciones
                                    </span>

                                    <span className="text-xs text-muted-foreground">
                                      · {career.personas_evaluadas} {career.personas_evaluadas === 1 ? "persona evaluada" : "personas evaluadas"}
                                    </span>
                                  </div>
                                </div>

                                <div className="shrink-0 text-right">
                                  {tieneDatos ? (
                                    <>
                                      <p className="text-2xl font-semibold">
                                        {career.indice?.toFixed(2)}
                                      </p>
                                      <span
                                        className={`mt-1 inline-flex rounded-full border px-2 py-0.5 text-[10px] font-medium ${getNivelClasses(
                                          career.indice,
                                        )}`}
                                      >
                                        {getNivel(career.indice)}
                                      </span>
                                    </>
                                  ) : (
                                    <span className="inline-flex rounded-full border px-2.5 py-1 text-[10px] font-medium text-muted-foreground">
                                      Sin respuestas
                                    </span>
                                  )}
                                </div>
                              </div>

                              {tieneDatos && (
                                <div className="mt-4 h-2 overflow-hidden rounded-full bg-muted">
                                  <div
                                    className="h-full rounded-full bg-primary"
                                    style={{
                                      width: `${Math.max(
                                        0,
                                        Math.min(100, career.indice ?? 0),
                                      )}%`,
                                    }}
                                  />
                                </div>
                              )}
                            </button>
                          )
                        })}
                      </div>
                    </div>
                  )}

                  {staffView === "persona" && (
                    <div className="rounded-2xl border bg-card p-6 shadow-sm">
                      <div>
                        <p className="text-sm font-semibold">
                          Comparativo institucional por persona
                        </p>
                        <p className="mt-1 text-xs text-muted-foreground">
                          Selecciona una persona para abrir al costado su resultado,
                          los cuatro criterios evaluados y los comentarios asociados.
                        </p>
                      </div>

                      <div className="mt-5 grid max-h-[520px] gap-3 overflow-y-auto pr-2 md:grid-cols-2">
                        {[...selectedSectionPeople]
                          .sort(
                            (a, b) =>
                              (b.indice ?? -1) - (a.indice ?? -1),
                          )
                          .map((person) => {
                            const personKey = makeStaffPersonKey(
                              person.persona_evaluada,
                              person.rol_evaluado,
                            )

                            return (
                              <button
                                key={personKey}
                                type="button"
                                onClick={() => {
                                  setSelectedStaffOverviewPersonKey(personKey)
                                  setStaffPersonDrawerOpen(true)
                                }}
                                className="rounded-xl border bg-background p-4 text-left transition hover:border-primary/40 hover:bg-muted/20"
                              >
                                <div className="flex items-start justify-between gap-4">
                                  <div className="min-w-0">
                                    <p className="text-xs text-muted-foreground">
                                      {person.rol_evaluado}
                                    </p>
                                    <p className="mt-1 font-semibold">
                                      {person.persona_evaluada}
                                    </p>
                                    <p className="mt-1 text-xs text-muted-foreground">
                                      {person.evaluaciones.toLocaleString()} evaluaciones
                                      {" · "}
                                      {person.preguntas_con_datos} criterios
                                    </p>
                                  </div>
                                  <div className="shrink-0 text-right">
                                    <p className="text-xs text-muted-foreground">
                                      Promedio
                                    </p>
                                    <p className="mt-1 text-2xl font-semibold">
                                      {person.indice?.toFixed(2) ?? "—"}
                                    </p>
                                  </div>
                                </div>
                              </button>
                            )
                          })}
                      </div>
                    </div>
                  )}

                  {staffPersonDrawerOpen && effectiveStaffOverviewPerson && (
                    <div className="fixed inset-0 z-[125] flex justify-end bg-black/20">
                      <button
                        type="button"
                        aria-label="Cerrar detalle de persona"
                        onClick={() => setStaffPersonDrawerOpen(false)}
                        className="absolute inset-0"
                      />

                      <aside className="relative z-10 flex h-full w-full max-w-2xl flex-col border-l bg-background shadow-2xl">
                        <div className="border-b p-6">
                          <div className="flex items-start justify-between gap-4">
                            <div>
                              <p className="sia-eyebrow">
                                DETALLE POR PERSONA
                              </p>
                              <p className="mt-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
                                {effectiveStaffOverviewPerson.rol_evaluado}
                              </p>
                              <h3 className="mt-1 text-2xl font-semibold">
                                {effectiveStaffOverviewPerson.persona_evaluada}
                              </h3>
                              <p className="mt-2 text-sm text-muted-foreground">
                                Resultado institucional dentro de los filtros seleccionados.
                              </p>
                            </div>

                            <button
                              type="button"
                              onClick={() => setStaffPersonDrawerOpen(false)}
                              className="rounded-xl border px-3 py-2 text-sm"
                            >
                              Cerrar
                            </button>
                          </div>
                        </div>

                        <div className="flex-1 overflow-y-auto p-6">
                          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                            <div className="rounded-xl border bg-card p-4">
                              <p className="text-xs text-muted-foreground">
                                Promedio
                              </p>
                              <p className="mt-1 text-2xl font-semibold">
                                {effectiveStaffOverviewPerson.indice?.toFixed(2) ?? "—"}
                              </p>
                              <span
                                className={`mt-2 inline-flex rounded-full border px-2 py-0.5 text-[10px] font-medium ${getNivelClasses(
                                  effectiveStaffOverviewPerson.indice,
                                )}`}
                              >
                                {getNivel(effectiveStaffOverviewPerson.indice)}
                              </span>
                            </div>

                            <div className="rounded-xl border bg-card p-4">
                              <p className="text-xs text-muted-foreground">
                                Evaluaciones
                              </p>
                              <p className="mt-1 text-2xl font-semibold">
                                {effectiveStaffOverviewPerson.evaluaciones.toLocaleString()}
                              </p>
                            </div>

                            <div className="rounded-xl border bg-card p-4">
                              <p className="text-xs text-muted-foreground">
                                Criterios
                              </p>
                              <p className="mt-1 text-2xl font-semibold">
                                {staffOverviewQuestions.length}
                              </p>
                            </div>

                            <div className="rounded-xl border bg-card p-4">
                              <p className="text-xs text-muted-foreground">
                                Comentarios útiles
                              </p>
                              <p className="mt-1 text-2xl font-semibold">
                                {staffOverviewMeaningfulComments.length}
                              </p>
                            </div>
                          </div>

                          {staffOverviewAssignments.length > 0 && (
                            <div className="mt-7">
                              <p className="text-sm font-semibold">
                                Carreras asignadas
                              </p>
                              <div className="mt-3 flex flex-wrap gap-2">
                                {staffOverviewAssignments.map((assignment) => (
                                  <span
                                    key={`${assignment.carrera_base}|||${assignment.modalidad_academica}`}
                                    className="rounded-full border bg-card px-3 py-1.5 text-xs"
                                  >
                                    {assignment.carrera_base}
                                    {assignment.modalidad_academica
                                      ? ` · ${assignment.modalidad_academica}`
                                      : ""}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}

                          <div className="mt-7 rounded-2xl border border-dashed bg-muted/20 p-5">
                            <div className="flex flex-wrap items-center gap-2">
                              <p className="text-sm font-semibold">
                                Resumen de comentarios
                              </p>
                              <span className="rounded-full border bg-background px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                                En construcción
                              </span>
                            </div>
                            <p className="mt-2 text-sm leading-6 text-muted-foreground">
                              Este espacio incorporará una síntesis semántica de los comentarios asociados a la persona seleccionada. La versión final identificará temas recurrentes, menciones relevantes y oportunidades de mejora sin perder el contexto de las respuestas originales.
                            </p>
                          </div>

                          <div className="mt-7">
                            <p className="text-sm font-semibold">
                              Resultado por criterio
                            </p>
                            <p className="mt-1 text-xs text-muted-foreground">
                              Los cuatro componentes de atención se muestran por separado
                              para evitar que el promedio oculte diferencias.
                            </p>

                            <div className="mt-4 space-y-3">
                              {staffOverviewQuestions.map((question) => (
                                <div
                                  key={question.pregunta_id}
                                  className="rounded-xl border bg-card p-4"
                                >
                                  <div className="flex items-start justify-between gap-4">
                                    <div>
                                      <p className="text-sm font-semibold">
                                        {question.texto_corto}
                                      </p>
                                      <p className="mt-1 text-xs text-muted-foreground">
                                        {question.n.toLocaleString()} respuestas
                                      </p>
                                    </div>
                                    <div className="text-right">
                                      <p className="text-[10px] uppercase tracking-wide text-muted-foreground">2026</p>
                                      <p className="text-xl font-semibold">
                                        {question.indice?.toFixed(2) ?? "—"}
                                      </p>
                                    </div>
                                  </div>

                                  <div className="mt-3 rounded-lg border bg-background px-3 py-2 text-xs">
                                    <span className="text-muted-foreground">2025: </span>
                                    <span className="font-medium">Sin antecedente comparable por persona</span>
                                  </div>

                                  <div className="mt-4 flex h-2 overflow-hidden rounded-full bg-muted">
                                    <div
                                      className="bg-emerald-500"
                                      style={{
                                        width: `${question.favorable_pct ?? 0}%`,
                                      }}
                                    />
                                    <div
                                      className="bg-amber-400"
                                      style={{
                                        width: `${question.neutral_pct ?? 0}%`,
                                      }}
                                    />
                                    <div
                                      className="bg-red-500"
                                      style={{
                                        width: `${question.desfavorable_pct ?? 0}%`,
                                      }}
                                    />
                                  </div>

                                  <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[10px]">
                                    <span className="text-emerald-700">
                                      Favorable {question.favorable_pct?.toFixed(1) ?? "0.0"}%
                                    </span>
                                    <span className="text-amber-700">
                                      Neutral {question.neutral_pct?.toFixed(1) ?? "0.0"}%
                                    </span>
                                    <span className="text-red-700">
                                      Desfavorable {question.desfavorable_pct?.toFixed(1) ?? "0.0"}%
                                    </span>
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>

                          <div className="mt-7 rounded-2xl border bg-muted/25 p-5">
                            <p className="sia-eyebrow">
                              LECTURA CUALITATIVA
                            </p>
                            <h4 className="mt-2 text-base font-semibold">
                              Interpretación general
                            </h4>
                            <p className="mt-2 text-sm leading-6 text-muted-foreground">
                              {staffOverviewInterpretation}
                            </p>
                            <p className="mt-3 text-[11px] leading-5 text-muted-foreground">
                              Esta lectura combina el nivel cuantitativo con la existencia
                              de comentarios; no sustituye la revisión del texto original.
                            </p>
                          </div>

                          <div className="mt-7">
                            <div className="flex items-end justify-between gap-4">
                              <div>
                                <p className="text-sm font-semibold">
                                  Comentarios de estudiantes
                                </p>
                                <p className="mt-1 text-xs text-muted-foreground">
                                  Se omiten respuestas vacías, de relleno o sin contenido
                                  sustantivo para que la lectura sea útil.
                                </p>
                              </div>
                              <span className="rounded-full border px-2.5 py-1 text-[10px] font-medium text-muted-foreground">
                                {staffOverviewMeaningfulComments.length} de {staffOverviewComments.length}
                              </span>
                            </div>

                            {staffOverviewMeaningfulComments.length > 0 ? (
                              <div className="mt-4 space-y-3">
                                {staffOverviewMeaningfulComments.map(
                                  (comment, index) => (
                                    <div
                                      key={`${comment.fila_origen}-${comment.pregunta_id}-${index}`}
                                      className="rounded-xl border bg-card p-4"
                                    >
                                      <p className="text-sm leading-6">
                                        {comment.texto}
                                      </p>
                                      <div className="mt-3 flex flex-wrap gap-2 text-[10px] text-muted-foreground">
                                        {comment.servicio_programa && (
                                          <span className="rounded-full border px-2 py-0.5">
                                            {comment.servicio_programa}
                                          </span>
                                        )}
                                        <span className="rounded-full border px-2 py-0.5">
                                          Respuesta abierta
                                        </span>
                                      </div>
                                    </div>
                                  ),
                                )}
                              </div>
                            ) : (
                              <div className="mt-4 rounded-xl border bg-muted/20 p-5">
                                <p className="text-sm font-medium">
                                  Sin comentarios sustantivos
                                </p>
                                <p className="mt-1 text-xs leading-5 text-muted-foreground">
                                  No hay texto abierto útil para esta persona dentro de
                                  los filtros actuales. El resultado se interpreta a partir
                                  de las evaluaciones registradas.
                                </p>
                              </div>
                            )}
                          </div>
                        </div>
                      </aside>
                    </div>
                  )}

                  {staffDrawerOpen && selectedStaffModalitySummary && (
                    <div className="fixed inset-0 z-[120] flex justify-end bg-black/20">
                      <button
                        type="button"
                        aria-label="Cerrar detalle de población"
                        onClick={() => setStaffDrawerOpen(false)}
                        className="absolute inset-0"
                      />

                      <aside className="relative z-10 flex h-full w-full max-w-2xl flex-col border-l bg-background shadow-2xl">
                        <div className="border-b p-6">
                          <div className="flex items-start justify-between gap-4">
                            <div>
                              <p className="sia-eyebrow">
                                DETALLE DE POBLACIÓN
                              </p>

                              <div className="mt-2 flex flex-wrap items-center gap-2">
                                <h3 className="text-2xl font-semibold">
                                  {effectiveStaffCareer}
                                </h3>
                                <span className="rounded-full border px-3 py-1 text-xs font-medium text-muted-foreground">
                                  {effectiveStaffModality}
                                </span>
                              </div>

                              <p className="mt-2 text-sm text-muted-foreground">
                                Asistentes, coordinadores y responsables de esta
                                población.
                              </p>
                            </div>

                            <button
                              type="button"
                              onClick={() => setStaffDrawerOpen(false)}
                              className="rounded-xl border px-3 py-2 text-sm"
                            >
                              Cerrar
                            </button>
                          </div>
                        </div>

                        <div className="flex-1 overflow-y-auto p-6">
                          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                            <div className="rounded-xl border bg-card p-4">
                              <p className="text-xs text-muted-foreground">Promedio de la población</p>
                              <p className="mt-1 text-2xl font-semibold">
                                {selectedStaffModalitySummary.indice?.toFixed(2) ?? "—"}
                              </p>
                            </div>
                            <div className="rounded-xl border bg-card p-4">
                              <p className="text-xs text-muted-foreground">Evaluaciones</p>
                              <p className="mt-1 text-2xl font-semibold">
                                {selectedStaffModalitySummary.evaluaciones.toLocaleString()}
                              </p>
                            </div>
                            <div className="rounded-xl border bg-card p-4">
                              <p className="text-xs text-muted-foreground">Personal asignado</p>
                              <p className="mt-1 text-2xl font-semibold">
                                {selectedStaffModalitySummary.personas_asignadas}
                              </p>
                            </div>
                            <div className="rounded-xl border bg-card p-4">
                              <p className="text-xs text-muted-foreground">Con respuestas</p>
                              <p className="mt-1 text-2xl font-semibold">
                                {selectedStaffModalitySummary.personas_evaluadas}
                              </p>
                            </div>
                          </div>

                          {staffPeopleForCareer.length > 0 ? (
                            <div className="mt-7">
                              <p className="text-sm font-semibold">
                                Personas evaluadas
                              </p>
                              <p className="mt-1 text-xs text-muted-foreground">
                                Selecciona una persona para consultar sus reactivos.
                              </p>

                              <div className="mt-4 grid gap-2 sm:grid-cols-2">
                                {staffPeopleForCareer.map((person) => {
                                  const personKey = makeStaffPersonKey(
                                    person.persona_evaluada,
                                    person.rol_evaluado,
                                  )
                                  const selected =
                                    effectiveStaffPersonKey === personKey

                                  return (
                                    <button
                                      key={personKey}
                                      type="button"
                                      onClick={() =>
                                        setSelectedStaffPersonKey(personKey)
                                      }
                                      className={`rounded-xl border p-4 text-left transition ${
                                        selected
                                          ? "border-primary bg-primary/5"
                                          : "bg-card hover:bg-muted/30"
                                      }`}
                                    >
                                      <div className="flex items-start justify-between gap-3">
                                        <div>
                                          <p className="text-xs text-muted-foreground">
                                            {person.rol_evaluado}
                                          </p>
                                          <p className="mt-1 font-semibold">
                                            {person.persona_evaluada}
                                          </p>
                                          <p className="mt-1 text-[11px] text-muted-foreground">
                                            {person.evaluaciones} evaluaciones
                                          </p>
                                        </div>
                                        <p className="text-lg font-semibold">
                                          {person.indice?.toFixed(2) ?? "—"}
                                        </p>
                                      </div>
                                    </button>
                                  )
                                })}
                              </div>
                            </div>
                          ) : (
                            <div className="mt-7 rounded-xl border bg-muted/30 p-5">
                              <p className="text-sm font-medium">
                                Sin personas evaluadas
                              </p>
                              <p className="mt-1 text-xs text-muted-foreground">
                                Esta población existe, pero no tiene respuestas válidas
                                asociadas a personal en los filtros actuales.
                              </p>
                            </div>
                          )}

                          {effectiveStaffPerson && (
                            <div className="mt-7 border-t pt-6">
                              <div className="flex items-start justify-between gap-4">
                                <div>
                                  <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                                    {effectiveStaffPerson.rol_evaluado}
                                  </p>
                                  <h4 className="mt-1 text-xl font-semibold">
                                    {effectiveStaffPerson.persona_evaluada}
                                  </h4>
                                </div>
                                <div className="text-right">
                                  <p className="text-xs text-muted-foreground">
                                    Promedio de la persona
                                  </p>
                                  <p className="mt-1 text-2xl font-semibold">
                                    {effectiveStaffPerson.indice?.toFixed(2) ?? "—"}
                                  </p>
                                </div>
                              </div>

                              <div className="mt-6 rounded-2xl border border-dashed bg-muted/20 p-5">
                                <div className="flex flex-wrap items-center gap-2">
                                  <p className="text-sm font-semibold">
                                    Resumen de comentarios
                                  </p>
                                  <span className="rounded-full border bg-background px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                                    En construcción
                                  </span>
                                </div>
                                <p className="mt-2 text-sm leading-6 text-muted-foreground">
                                  Aquí se mostrará la síntesis de los comentarios correspondientes a {effectiveStaffPerson.persona_evaluada} dentro de {effectiveStaffCareer}. La versión final utilizará análisis semántico para detectar patrones, nombres recurrentes y solicitudes específicas.
                                </p>
                              </div>

                              <div className="mt-5 space-y-3">
                                {staffQuestionsByCareer.map((question) => (
                                  <div
                                    key={question.pregunta_id}
                                    className="rounded-xl border bg-card p-4"
                                  >
                                    <div className="flex items-start justify-between gap-4">
                                      <div>
                                        <p className="text-sm font-semibold">
                                          {question.texto_corto}
                                        </p>
                                        <p className="mt-1 text-xs text-muted-foreground">
                                          n = {question.n}
                                        </p>
                                      </div>
                                      <p className="text-lg font-semibold">
                                        {question.indice?.toFixed(2) ?? "—"}
                                      </p>
                                    </div>

                                    <div className="mt-4 flex h-2 overflow-hidden rounded-full bg-muted">
                                      <div
                                        className="bg-emerald-500"
                                        style={{
                                          width: `${question.favorable_pct ?? 0}%`,
                                        }}
                                      />
                                      <div
                                        className="bg-amber-400"
                                        style={{
                                          width: `${question.neutral_pct ?? 0}%`,
                                        }}
                                      />
                                      <div
                                        className="bg-red-500"
                                        style={{
                                          width: `${question.desfavorable_pct ?? 0}%`,
                                        }}
                                      />
                                    </div>

                                    <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[10px]">
                                      <span className="text-emerald-700">
                                        Favorable {question.favorable_pct?.toFixed(1) ?? "0.0"}%
                                      </span>
                                      <span className="text-amber-700">
                                        Neutral {question.neutral_pct?.toFixed(1) ?? "0.0"}%
                                      </span>
                                      <span className="text-red-700">
                                        Desfavorable {question.desfavorable_pct?.toFixed(1) ?? "0.0"}%
                                      </span>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      </aside>
                    </div>
                  )}
                </div>
            ) : (
                <div className="space-y-6">
                  {!selectedSectionEntersGeneral &&
                    effectiveSectionId !== "ESC_00_PERFIL" && (
                    <div className="rounded-2xl border bg-muted/30 px-5 py-4">
                      <p className="text-sm font-medium">
                        Sección informativa
                      </p>
                      <p className="mt-1 text-sm text-muted-foreground">
                        Los resultados de esta sección se analizan de
                        forma independiente y no modifican el Índice
                        General institucional.
                      </p>
                    </div>
                  )}

                  {effectiveSectionId === "ESC_02_DIRECCION" && (
                    <div className="space-y-6">
                      <div className="rounded-2xl border bg-card p-6 shadow-sm">
                        <div className="flex flex-col gap-5 lg:flex-row lg:items-start lg:justify-between">
                          <div>
                            <p className="sia-eyebrow">DIRECCIÓN DE CARRERA</p>
                            <h3 className="mt-2 text-xl font-semibold">
                              Resultados de Dirección
                            </h3>
                            <p className="mt-2 max-w-3xl text-sm text-muted-foreground">
                              Consulta los resultados por carrera y modalidad o consolida
                              las carreras que dependen de una misma persona directora.
                            </p>
                          </div>

                          <div className="inline-flex rounded-xl border bg-muted/20 p-1">
                            {[
                              ["carrera", "Por carrera"],
                              ["director", "Por director(a)"],
                            ].map(([id, label]) => (
                              <button
                                key={id}
                                type="button"
                                onClick={() =>
                                  setDirectionView(id as "carrera" | "director")
                                }
                                className={`rounded-lg px-4 py-2 text-sm font-medium transition ${
                                  directionView === id
                                    ? "bg-background shadow-sm"
                                    : "text-muted-foreground hover:text-foreground"
                                }`}
                              >
                                {label}
                              </button>
                            ))}
                          </div>
                        </div>
                      </div>

                      {directionView === "carrera" && (
                        <div className="rounded-2xl border bg-card p-6 shadow-sm">
                          <div>
                            <p className="text-sm font-semibold">
                              Resultado por carrera y modalidad
                            </p>
                            <p className="mt-1 max-w-3xl text-xs text-muted-foreground">
                              Selecciona una población para abrir al costado sus cuatro
                              criterios, la interpretación y los comentarios asociados.
                            </p>
                          </div>

                          <div className="mt-5 grid max-h-[520px] gap-3 overflow-y-auto pr-2 lg:grid-cols-2">
                            {directionCareerModalities.map((career) => {
                              const conDatos =
                                career.estado === "con_datos" && career.indice !== null

                              const estadoTexto = conDatos
                                ? `${career.encuestas.toLocaleString()} evaluaciones`
                                : career.estado === "sin_encuestas"
                                  ? "Sin encuestas aplicadas"
                                  : career.estado === "sin_datos"
                                    ? "Encuestas sin respuestas válidas de Dirección"
                                    : "Sin evaluación de Dirección en esta ruta"

                              return (
                                <button
                                  key={`${career.carrera_base}|||${career.modalidad_academica}`}
                                  type="button"
                                  onClick={() => {
                                    setSelectedDirectionCareer(career.carrera_base)
                                    setSelectedDirectionModality(
                                      career.modalidad_academica,
                                    )
                                    setDirectionCareerDrawerOpen(true)
                                  }}
                                  className="rounded-xl border bg-background p-4 text-left transition hover:border-primary/40 hover:bg-muted/20"
                                >
                                  <div className="flex items-start justify-between gap-4">
                                    <div className="min-w-0 flex-1">
                                      <p className="font-semibold">
                                        {career.carrera_base}
                                      </p>
                                      <div className="mt-2 flex flex-wrap items-center gap-2">
                                        <span className="rounded-full border px-2.5 py-1 text-[10px] font-medium text-muted-foreground">
                                          {career.modalidad_academica}
                                        </span>
                                        <span className="text-xs text-muted-foreground">
                                          {estadoTexto}
                                        </span>
                                      </div>
                                    </div>

                                    <div className="shrink-0 text-right">
                                      {conDatos ? (
                                        <>
                                          <p className="text-2xl font-semibold">
                                            {career.indice?.toFixed(2)}
                                          </p>
                                          <span
                                            className={`mt-1 inline-flex rounded-full border px-2 py-0.5 text-[10px] font-medium ${getNivelClasses(
                                              career.indice,
                                            )}`}
                                          >
                                            {getNivel(career.indice)}
                                          </span>
                                        </>
                                      ) : (
                                        <span className="inline-flex rounded-full border bg-slate-50 px-2.5 py-1 text-[10px] font-medium text-slate-600">
                                          Sin datos
                                        </span>
                                      )}
                                    </div>
                                  </div>

                                  {conDatos && (
                                    <div className="mt-4 h-2 overflow-hidden rounded-full bg-muted">
                                      <div
                                        className="h-full rounded-full bg-primary"
                                        style={{
                                          width: `${Math.max(
                                            0,
                                            Math.min(100, career.indice ?? 0),
                                          )}%`,
                                        }}
                                      />
                                    </div>
                                  )}
                                </button>
                              )
                            })}
                          </div>
                        </div>
                      )}

                      {directionView === "director" && (
                        <div className="rounded-2xl border bg-card p-6 shadow-sm">
                          <div>
                            <p className="text-sm font-semibold">
                              Resultado consolidado por director(a)
                            </p>
                            <p className="mt-1 max-w-3xl text-xs text-muted-foreground">
                              Agrupa todas las carreras y modalidades asignadas a la misma
                              persona en el catálogo institucional.
                            </p>
                          </div>

                          {directionDirectors.length > 0 ? (
                            <div className="mt-5 grid max-h-[520px] gap-3 overflow-y-auto pr-2 md:grid-cols-2">
                              {directionDirectors.map((director) => {
                                const directorKey = makeDirectionDirectorKey(
                                  director.persona_evaluada,
                                  director.rol_evaluado,
                                )

                                return (
                                  <button
                                    key={directorKey}
                                    type="button"
                                    onClick={() => {
                                      setSelectedDirectionDirectorKey(directorKey)
                                      setDirectionDirectorDrawerOpen(true)
                                    }}
                                    className="rounded-xl border bg-background p-4 text-left transition hover:border-primary/40 hover:bg-muted/20"
                                  >
                                    <div className="flex items-start justify-between gap-4">
                                      <div className="min-w-0 flex-1">
                                        <p className="text-xs text-muted-foreground">
                                          {director.rol_evaluado || "Dirección de carrera"}
                                        </p>
                                        <p className="mt-1 font-semibold">
                                          {director.persona_evaluada}
                                        </p>
                                        <p className="mt-2 text-xs text-muted-foreground">
                                          {director.carreras_a_cargo} {director.carreras_a_cargo === 1 ? "población" : "poblaciones"}
                                          {" · "}
                                          {director.evaluaciones.toLocaleString()} evaluaciones
                                        </p>
                                      </div>

                                      <div className="shrink-0 text-right">
                                        {director.indice !== null ? (
                                          <>
                                            <p className="text-2xl font-semibold">
                                              {director.indice.toFixed(2)}
                                            </p>
                                            <span
                                              className={`mt-1 inline-flex rounded-full border px-2 py-0.5 text-[10px] font-medium ${getNivelClasses(
                                                director.indice,
                                              )}`}
                                            >
                                              {getNivel(director.indice)}
                                            </span>
                                          </>
                                        ) : (
                                          <span className="inline-flex rounded-full border bg-slate-50 px-2.5 py-1 text-[10px] font-medium text-slate-600">
                                            Sin encuestas
                                          </span>
                                        )}
                                      </div>
                                    </div>
                                  </button>
                                )
                              })}
                            </div>
                          ) : (
                            <div className="mt-5 rounded-xl border bg-muted/30 p-5">
                              <p className="text-sm font-semibold">
                                Falta vincular directores con sus carreras
                              </p>
                              <p className="mt-2 text-sm text-muted-foreground">
                                El análisis por carrera ya está disponible. Para habilitar
                                la vista por director(a), el catálogo 04_PERSONAS_EVALUADAS
                                debe incluir a cada director con su carrera, modalidad y rol.
                              </p>
                            </div>
                          )}
                        </div>
                      )}

                      {directionCareerDrawerOpen && selectedDirectionPopulation && (
                        <div className="fixed inset-0 z-50 flex justify-end bg-black/30">
                          <button
                            type="button"
                            aria-label="Cerrar detalle"
                            className="absolute inset-0"
                            onClick={() => setDirectionCareerDrawerOpen(false)}
                          />

                          <aside className="relative z-10 flex h-full w-full max-w-2xl flex-col border-l bg-background shadow-2xl">
                            <div className="border-b p-6">
                              <div className="flex items-start justify-between gap-4">
                                <div>
                                  <p className="sia-eyebrow">DETALLE DE DIRECCIÓN</p>
                                  <h3 className="mt-2 text-2xl font-semibold">
                                    {effectiveDirectionCareer}
                                  </h3>
                                  <p className="mt-1 text-sm text-muted-foreground">
                                    {effectiveDirectionModality}
                                  </p>
                                </div>
                                <button
                                  type="button"
                                  onClick={() => setDirectionCareerDrawerOpen(false)}
                                  className="rounded-xl border px-3 py-2 text-sm"
                                >
                                  Cerrar
                                </button>
                              </div>
                            </div>

                            <div className="flex-1 overflow-y-auto p-6">
                              <div className="grid gap-3 sm:grid-cols-3">
                                <div className="rounded-xl border bg-card p-4">
                                  <p className="text-xs text-muted-foreground">Promedio</p>
                                  <p className="mt-1 text-2xl font-semibold">
                                    {selectedDirectionPopulation.indice?.toFixed(2) ?? "—"}
                                  </p>
                                </div>
                                <div className="rounded-xl border bg-card p-4">
                                  <p className="text-xs text-muted-foreground">Evaluaciones</p>
                                  <p className="mt-1 text-2xl font-semibold">
                                    {selectedDirectionPopulation.encuestas.toLocaleString()}
                                  </p>
                                </div>
                                <div className="rounded-xl border bg-card p-4">
                                  <p className="text-xs text-muted-foreground">Criterios</p>
                                  <p className="mt-1 text-2xl font-semibold">
                                    {selectedDirectionPopulation.preguntas_con_datos}
                                  </p>
                                </div>
                              </div>

                              {selectedDirectionPopulation.estado !== "con_datos" && (
                                <div className="mt-7 rounded-2xl border bg-muted/30 p-5">
                                  <p className="text-sm font-semibold">
                                    Sin encuestas aplicadas
                                  </p>
                                  <p className="mt-2 text-sm leading-6 text-muted-foreground">
                                    La carrera forma parte del catálogo institucional, pero no cuenta con evaluaciones válidas de Dirección para el periodo y filtros seleccionados.
                                  </p>
                                </div>
                              )}

                              <div className="mt-7">
                                <p className="text-sm font-semibold">Criterios evaluados</p>
                                <div className="mt-4 space-y-3">
                                  {directionQuestions.map((question) => {
                                    const previous = findHistoricalDirectionQuestion(question)
                                    const previousValue = previous?.indice ?? null
                                    const delta =
                                      question.indice !== null && previousValue !== null
                                        ? question.indice - previousValue
                                        : null

                                    return (
                                      <div
                                        key={question.pregunta_id}
                                        className="rounded-xl border bg-card p-4"
                                      >
                                        <div className="flex items-start justify-between gap-4">
                                          <div>
                                            <p className="text-sm font-semibold">{question.texto_corto}</p>
                                            <p className="mt-1 text-xs text-muted-foreground">n = {question.n}</p>
                                          </div>
                                          <div className="text-right">
                                            <p className="text-[10px] uppercase tracking-wide text-muted-foreground">2026</p>
                                            <p className="text-xl font-semibold">
                                              {question.indice?.toFixed(2) ?? "—"}
                                            </p>
                                          </div>
                                        </div>

                                        <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
                                          <div className="rounded-lg border bg-background px-3 py-2">
                                            <p className="text-[10px] uppercase tracking-wide text-muted-foreground">2025</p>
                                            <p className="mt-0.5 font-semibold">
                                              {previousValue !== null ? previousValue.toFixed(2) : "Sin comparable"}
                                            </p>
                                          </div>
                                          <div className="rounded-lg border bg-background px-3 py-2">
                                            <p className="text-[10px] uppercase tracking-wide text-muted-foreground">Variación</p>
                                            <p
                                              className={`mt-0.5 font-semibold ${
                                                delta === null
                                                  ? "text-muted-foreground"
                                                  : delta >= 0
                                                    ? "text-emerald-700"
                                                    : "text-red-700"
                                              }`}
                                            >
                                              {delta === null
                                                ? "—"
                                                : `${delta >= 0 ? "+" : ""}${delta.toFixed(2)} pts`}
                                            </p>
                                          </div>
                                        </div>

                                        <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-[10px]">
                                          <span className="text-emerald-700">Favorable {question.favorable_pct?.toFixed(1) ?? "0.0"}%</span>
                                          <span className="text-amber-700">Neutral {question.neutral_pct?.toFixed(1) ?? "0.0"}%</span>
                                          <span className="text-red-700">Desfavorable {question.desfavorable_pct?.toFixed(1) ?? "0.0"}%</span>
                                        </div>
                                      </div>
                                    )
                                  })}
                                </div>
                              </div>

                              <div className="mt-7 rounded-2xl border bg-muted/30 p-5">
                                <p className="text-sm font-semibold">Lectura del resultado</p>
                                <p className="mt-2 text-sm leading-6 text-muted-foreground">
                                  {getDirectionInterpretation(
                                    selectedDirectionPopulation.indice,
                                    directionCommentsForCareer.length,
                                  )}
                                </p>
                              </div>

                              <div className="mt-7 rounded-2xl border bg-muted/25 p-5">
                                <div className="flex flex-wrap items-center gap-2">
                                  <p className="text-sm font-semibold">Resumen preliminar de comentarios</p>
                                  <span className="rounded-full border bg-background px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                                    En construcción
                                  </span>
                                </div>
                                <p className="mt-2 text-sm leading-6 text-muted-foreground">
                                  {directionCareerCommentSummary}
                                </p>
                                <p className="mt-3 text-[11px] leading-5 text-muted-foreground">
                                  Vista preliminar generada con reglas básicas de clasificación. La versión final incorporará análisis semántico con IA para identificar tendencias, menciones recurrentes y contexto específico de los comentarios.
                                </p>
                              </div>

                              <div className="mt-7">
                                <p className="text-sm font-semibold">Comentarios asociados</p>
                                <p className="mt-1 text-xs text-muted-foreground">
                                  Se muestran únicamente comentarios sustantivos de esta carrera.
                                </p>
                                {directionCommentsForCareer.length > 0 ? (
                                  <div className="mt-4 space-y-3">
                                    {directionCommentsForCareer.map((comment, index) => (
                                      <div
                                        key={`${comment.fila_origen}-${comment.pregunta_id}-${index}`}
                                        className="rounded-xl border bg-card p-4"
                                      >
                                        <p className="text-sm leading-6">{comment.texto}</p>
                                      </div>
                                    ))}
                                  </div>
                                ) : (
                                  <div className="mt-4 rounded-xl border bg-card p-4 text-sm text-muted-foreground">
                                    No hay comentarios sustantivos para esta selección.
                                  </div>
                                )}
                              </div>
                            </div>
                          </aside>
                        </div>
                      )}

                      {directionDirectorDrawerOpen && effectiveDirectionDirector && (
                        <div className="fixed inset-0 z-50 flex justify-end bg-black/30">
                          <button
                            type="button"
                            aria-label="Cerrar detalle"
                            className="absolute inset-0"
                            onClick={() => setDirectionDirectorDrawerOpen(false)}
                          />

                          <aside className="relative z-10 flex h-full w-full max-w-2xl flex-col border-l bg-background shadow-2xl">
                            <div className="border-b p-6">
                              <div className="flex items-start justify-between gap-4">
                                <div>
                                  <p className="sia-eyebrow">DETALLE POR DIRECTOR(A)</p>
                                  <h3 className="mt-2 text-2xl font-semibold">
                                    {effectiveDirectionDirector.persona_evaluada}
                                  </h3>
                                  <p className="mt-1 text-sm text-muted-foreground">
                                    {effectiveDirectionDirector.rol_evaluado || "Dirección de carrera"}
                                  </p>
                                </div>
                                <button
                                  type="button"
                                  onClick={() => setDirectionDirectorDrawerOpen(false)}
                                  className="rounded-xl border px-3 py-2 text-sm"
                                >
                                  Cerrar
                                </button>
                              </div>
                            </div>

                            <div className="flex-1 overflow-y-auto p-6">
                              <div className="grid gap-3 sm:grid-cols-3">
                                <div className="rounded-xl border bg-card p-4">
                                  <p className="text-xs text-muted-foreground">Promedio global</p>
                                  <p className="mt-1 text-2xl font-semibold">
                                    {effectiveDirectionDirector.indice?.toFixed(2) ?? "—"}
                                  </p>
                                </div>
                                <div className="rounded-xl border bg-card p-4">
                                  <p className="text-xs text-muted-foreground">Evaluaciones</p>
                                  <p className="mt-1 text-2xl font-semibold">
                                    {effectiveDirectionDirector.evaluaciones.toLocaleString()}
                                  </p>
                                </div>
                                <div className="rounded-xl border bg-card p-4">
                                  <p className="text-xs text-muted-foreground">Poblaciones</p>
                                  <p className="mt-1 text-2xl font-semibold">
                                    {effectiveDirectionDirector.carreras_a_cargo}
                                  </p>
                                </div>
                              </div>

                              <div className="mt-7">
                                <p className="text-sm font-semibold">Carreras a cargo</p>
                                <div className="mt-4 space-y-2">
                                  {effectiveDirectionDirector.poblaciones.map((population) => (
                                    <div
                                      key={`${population.carrera_base}|||${population.modalidad_academica}`}
                                      className="flex items-center justify-between gap-4 rounded-xl border bg-card p-4"
                                    >
                                      <div>
                                        <p className="text-sm font-semibold">{population.carrera_base}</p>
                                        <p className="mt-1 text-xs text-muted-foreground">{population.modalidad_academica} · {population.encuestas} evaluaciones</p>
                                      </div>
                                      <p className="text-xl font-semibold">{population.indice?.toFixed(2) ?? "—"}</p>
                                    </div>
                                  ))}
                                </div>
                              </div>

                              <div className="mt-7 rounded-xl border bg-muted/20 p-4">
                                <p className="text-sm font-semibold">Comparativo histórico nominal</p>
                                <p className="mt-1 text-xs leading-5 text-muted-foreground">
                                  2025 no identifica de forma comparable a cada director(a) por nombre. Para evitar una comparación incorrecta, el histórico se muestra en el nivel de carrera/criterio cuando existe equivalencia válida.
                                </p>
                              </div>

                              <div className="mt-7">
                                <p className="text-sm font-semibold">Criterios consolidados</p>
                                <div className="mt-4 space-y-3">
                                  {directionDirectorQuestions.map((question) => (
                                    <div key={question.pregunta_id} className="rounded-xl border bg-card p-4">
                                      <div className="flex items-start justify-between gap-4">
                                        <div>
                                          <p className="text-sm font-semibold">{question.texto_corto}</p>
                                          <p className="mt-1 text-xs text-muted-foreground">n = {question.n}</p>
                                        </div>
                                        <p className="text-xl font-semibold">{question.indice?.toFixed(2) ?? "—"}</p>
                                      </div>
                                      <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-[10px]">
                                        <span className="text-emerald-700">Favorable {question.favorable_pct?.toFixed(1) ?? "0.0"}%</span>
                                        <span className="text-amber-700">Neutral {question.neutral_pct?.toFixed(1) ?? "0.0"}%</span>
                                        <span className="text-red-700">Desfavorable {question.desfavorable_pct?.toFixed(1) ?? "0.0"}%</span>
                                      </div>
                                    </div>
                                  ))}
                                </div>
                              </div>

                              <div className="mt-7 rounded-2xl border bg-muted/30 p-5">
                                <p className="text-sm font-semibold">Lectura del resultado</p>
                                <p className="mt-2 text-sm leading-6 text-muted-foreground">
                                  {getDirectionInterpretation(
                                    effectiveDirectionDirector.indice,
                                    directionCommentsForDirector.length,
                                  )}
                                </p>
                              </div>

                              <div className="mt-7 rounded-2xl border bg-muted/25 p-5">
                                <div className="flex flex-wrap items-center gap-2">
                                  <p className="text-sm font-semibold">Resumen preliminar de comentarios</p>
                                  <span className="rounded-full border bg-background px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                                    En construcción
                                  </span>
                                </div>
                                <p className="mt-2 text-sm leading-6 text-muted-foreground">
                                  {directionDirectorCommentSummary}
                                </p>
                                <p className="mt-3 text-[11px] leading-5 text-muted-foreground">
                                  Vista preliminar generada con reglas básicas de clasificación. La versión final incorporará análisis semántico con IA para reconocer temas recurrentes, nombres mencionados de forma reiterada y solicitudes concretas del alumnado.
                                </p>
                              </div>

                              <div className="mt-7">
                                <p className="text-sm font-semibold">Comentarios asociados</p>
                                <p className="mt-1 text-xs text-muted-foreground">
                                  Se reúnen los comentarios sustantivos de las carreras bajo su dirección.
                                </p>
                                {directionCommentsForDirector.length > 0 ? (
                                  <div className="mt-4 space-y-3">
                                    {directionCommentsForDirector.map((comment, index) => (
                                      <div key={`${comment.fila_origen}-${comment.pregunta_id}-${index}`} className="rounded-xl border bg-card p-4">
                                        <p className="text-xs font-medium text-muted-foreground">{comment.servicio_programa}</p>
                                        <p className="mt-2 text-sm leading-6">{comment.texto}</p>
                                      </div>
                                    ))}
                                  </div>
                                ) : (
                                  <div className="mt-4 rounded-xl border bg-card p-4 text-sm text-muted-foreground">
                                    No hay comentarios sustantivos asociados a sus carreras.
                                  </div>
                                )}
                              </div>
                            </div>
                          </aside>
                        </div>
                      )}
                    </div>
                  )}

            {effectiveSectionId === "ESC_03_SERVICIOS" &&
              institutionalServiceQuestions.length > 0 && (
                <div className="space-y-6">
                  <div className="rounded-2xl border bg-card p-6 shadow-sm">
                    <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
                      <div>
                        <p className="sia-eyebrow">LECTURA POR SERVICIO</p>
                        <h3 className="mt-2 text-2xl font-semibold">
                          Resultado de servicios institucionales
                        </h3>
                        <p className="mt-2 max-w-3xl text-sm text-muted-foreground">
                          Los servicios se ordenan de menor a mayor resultado. Selecciona uno para revisar su evaluación, el comparable 2025 y los comentarios relacionados con ese tema.
                        </p>
                      </div>
                      <div className="rounded-xl border bg-background px-4 py-3 text-right">
                        <p className="text-xs text-muted-foreground">Índice de la sección</p>
                        <p className="mt-1 text-3xl font-semibold">
                          {selectedSection?.indice?.toFixed(2) ?? "—"}
                        </p>
                      </div>
                    </div>

                    <div className="mt-5 rounded-xl border bg-muted/10 p-4">
                      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
                        <div>
                          <p className="text-sm font-semibold">Vista compacta por servicio</p>
                          <p className="mt-1 text-xs text-muted-foreground">
                            Ordenados de menor a mayor resultado. Haz clic en cualquier servicio para abrir el detalle.
                          </p>
                        </div>

                        <div className="flex flex-wrap items-center gap-3 text-[10px] text-muted-foreground">
                          <span>Crítico ≤ 60</span>
                          <span>Atención 60–75</span>
                          <span>Adecuado 75–85</span>
                          <span>Fortaleza &gt; 85</span>
                        </div>
                      </div>

                      <div className="grid gap-2 lg:grid-cols-2">
                        {institutionalServiceQuestions.map((question) => {
                          const previous = question.historical_index
                          const delta =
                            question.indice !== null &&
                            previous !== null &&
                            previous !== undefined
                              ? question.indice - previous
                              : null

                          const related = commentsForInstitutionalServiceQuestion(
                            question,
                            summary?.comentarios ?? [],
                          )

                          const score = Math.max(
                            0,
                            Math.min(100, question.indice ?? 0),
                          )

                          return (
                            <button
                              key={question.pregunta_id}
                              type="button"
                              onClick={() => {
                                setSelectedInstitutionalServiceQuestionId(
                                  question.pregunta_id,
                                )
                                setInstitutionalServiceDrawerOpen(true)
                              }}
                              className="rounded-xl border bg-background p-3 text-left transition hover:border-primary/40 hover:bg-muted/20"
                            >
                              <div className="flex items-start justify-between gap-3">
                                <div className="min-w-0 flex-1">
                                  <p className="truncate text-sm font-semibold">
                                    {institutionalServiceDisplayName(question)}
                                  </p>
                                  <p className="mt-1 text-[11px] text-muted-foreground">
                                    {question.n.toLocaleString()} respuestas
                                    {" · "}
                                    {related.length} comentario
                                    {related.length === 1 ? "" : "s"}
                                  </p>
                                </div>

                                <div className="shrink-0 text-right">
                                  <p className="text-lg font-semibold">
                                    {question.indice?.toFixed(1) ?? "—"}
                                  </p>
                                  <p className="text-[10px] text-muted-foreground">
                                    2026
                                  </p>
                                </div>
                              </div>

                              <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-muted">
                                <div
                                  className="h-full rounded-full"
                                  style={{
                                    width: `${score}%`,
                                    backgroundColor: getBarColor(question.indice),
                                  }}
                                />
                              </div>

                              <div className="mt-2 flex items-center justify-between gap-3 text-[10px]">
                                <span className="text-muted-foreground">
                                  2025: {previous?.toFixed(1) ?? "Sin comparable"}
                                </span>
                                <span
                                  className={
                                    delta === null
                                      ? "text-muted-foreground"
                                      : delta >= 0
                                        ? "text-emerald-700"
                                        : "text-red-700"
                                  }
                                >
                                  {delta === null
                                    ? "—"
                                    : `${delta >= 0 ? "+" : ""}${delta.toFixed(1)} pts`}
                                </span>
                              </div>
                            </button>
                          )
                        })}
                      </div>
                    </div>
                  </div>

                  {institutionalServiceDrawerOpen && selectedInstitutionalServiceQuestion && (
                    <div className="fixed inset-0 z-[130] flex justify-end bg-black/30">
                      <button
                        type="button"
                        aria-label="Cerrar detalle de servicio"
                        onClick={() => setInstitutionalServiceDrawerOpen(false)}
                        className="absolute inset-0"
                      />
                      <aside className="relative z-10 flex h-full w-full max-w-2xl flex-col border-l bg-background shadow-2xl">
                        <div className="border-b p-6">
                          <div className="flex items-start justify-between gap-4">
                            <div>
                              <p className="sia-eyebrow">DETALLE DEL SERVICIO</p>
                              <h3 className="mt-2 text-2xl font-semibold">
                                {institutionalServiceDisplayName(selectedInstitutionalServiceQuestion)}
                              </h3>
                              <p className="mt-1 text-sm text-muted-foreground">
                                Servicios institucionales · {filters.periodo || "Periodo actual"}
                              </p>
                            </div>
                            <button
                              type="button"
                              onClick={() => setInstitutionalServiceDrawerOpen(false)}
                              className="rounded-xl border px-3 py-2 text-sm"
                            >
                              Cerrar
                            </button>
                          </div>
                        </div>

                        <div className="flex-1 overflow-y-auto p-6">
                          <div className="grid gap-3 sm:grid-cols-3">
                            <div className="rounded-xl border bg-card p-4">
                              <p className="text-xs text-muted-foreground">Resultado 2026</p>
                              <p className="mt-1 text-2xl font-semibold">
                                {selectedInstitutionalServiceQuestion.indice?.toFixed(2) ?? "—"}
                              </p>
                            </div>
                            <div className="rounded-xl border bg-card p-4">
                              <p className="text-xs text-muted-foreground">Comparable 2025</p>
                              <p className="mt-1 text-2xl font-semibold">
                                {selectedInstitutionalServiceQuestion.historical_index?.toFixed(2) ?? "—"}
                              </p>
                            </div>
                            <div className="rounded-xl border bg-card p-4">
                              <p className="text-xs text-muted-foreground">Variación</p>
                              <p className={`mt-1 text-2xl font-semibold ${
                                institutionalServiceDelta === null
                                  ? ""
                                  : institutionalServiceDelta >= 0
                                    ? "text-emerald-700"
                                    : "text-red-700"
                              }`}>
                                {institutionalServiceDelta === null
                                  ? "—"
                                  : `${institutionalServiceDelta >= 0 ? "+" : ""}${institutionalServiceDelta.toFixed(2)}`}
                              </p>
                            </div>
                          </div>

                          <div className="mt-7">
                            <p className="text-sm font-semibold">Distribución de respuestas</p>
                            <p className="mt-1 text-xs text-muted-foreground">
                              Lectura agrupada de respuestas favorables, neutrales y desfavorables.
                            </p>
                            <div className="mt-4 flex h-4 overflow-hidden rounded-full bg-muted">
                              <div
                                className="bg-emerald-500"
                                style={{ width: `${selectedInstitutionalServiceQuestion.favorable_pct ?? 0}%` }}
                              />
                              <div
                                className="bg-amber-400"
                                style={{ width: `${selectedInstitutionalServiceQuestion.neutral_pct ?? 0}%` }}
                              />
                              <div
                                className="bg-red-500"
                                style={{ width: `${selectedInstitutionalServiceQuestion.desfavorable_pct ?? 0}%` }}
                              />
                            </div>
                            <div className="mt-3 grid gap-2 sm:grid-cols-3">
                              <div className="rounded-lg border p-3 text-sm">
                                <span className="text-emerald-700">Favorable</span>
                                <span className="float-right font-semibold">{selectedInstitutionalServiceQuestion.favorable_pct?.toFixed(1) ?? "0.0"}%</span>
                              </div>
                              <div className="rounded-lg border p-3 text-sm">
                                <span className="text-amber-700">Neutral</span>
                                <span className="float-right font-semibold">{selectedInstitutionalServiceQuestion.neutral_pct?.toFixed(1) ?? "0.0"}%</span>
                              </div>
                              <div className="rounded-lg border p-3 text-sm">
                                <span className="text-red-700">Desfavorable</span>
                                <span className="float-right font-semibold">{selectedInstitutionalServiceQuestion.desfavorable_pct?.toFixed(1) ?? "0.0"}%</span>
                              </div>
                            </div>
                          </div>

                          <div className="mt-7">
                            <div className="flex items-start justify-between gap-4">
                              <div>
                                <p className="text-sm font-semibold">Comentarios relacionados</p>
                                <p className="mt-1 text-xs text-muted-foreground">
                                  Clasificación temática automática por palabras clave. Un comentario puede aparecer en más de un servicio cuando aborda varios temas.
                                </p>
                              </div>
                              <span className="rounded-full border bg-muted/30 px-3 py-1 text-xs font-medium">
                                {selectedInstitutionalServiceComments.length}
                              </span>
                            </div>

                            {selectedInstitutionalServiceComments.length > 0 ? (
                              <div className="mt-4 space-y-3">
                                {selectedInstitutionalServiceComments.map((comment, index) => (
                                  <div
                                    key={`${comment.fila_origen}-${comment.pregunta_id}-${index}`}
                                    className="rounded-xl border bg-card p-4"
                                  >
                                    <p className="text-sm leading-6">{comment.texto}</p>
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <div className="mt-4 rounded-xl border bg-muted/20 p-4 text-sm text-muted-foreground">
                                No se identificaron comentarios relacionados directamente con este servicio.
                              </div>
                            )}
                          </div>
                        </div>
                      </aside>
                    </div>
                  )}
                </div>
              )}


            {effectiveSectionId === "ESC_04_ACADEMICOS" &&
              academicServiceQuestions.length > 0 && (
                <div className="space-y-6">
                  <div className="rounded-2xl border bg-card p-6 shadow-sm">
                    <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
                      <div>
                        <p className="sia-eyebrow">LECTURA POR SERVICIO ACADÉMICO</p>
                        <h3 className="mt-2 text-2xl font-semibold">
                          Resultado de servicios académicos
                        </h3>
                        <p className="mt-2 max-w-3xl text-sm text-muted-foreground">
                          Vista compacta de los servicios académicos, ordenados de menor a mayor resultado. Selecciona uno para revisar distribución, comparativo histórico y comentarios relacionados.
                        </p>
                      </div>
                    </div>

                    <div className="mt-6 grid gap-3 lg:grid-cols-2">
                      {academicServiceQuestions.map((question) => {
                        const related = commentsForAcademicServiceQuestion(
                          question,
                          summary?.comentarios ?? [],
                        )
                        const previous = question.historical_index ?? null
                        const delta =
                          question.indice !== null && previous !== null
                            ? question.indice - previous
                            : null

                        return (
                          <button
                            key={question.pregunta_id}
                            type="button"
                            onClick={() => {
                              setSelectedAcademicServiceQuestionId(question.pregunta_id)
                              setAcademicServiceDrawerOpen(true)
                            }}
                            className="rounded-xl border bg-background p-4 text-left transition hover:border-primary/40 hover:bg-muted/20"
                          >
                            <div className="flex items-start justify-between gap-4">
                              <div className="min-w-0 flex-1">
                                <p className="font-semibold">
                                  {academicServiceDisplayName(question)}
                                </p>
                                <p className="mt-1 text-xs text-muted-foreground">
                                  {question.n.toLocaleString()} respuestas · {related.length} comentarios
                                </p>
                              </div>
                              <div className="shrink-0 text-right">
                                <p className="text-xl font-semibold">
                                  {question.indice?.toFixed(1) ?? "—"}
                                </p>
                                <p className="text-[10px] text-muted-foreground">2026</p>
                              </div>
                            </div>

                            <div className="mt-3 h-2 overflow-hidden rounded-full bg-muted">
                              <div
                                className={`h-full rounded-full ${getBarColor(question.indice)}`}
                                style={{ width: `${Math.max(0, Math.min(100, question.indice ?? 0))}%` }}
                              />
                            </div>

                            <div className="mt-3 flex items-center justify-between gap-3 text-xs">
                              <span className="text-muted-foreground">
                                2025: {previous !== null ? previous.toFixed(1) : "Sin comparable"}
                              </span>
                              <span
                                className={
                                  delta === null
                                    ? "text-muted-foreground"
                                    : delta >= 0
                                      ? "text-emerald-700"
                                      : "text-red-700"
                                }
                              >
                                {delta === null
                                  ? "—"
                                  : `${delta >= 0 ? "+" : ""}${delta.toFixed(1)} pts`}
                              </span>
                            </div>
                          </button>
                        )
                      })}
                    </div>
                  </div>

                  {academicServiceDrawerOpen && selectedAcademicServiceQuestion && (
                    <div className="fixed inset-0 z-[130] flex justify-end bg-black/30">
                      <button
                        type="button"
                        aria-label="Cerrar detalle de servicio académico"
                        onClick={() => setAcademicServiceDrawerOpen(false)}
                        className="absolute inset-0"
                      />
                      <aside className="relative z-10 flex h-full w-full max-w-2xl flex-col border-l bg-background shadow-2xl">
                        <div className="border-b p-6">
                          <div className="flex items-start justify-between gap-4">
                            <div>
                              <p className="sia-eyebrow">DETALLE DEL SERVICIO ACADÉMICO</p>
                              <h3 className="mt-2 text-2xl font-semibold">
                                {academicServiceDisplayName(selectedAcademicServiceQuestion)}
                              </h3>
                              <p className="mt-1 text-sm text-muted-foreground">
                                Servicios académicos · {filters.periodo || "Periodo actual"}
                              </p>
                            </div>
                            <button
                              type="button"
                              onClick={() => setAcademicServiceDrawerOpen(false)}
                              className="rounded-xl border px-3 py-2 text-sm"
                            >
                              Cerrar
                            </button>
                          </div>
                        </div>

                        <div className="flex-1 overflow-y-auto p-6">
                          <div className="grid gap-3 sm:grid-cols-3">
                            <div className="rounded-xl border bg-card p-4">
                              <p className="text-xs text-muted-foreground">Resultado 2026</p>
                              <p className="mt-1 text-2xl font-semibold">
                                {selectedAcademicServiceQuestion.indice?.toFixed(2) ?? "—"}
                              </p>
                            </div>
                            <div className="rounded-xl border bg-card p-4">
                              <p className="text-xs text-muted-foreground">Comparable 2025</p>
                              <p className="mt-1 text-2xl font-semibold">
                                {selectedAcademicServiceQuestion.historical_index?.toFixed(2) ?? "—"}
                              </p>
                            </div>
                            <div className="rounded-xl border bg-card p-4">
                              <p className="text-xs text-muted-foreground">Variación</p>
                              <p className={`mt-1 text-2xl font-semibold ${
                                academicServiceDelta === null
                                  ? ""
                                  : academicServiceDelta >= 0
                                    ? "text-emerald-700"
                                    : "text-red-700"
                              }`}>
                                {academicServiceDelta === null
                                  ? "—"
                                  : `${academicServiceDelta >= 0 ? "+" : ""}${academicServiceDelta.toFixed(2)}`}
                              </p>
                            </div>
                          </div>

                          <div className="mt-7">
                            <p className="text-sm font-semibold">Distribución de respuestas</p>
                            <p className="mt-1 text-xs text-muted-foreground">
                              Lectura agrupada de respuestas favorables, neutrales y desfavorables.
                            </p>
                            <div className="mt-4 flex h-4 overflow-hidden rounded-full bg-muted">
                              <div
                                className="bg-emerald-500"
                                style={{ width: `${selectedAcademicServiceQuestion.favorable_pct ?? 0}%` }}
                              />
                              <div
                                className="bg-amber-400"
                                style={{ width: `${selectedAcademicServiceQuestion.neutral_pct ?? 0}%` }}
                              />
                              <div
                                className="bg-red-500"
                                style={{ width: `${selectedAcademicServiceQuestion.desfavorable_pct ?? 0}%` }}
                              />
                            </div>
                            <div className="mt-3 grid gap-2 sm:grid-cols-3">
                              <div className="rounded-lg border p-3 text-sm">
                                <span className="text-emerald-700">Favorable</span>
                                <span className="float-right font-semibold">{selectedAcademicServiceQuestion.favorable_pct?.toFixed(1) ?? "0.0"}%</span>
                              </div>
                              <div className="rounded-lg border p-3 text-sm">
                                <span className="text-amber-700">Neutral</span>
                                <span className="float-right font-semibold">{selectedAcademicServiceQuestion.neutral_pct?.toFixed(1) ?? "0.0"}%</span>
                              </div>
                              <div className="rounded-lg border p-3 text-sm">
                                <span className="text-red-700">Desfavorable</span>
                                <span className="float-right font-semibold">{selectedAcademicServiceQuestion.desfavorable_pct?.toFixed(1) ?? "0.0"}%</span>
                              </div>
                            </div>
                          </div>

                          <div className="mt-7">
                            <div className="flex items-start justify-between gap-4">
                              <div>
                                <p className="text-sm font-semibold">Comentarios relacionados</p>
                                <p className="mt-1 text-xs text-muted-foreground">
                                  Clasificación temática automática por palabras clave. Un comentario puede relacionarse con más de un servicio académico.
                                </p>
                              </div>
                              <span className="rounded-full border bg-muted/30 px-3 py-1 text-xs font-medium">
                                {selectedAcademicServiceComments.length}
                              </span>
                            </div>

                            {selectedAcademicServiceComments.length > 0 ? (
                              <div className="mt-4 space-y-3">
                                {selectedAcademicServiceComments.map((comment, index) => (
                                  <div
                                    key={`${comment.fila_origen}-${comment.pregunta_id}-${index}`}
                                    className="rounded-xl border bg-card p-4"
                                  >
                                    <p className="text-sm leading-6">{comment.texto}</p>
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <div className="mt-4 rounded-xl border bg-muted/20 p-4 text-sm text-muted-foreground">
                                No se identificaron comentarios relacionados directamente con este servicio académico.
                              </div>
                            )}
                          </div>
                        </div>
                      </aside>
                    </div>
                  )}
                </div>
              )}


            {effectiveSectionId === "ESC_05_IDIOMAS" &&
              languageCenterQuestions.length > 0 && (
                <div className="space-y-6">
                  <div className="rounded-2xl border bg-card p-6 shadow-sm">
                    <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
                      <div>
                        <p className="sia-eyebrow">CENTRO DE IDIOMAS</p>
                        <h3 className="mt-2 text-2xl font-semibold">
                          Experiencia con el Centro de Idiomas
                        </h3>
                        <p className="mt-2 max-w-3xl text-sm text-muted-foreground">
                          Lectura compacta de los criterios evaluados. Selecciona un
                          criterio para revisar su distribución, comparación con 2025
                          y comentarios relacionados.
                        </p>
                      </div>
                    </div>

                    <div className="mt-6 grid gap-3 md:grid-cols-3">
                      {languageCenterQuestions.map((question) => {
                        const related = commentsForLanguageCenterQuestion(
                          question,
                          summary?.comentarios ?? [],
                        )
                        const previous = question.historical_index ?? null
                        const delta =
                          question.indice !== null && previous !== null
                            ? question.indice - previous
                            : null

                        return (
                          <button
                            key={question.pregunta_id}
                            type="button"
                            onClick={() => {
                              setSelectedLanguageCenterQuestionId(
                                question.pregunta_id,
                              )
                              setLanguageCenterDrawerOpen(true)
                            }}
                            className="rounded-xl border bg-background p-4 text-left transition hover:border-primary/40 hover:bg-muted/20"
                          >
                            <div className="flex items-start justify-between gap-3">
                              <div className="min-w-0 flex-1">
                                <p className="font-semibold">
                                  {languageCenterDisplayName(question)}
                                </p>
                                <p className="mt-1 text-xs text-muted-foreground">
                                  {question.n.toLocaleString()} respuestas ·{" "}
                                  {related.length} comentarios relacionados
                                </p>
                              </div>
                              <div className="shrink-0 text-right">
                                <p className="text-xl font-semibold">
                                  {question.indice?.toFixed(1) ?? "—"}
                                </p>
                                <p className="text-[10px] text-muted-foreground">
                                  2026
                                </p>
                              </div>
                            </div>

                            <div className="mt-3 h-2 overflow-hidden rounded-full bg-muted">
                              <div
                                className={`h-full rounded-full ${getBarColor(
                                  question.indice,
                                )}`}
                                style={{
                                  width: `${Math.max(
                                    0,
                                    Math.min(100, question.indice ?? 0),
                                  )}%`,
                                }}
                              />
                            </div>

                            <div className="mt-3 flex items-center justify-between gap-3 text-xs">
                              <span className="text-muted-foreground">
                                2025:{" "}
                                {previous !== null
                                  ? previous.toFixed(1)
                                  : "Sin comparable"}
                              </span>
                              <span
                                className={
                                  delta === null
                                    ? "text-muted-foreground"
                                    : delta >= 0
                                      ? "text-emerald-700"
                                      : "text-red-700"
                                }
                              >
                                {delta === null
                                  ? "—"
                                  : `${delta >= 0 ? "+" : ""}${delta.toFixed(
                                      1,
                                    )} pts`}
                              </span>
                            </div>
                          </button>
                        )
                      })}
                    </div>
                  </div>

                  <div className="rounded-2xl border bg-card p-5 shadow-sm">
                    <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                      <div>
                        <p className="sia-eyebrow">COMENTARIOS DEL SERVICIO</p>
                        <h4 className="mt-2 text-lg font-semibold">
                          Lectura rápida de la experiencia
                        </h4>
                        <p className="mt-1 max-w-3xl text-xs text-muted-foreground">
                          Clasificación práctica de las respuestas abiertas para evitar
                          duplicar el listado completo y facilitar la identificación de
                          fortalezas y oportunidades de mejora.
                        </p>
                      </div>

                      <div className="flex flex-wrap gap-2">
                        <span className="rounded-full border bg-emerald-50 px-3 py-1.5 text-xs font-medium text-emerald-700">
                          {languageCenterStrengths.length} favorables
                        </span>
                        <span className="rounded-full border bg-amber-50 px-3 py-1.5 text-xs font-medium text-amber-700">
                          {languageCenterOpportunities.length} mejoras
                        </span>
                        {languageCenterGeneralComments.length > 0 && (
                          <span className="rounded-full border bg-muted/30 px-3 py-1.5 text-xs font-medium text-muted-foreground">
                            {languageCenterGeneralComments.length} generales
                          </span>
                        )}
                      </div>
                    </div>

                    <div className="mt-5 grid gap-5 xl:grid-cols-[0.85fr_1.15fr]">
                      <div className="rounded-xl border bg-muted/15 p-4">
                        <div className="flex items-center justify-between gap-3">
                          <div>
                            <p className="text-sm font-semibold">
                              Lo que más valoran
                            </p>
                            <p className="mt-1 text-[11px] text-muted-foreground">
                              Comentarios favorables o respuestas que indican ausencia
                              de problemas.
                            </p>
                          </div>
                          <span className="text-2xl font-semibold text-emerald-700">
                            {languageCenterStrengths.length}
                          </span>
                        </div>

                        {languageCenterStrengths.length > 0 ? (
                          <div className="mt-3 grid gap-2">
                            {languageCenterStrengths.slice(0, 4).map(
                              (comment, index) => (
                                <div
                                  key={`${comment.fila_origen}-${comment.pregunta_id}-${index}`}
                                  className="rounded-lg border bg-background px-3 py-2.5"
                                >
                                  <p className="text-xs leading-5">
                                    {comment.texto}
                                  </p>
                                </div>
                              ),
                            )}
                          </div>
                        ) : (
                          <div className="mt-3 rounded-lg border border-dashed bg-background px-3 py-3">
                            <p className="text-xs leading-5 text-muted-foreground">
                              No se registraron comentarios favorables explícitos en la
                              selección actual.
                            </p>
                          </div>
                        )}
                      </div>

                      <div className="rounded-xl border bg-muted/15 p-4">
                        <div className="flex items-center justify-between gap-3">
                          <div>
                            <p className="text-sm font-semibold">
                              Oportunidades de mejora
                            </p>
                            <p className="mt-1 text-[11px] text-muted-foreground">
                              Observaciones que señalan problemas, solicitudes o cambios
                              concretos.
                            </p>
                          </div>
                          <span className="text-2xl font-semibold text-amber-700">
                            {languageCenterOpportunities.length}
                          </span>
                        </div>

                        {languageCenterOpportunities.length > 0 ? (
                          <div className="mt-3 grid gap-2 sm:grid-cols-2">
                            {languageCenterOpportunities.slice(0, 6).map(
                              (comment, index) => (
                                <div
                                  key={`${comment.fila_origen}-${comment.pregunta_id}-${index}`}
                                  className="rounded-lg border bg-background px-3 py-2.5"
                                >
                                  <p className="text-xs leading-5">
                                    {comment.texto}
                                  </p>
                                </div>
                              ),
                            )}
                          </div>
                        ) : (
                          <div className="mt-3 rounded-lg border border-dashed bg-background px-3 py-3">
                            <p className="text-xs leading-5 text-muted-foreground">
                              No se identificaron oportunidades de mejora específicas.
                            </p>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  {languageCenterDrawerOpen &&
                    selectedLanguageCenterQuestion && (
                      <div className="fixed inset-0 z-[130] flex justify-end bg-black/30">
                        <button
                          type="button"
                          aria-label="Cerrar detalle del Centro de Idiomas"
                          onClick={() => setLanguageCenterDrawerOpen(false)}
                          className="absolute inset-0"
                        />

                        <aside className="relative z-10 flex h-full w-full max-w-2xl flex-col border-l bg-background shadow-2xl">
                          <div className="border-b p-6">
                            <div className="flex items-start justify-between gap-4">
                              <div>
                                <p className="sia-eyebrow">
                                  DETALLE · CENTRO DE IDIOMAS
                                </p>
                                <h3 className="mt-2 text-2xl font-semibold">
                                  {languageCenterDisplayName(
                                    selectedLanguageCenterQuestion,
                                  )}
                                </h3>
                                <p className="mt-1 text-sm text-muted-foreground">
                                  {filters.periodo || "Periodo actual"}
                                </p>
                              </div>
                              <button
                                type="button"
                                onClick={() =>
                                  setLanguageCenterDrawerOpen(false)
                                }
                                className="rounded-xl border px-3 py-2 text-sm"
                              >
                                Cerrar
                              </button>
                            </div>
                          </div>

                          <div className="flex-1 overflow-y-auto p-6">
                            <div className="grid gap-3 sm:grid-cols-3">
                              <div className="rounded-xl border bg-card p-4">
                                <p className="text-xs text-muted-foreground">
                                  Resultado 2026
                                </p>
                                <p className="mt-1 text-2xl font-semibold">
                                  {selectedLanguageCenterQuestion.indice?.toFixed(
                                    2,
                                  ) ?? "—"}
                                </p>
                              </div>
                              <div className="rounded-xl border bg-card p-4">
                                <p className="text-xs text-muted-foreground">
                                  Comparable 2025
                                </p>
                                <p className="mt-1 text-2xl font-semibold">
                                  {selectedLanguageCenterQuestion.historical_index?.toFixed(
                                    2,
                                  ) ?? "—"}
                                </p>
                              </div>
                              <div className="rounded-xl border bg-card p-4">
                                <p className="text-xs text-muted-foreground">
                                  Variación
                                </p>
                                <p
                                  className={`mt-1 text-2xl font-semibold ${
                                    languageCenterDelta === null
                                      ? ""
                                      : languageCenterDelta >= 0
                                        ? "text-emerald-700"
                                        : "text-red-700"
                                  }`}
                                >
                                  {languageCenterDelta === null
                                    ? "—"
                                    : `${
                                        languageCenterDelta >= 0 ? "+" : ""
                                      }${languageCenterDelta.toFixed(2)}`}
                                </p>
                              </div>
                            </div>

                            <div className="mt-7">
                              <p className="text-sm font-semibold">
                                Distribución de respuestas
                              </p>
                              <p className="mt-1 text-xs text-muted-foreground">
                                Lectura agrupada de respuestas favorables,
                                neutrales y desfavorables.
                              </p>

                              <div className="mt-4 flex h-4 overflow-hidden rounded-full bg-muted">
                                <div
                                  className="bg-emerald-500"
                                  style={{
                                    width: `${
                                      selectedLanguageCenterQuestion.favorable_pct ??
                                      0
                                    }%`,
                                  }}
                                />
                                <div
                                  className="bg-amber-400"
                                  style={{
                                    width: `${
                                      selectedLanguageCenterQuestion.neutral_pct ??
                                      0
                                    }%`,
                                  }}
                                />
                                <div
                                  className="bg-red-500"
                                  style={{
                                    width: `${
                                      selectedLanguageCenterQuestion.desfavorable_pct ??
                                      0
                                    }%`,
                                  }}
                                />
                              </div>

                              <div className="mt-3 grid gap-2 sm:grid-cols-3">
                                <div className="rounded-lg border p-3 text-sm">
                                  <span className="text-emerald-700">
                                    Favorable
                                  </span>
                                  <span className="float-right font-semibold">
                                    {selectedLanguageCenterQuestion.favorable_pct?.toFixed(
                                      1,
                                    ) ?? "0.0"}
                                    %
                                  </span>
                                </div>
                                <div className="rounded-lg border p-3 text-sm">
                                  <span className="text-amber-700">
                                    Neutral
                                  </span>
                                  <span className="float-right font-semibold">
                                    {selectedLanguageCenterQuestion.neutral_pct?.toFixed(
                                      1,
                                    ) ?? "0.0"}
                                    %
                                  </span>
                                </div>
                                <div className="rounded-lg border p-3 text-sm">
                                  <span className="text-red-700">
                                    Desfavorable
                                  </span>
                                  <span className="float-right font-semibold">
                                    {selectedLanguageCenterQuestion.desfavorable_pct?.toFixed(
                                      1,
                                    ) ?? "0.0"}
                                    %
                                  </span>
                                </div>
                              </div>
                            </div>

                            <div className="mt-7">
                              <div className="flex items-start justify-between gap-4">
                                <div>
                                  <p className="text-sm font-semibold">
                                    Comentarios relacionados
                                  </p>
                                  <p className="mt-1 text-xs text-muted-foreground">
                                    Clasificación temática automática para
                                    contextualizar este criterio.
                                  </p>
                                </div>
                                <span className="rounded-full border bg-muted/30 px-3 py-1 text-xs font-medium">
                                  {selectedLanguageCenterComments.length}
                                </span>
                              </div>

                              {selectedLanguageCenterComments.length > 0 ? (
                                <div className="mt-4 space-y-3">
                                  {selectedLanguageCenterComments.map(
                                    (comment, index) => (
                                      <div
                                        key={`${comment.fila_origen}-${comment.pregunta_id}-${index}`}
                                        className="rounded-xl border bg-card p-4"
                                      >
                                        <p className="text-sm leading-6">
                                          {comment.texto}
                                        </p>
                                      </div>
                                    ),
                                  )}
                                </div>
                              ) : (
                                <div className="mt-4 rounded-xl border bg-muted/20 p-4 text-sm text-muted-foreground">
                                  No se identificaron comentarios relacionados
                                  directamente con este criterio.
                                </div>
                              )}
                            </div>
                          </div>
                        </aside>
                      </div>
                    )}
                </div>
              )}

            {isContinuingEducationSection && continuingEducationQuestions.length > 0 && (
              <div className="space-y-6">
                <div className="rounded-2xl border bg-card p-6 shadow-sm">
                  <div>
                    <p className="sia-eyebrow">EDUCACIÓN CONTINUA</p>
                    <h3 className="mt-2 text-2xl font-semibold">Experiencia con Educación Continua</h3>
                    <p className="mt-2 max-w-3xl text-sm text-muted-foreground">
                      Vista compacta de calidad, pertinencia, horarios, recomendación y conocimiento de la oferta. Selecciona un indicador para consultar su detalle.
                    </p>
                  </div>
                  <div className="mt-6 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                    {continuingEducationQuestions.map((question) => {
                      const previous = question.historical_index ?? null
                      const delta = question.indice !== null && previous !== null ? question.indice - previous : null
                      const binary = normalizeAcademicLabel(question.tipo_dato ?? "").includes("bin") ||
                        normalizeAcademicLabel(question.texto_corto ?? "").includes("recomendar") ||
                        normalizeAcademicLabel(question.texto_corto ?? "").includes("conozco") ||
                        normalizeAcademicLabel(question.texto_corto ?? "").includes("oferta academica")
                      return (
                        <button
                          key={question.pregunta_id}
                          type="button"
                          onClick={() => {
                            setSelectedContinuingEducationQuestionId(question.pregunta_id)
                            setContinuingEducationDrawerOpen(true)
                          }}
                          className="rounded-xl border bg-background p-4 text-left transition hover:border-primary/40 hover:bg-muted/20"
                        >
                          <div className="flex items-start justify-between gap-3">
                            <div className="min-w-0 flex-1">
                              <p className="font-semibold">{continuingEducationDisplayName(question)}</p>
                              <p className="mt-1 text-xs text-muted-foreground">
                                {question.n.toLocaleString()} respuestas · {binary ? "Indicador de cobertura/percepción" : "Indicador de satisfacción"}
                              </p>
                            </div>
                            <div className="shrink-0 text-right">
                              <p className="text-xl font-semibold">{question.indice?.toFixed(1) ?? "—"}</p>
                              <p className="text-[10px] text-muted-foreground">2026</p>
                            </div>
                          </div>
                          <div className="mt-3 h-2 overflow-hidden rounded-full bg-muted">
                            <div className={`h-full rounded-full ${getBarColor(question.indice)}`} style={{ width: `${Math.max(0, Math.min(100, question.indice ?? 0))}%` }} />
                          </div>
                          <div className="mt-3 flex items-center justify-between gap-3 text-xs">
                            <span className="text-muted-foreground">2025: {previous !== null ? previous.toFixed(1) : "Sin comparable"}</span>
                            <span className={delta === null ? "text-muted-foreground" : delta >= 0 ? "text-emerald-700" : "text-red-700"}>
                              {delta === null ? "—" : `${delta >= 0 ? "+" : ""}${delta.toFixed(1)} pts`}
                            </span>
                          </div>
                        </button>
                      )
                    })}
                  </div>
                </div>

                <div className="rounded-2xl border bg-card p-5 shadow-sm">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <p className="text-sm font-semibold">Áreas de mejora señaladas por estudiantes</p>
                      <p className="mt-1 text-xs text-muted-foreground">
                        Se muestran únicamente sugerencias sustantivas; respuestas como “todo bien”, “ninguna” o “nada que mejorar” se excluyen de este bloque.
                      </p>
                    </div>
                    <span className="rounded-full border bg-muted/30 px-2.5 py-1 text-[10px] font-medium">{continuingEducationImprovementList.length}</span>
                  </div>
                  {continuingEducationImprovementList.length > 0 ? (
                    <div className="mt-4 grid gap-2 md:grid-cols-2">
                      {continuingEducationImprovementList.slice(0, 8).map((comment, index) => (
                        <div key={`${comment.fila_origen}-${comment.pregunta_id}-${index}`} className="rounded-xl border bg-background p-3">
                          <p className="text-sm leading-5">{comment.texto}</p>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="mt-4 rounded-xl border bg-muted/20 p-4 text-sm text-muted-foreground">No se identificaron sugerencias sustantivas de mejora en la selección actual.</div>
                  )}
                </div>

                {continuingEducationDrawerOpen && selectedContinuingEducationQuestion && (
                  <div className="fixed inset-0 z-[130] flex justify-end bg-black/30">
                    <button type="button" aria-label="Cerrar detalle de Educación Continua" onClick={() => setContinuingEducationDrawerOpen(false)} className="absolute inset-0" />
                    <aside className="relative z-10 flex h-full w-full max-w-2xl flex-col border-l bg-background shadow-2xl">
                      <div className="border-b p-6">
                        <div className="flex items-start justify-between gap-4">
                          <div>
                            <p className="sia-eyebrow">DETALLE · EDUCACIÓN CONTINUA</p>
                            <h3 className="mt-2 text-2xl font-semibold">{continuingEducationDisplayName(selectedContinuingEducationQuestion)}</h3>
                            <p className="mt-1 text-sm text-muted-foreground">{filters.periodo || "Periodo actual"}</p>
                          </div>
                          <button type="button" onClick={() => setContinuingEducationDrawerOpen(false)} className="rounded-xl border px-3 py-2 text-sm">Cerrar</button>
                        </div>
                      </div>
                      <div className="flex-1 overflow-y-auto p-6">
                        <div className="grid gap-3 sm:grid-cols-3">
                          <div className="rounded-xl border bg-card p-4"><p className="text-xs text-muted-foreground">Resultado 2026</p><p className="mt-1 text-2xl font-semibold">{selectedContinuingEducationQuestion.indice?.toFixed(2) ?? "—"}</p></div>
                          <div className="rounded-xl border bg-card p-4"><p className="text-xs text-muted-foreground">Comparable 2025</p><p className="mt-1 text-2xl font-semibold">{selectedContinuingEducationQuestion.historical_index?.toFixed(2) ?? "—"}</p></div>
                          <div className="rounded-xl border bg-card p-4"><p className="text-xs text-muted-foreground">Variación</p><p className={`mt-1 text-2xl font-semibold ${continuingEducationDelta === null ? "" : continuingEducationDelta >= 0 ? "text-emerald-700" : "text-red-700"}`}>{continuingEducationDelta === null ? "—" : `${continuingEducationDelta >= 0 ? "+" : ""}${continuingEducationDelta.toFixed(2)}`}</p></div>
                        </div>
                        <div className="mt-7">
                          <p className="text-sm font-semibold">Distribución de respuestas</p>
                          <div className="mt-4 flex h-4 overflow-hidden rounded-full bg-muted">
                            <div className="bg-emerald-500" style={{ width: `${selectedContinuingEducationQuestion.favorable_pct ?? 0}%` }} />
                            <div className="bg-amber-400" style={{ width: `${selectedContinuingEducationQuestion.neutral_pct ?? 0}%` }} />
                            <div className="bg-red-500" style={{ width: `${selectedContinuingEducationQuestion.desfavorable_pct ?? 0}%` }} />
                          </div>
                          <div className="mt-3 grid gap-2 sm:grid-cols-3">
                            <div className="rounded-lg border p-3 text-sm"><span className="text-emerald-700">Favorable</span><span className="float-right font-semibold">{selectedContinuingEducationQuestion.favorable_pct?.toFixed(1) ?? "0.0"}%</span></div>
                            <div className="rounded-lg border p-3 text-sm"><span className="text-amber-700">Neutral</span><span className="float-right font-semibold">{selectedContinuingEducationQuestion.neutral_pct?.toFixed(1) ?? "0.0"}%</span></div>
                            <div className="rounded-lg border p-3 text-sm"><span className="text-red-700">Desfavorable</span><span className="float-right font-semibold">{selectedContinuingEducationQuestion.desfavorable_pct?.toFixed(1) ?? "0.0"}%</span></div>
                          </div>
                        </div>
                        <div className="mt-7">
                          <div className="flex items-start justify-between gap-4"><div><p className="text-sm font-semibold">Comentarios relacionados</p><p className="mt-1 text-xs text-muted-foreground">Comentarios vinculados temáticamente con este indicador.</p></div><span className="rounded-full border bg-muted/30 px-3 py-1 text-xs font-medium">{selectedContinuingEducationComments.length}</span></div>
                          {selectedContinuingEducationComments.length > 0 ? (
                            <div className="mt-4 space-y-3">{selectedContinuingEducationComments.map((comment, index) => <div key={`${comment.fila_origen}-${comment.pregunta_id}-${index}`} className="rounded-xl border bg-card p-4"><p className="text-sm leading-6">{comment.texto}</p></div>)}</div>
                          ) : (
                            <div className="mt-4 rounded-xl border bg-muted/20 p-4 text-sm text-muted-foreground">No se identificaron comentarios relacionados directamente con este indicador.</div>
                          )}
                        </div>
                      </div>
                    </aside>
                  </div>
                )}
              </div>
            )}


            {isFacilitiesSection && facilitiesQuestions.length > 0 && (
              <div className="space-y-4">
                <div className="rounded-2xl border bg-card p-6 shadow-sm">
                  <div>
                    <p className="sia-eyebrow">INSTALACIONES Y EQUIPO TECNOLÓGICO</p>
                    <h3 className="mt-2 text-2xl font-semibold">Estado de instalaciones y recursos</h3>
                    <p className="mt-2 max-w-3xl text-sm text-muted-foreground">
                      Vista compacta de los rubros evaluados. Selecciona cualquiera para consultar su distribución, comparativo histórico y comentarios relacionados.
                    </p>
                  </div>
                  <div className="mt-6 grid gap-3 md:grid-cols-2">
                    {facilitiesQuestions.map((question) => {
                      const related = facilitiesCommentsForQuestion(question, summary?.comentarios ?? [])
                      const previous = question.historical_index ?? null
                      const delta = question.indice !== null && previous !== null ? question.indice - previous : null
                      return (
                        <button
                          key={question.pregunta_id}
                          type="button"
                          onClick={() => {
                            setSelectedFacilitiesQuestionId(question.pregunta_id)
                            setFacilitiesDrawerOpen(true)
                          }}
                          className="rounded-xl border bg-background p-4 text-left transition hover:border-primary/40 hover:bg-muted/20"
                        >
                          <div className="flex items-start justify-between gap-3">
                            <div className="min-w-0 flex-1">
                              <p className="font-semibold">{facilitiesDisplayName(question)}</p>
                              <p className="mt-1 text-xs text-muted-foreground">
                                {question.n.toLocaleString()} respuestas · {related.length} comentarios relacionados
                              </p>
                            </div>
                            <div className="shrink-0 text-right">
                              <p className="text-xl font-semibold">{question.indice?.toFixed(1) ?? "—"}</p>
                              <p className="text-[10px] text-muted-foreground">2026</p>
                            </div>
                          </div>
                          <div className="mt-3 h-2 overflow-hidden rounded-full bg-muted">
                            <div className={`h-full rounded-full ${getBarColor(question.indice)}`} style={{ width: `${Math.max(0, Math.min(100, question.indice ?? 0))}%` }} />
                          </div>
                          <div className="mt-3 flex items-center justify-between gap-3 text-xs">
                            <span className="text-muted-foreground">2025: {previous !== null ? previous.toFixed(1) : "Sin comparable"}</span>
                            <span className={delta === null ? "text-muted-foreground" : delta >= 0 ? "text-emerald-700" : "text-red-700"}>
                              {delta === null ? "—" : `${delta >= 0 ? "+" : ""}${delta.toFixed(1)} pts`}
                            </span>
                          </div>
                        </button>
                      )
                    })}
                  </div>
                </div>

                <div className="rounded-2xl border bg-card p-5 shadow-sm">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <p className="text-sm font-semibold">Sugerencias sobre instalaciones y equipo</p>
                      <p className="mt-1 text-xs text-muted-foreground">
                        Comentarios sustantivos registrados en la pregunta abierta de esta sección. Las respuestas sin propuesta de mejora se excluyen.
                      </p>
                    </div>
                    <span className="rounded-full border bg-muted/30 px-2.5 py-1 text-[10px] font-medium">{facilitiesSuggestions.length}</span>
                  </div>
                  {facilitiesSuggestions.length > 0 ? (
                    <div className="mt-4 grid gap-2 md:grid-cols-2">
                      {facilitiesSuggestions.slice(0, 8).map((comment, index) => (
                        <div key={`${comment.fila_origen}-${comment.pregunta_id}-${index}`} className="rounded-xl border bg-background p-3">
                          <p className="text-sm leading-5">{comment.texto}</p>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="mt-4 rounded-xl border bg-muted/20 p-4 text-sm text-muted-foreground">No se identificaron sugerencias sustantivas en la selección actual.</div>
                  )}
                </div>

                {facilitiesDrawerOpen && selectedFacilitiesQuestion && (
                  <div className="fixed inset-0 z-[130] flex justify-end bg-black/30">
                    <button type="button" aria-label="Cerrar detalle de instalaciones" onClick={() => setFacilitiesDrawerOpen(false)} className="absolute inset-0" />
                    <aside className="relative z-10 flex h-full w-full max-w-2xl flex-col border-l bg-background shadow-2xl">
                      <div className="border-b p-6">
                        <div className="flex items-start justify-between gap-4">
                          <div>
                            <p className="sia-eyebrow">DETALLE · INSTALACIONES</p>
                            <h3 className="mt-2 text-2xl font-semibold">{facilitiesDisplayName(selectedFacilitiesQuestion)}</h3>
                            <p className="mt-1 text-sm text-muted-foreground">{filters.periodo || "Periodo actual"}</p>
                          </div>
                          <button type="button" onClick={() => setFacilitiesDrawerOpen(false)} className="rounded-xl border px-3 py-2 text-sm">Cerrar</button>
                        </div>
                      </div>
                      <div className="flex-1 overflow-y-auto p-6">
                        <div className="grid gap-3 sm:grid-cols-3">
                          <div className="rounded-xl border bg-card p-4"><p className="text-xs text-muted-foreground">Resultado 2026</p><p className="mt-1 text-2xl font-semibold">{selectedFacilitiesQuestion.indice?.toFixed(2) ?? "—"}</p></div>
                          <div className="rounded-xl border bg-card p-4"><p className="text-xs text-muted-foreground">Comparable 2025</p><p className="mt-1 text-2xl font-semibold">{selectedFacilitiesQuestion.historical_index?.toFixed(2) ?? "—"}</p></div>
                          <div className="rounded-xl border bg-card p-4"><p className="text-xs text-muted-foreground">Variación</p><p className={`mt-1 text-2xl font-semibold ${facilitiesDelta === null ? "" : facilitiesDelta >= 0 ? "text-emerald-700" : "text-red-700"}`}>{facilitiesDelta === null ? "—" : `${facilitiesDelta >= 0 ? "+" : ""}${facilitiesDelta.toFixed(2)}`}</p></div>
                        </div>
                        <div className="mt-7">
                          <p className="text-sm font-semibold">Distribución de respuestas</p>
                          <div className="mt-4 flex h-4 overflow-hidden rounded-full bg-muted">
                            <div className="bg-emerald-500" style={{ width: `${selectedFacilitiesQuestion.favorable_pct ?? 0}%` }} />
                            <div className="bg-amber-400" style={{ width: `${selectedFacilitiesQuestion.neutral_pct ?? 0}%` }} />
                            <div className="bg-red-500" style={{ width: `${selectedFacilitiesQuestion.desfavorable_pct ?? 0}%` }} />
                          </div>
                          <div className="mt-3 grid gap-2 sm:grid-cols-3">
                            <div className="rounded-lg border p-3 text-sm"><span className="text-emerald-700">Favorable</span><span className="float-right font-semibold">{selectedFacilitiesQuestion.favorable_pct?.toFixed(1) ?? "0.0"}%</span></div>
                            <div className="rounded-lg border p-3 text-sm"><span className="text-amber-700">Neutral</span><span className="float-right font-semibold">{selectedFacilitiesQuestion.neutral_pct?.toFixed(1) ?? "0.0"}%</span></div>
                            <div className="rounded-lg border p-3 text-sm"><span className="text-red-700">Desfavorable</span><span className="float-right font-semibold">{selectedFacilitiesQuestion.desfavorable_pct?.toFixed(1) ?? "0.0"}%</span></div>
                          </div>
                        </div>
                        <div className="mt-7">
                          <div className="flex items-start justify-between gap-4">
                            <div><p className="text-sm font-semibold">Comentarios relacionados</p><p className="mt-1 text-xs text-muted-foreground">Sugerencias vinculadas temáticamente con este rubro.</p></div>
                            <span className="rounded-full border bg-muted/30 px-3 py-1 text-xs font-medium">{selectedFacilitiesComments.length}</span>
                          </div>
                          {selectedFacilitiesComments.length > 0 ? (
                            <div className="mt-4 space-y-3">{selectedFacilitiesComments.map((comment, index) => <div key={`${comment.fila_origen}-${comment.pregunta_id}-${index}`} className="rounded-xl border bg-card p-4"><p className="text-sm leading-6">{comment.texto}</p></div>)}</div>
                          ) : (
                            <div className="mt-4 rounded-xl border bg-muted/20 p-4 text-sm text-muted-foreground">No se identificaron comentarios relacionados directamente con este rubro.</div>
                          )}
                        </div>
                      </div>
                    </aside>
                  </div>
                )}
              </div>
            )}

            {effectiveSectionId !== "ESC_02_DIRECCION" &&
              effectiveSectionId !== "ESC_03_SERVICIOS" &&
              effectiveSectionId !== "ESC_04_ACADEMICOS" &&
              effectiveSectionId !== "ESC_05_IDIOMAS" &&
              !isContinuingEducationSection &&
              !isFacilitiesSection &&
              selectedSectionIndexQuestions.length > 0 && (
                    <div>
                      <div className="mb-4">
                        <p className="text-sm font-semibold">
                          Reactivos que integran el índice
                        </p>
                        <p className="mt-1 text-xs text-muted-foreground">
                          Estos reactivos determinan el resultado de la
                          sección.
                        </p>
                      </div>

                      <div className="space-y-4">
                        {selectedSectionIndexQuestions.map(
                          (question) => (
                            <QuestionResultCard
                              key={question.pregunta_id}
                              question={question}
                            />
                          ),
                        )}
                      </div>
                    </div>
                  )}

                  {effectiveSectionId !== "ESC_02_DIRECCION" &&
              effectiveSectionId !== "ESC_03_SERVICIOS" &&
              effectiveSectionId !== "ESC_04_ACADEMICOS" &&
              effectiveSectionId !== "ESC_05_IDIOMAS" &&
              !isContinuingEducationSection &&
              !isFacilitiesSection &&
              selectedSectionComplementaryQuestions.length > 0 && (
                    <div>
                      <div className="mb-4 border-t pt-6">
                        <p className="text-sm font-semibold">
                          Indicadores complementarios
                        </p>
                        <p className="mt-1 text-xs text-muted-foreground">
                          Aportan contexto para la toma de decisiones,
                          pero no forman parte del cálculo del
                          índice de la sección.
                        </p>
                      </div>

                      <div className="space-y-4">
                        {selectedSectionComplementaryQuestions.map(
                          (question) => (
                            <QuestionResultCard
                              key={question.pregunta_id}
                              question={question}
                            />
                          ),
                        )}
                      </div>
                    </div>
                  )}
                </div>
              )}


              {effectiveSectionId !== "ESC_00_PERFIL" &&
                effectiveSectionId !== "ESC_03_SERVICIOS" &&
                effectiveSectionId !== "ESC_04_ACADEMICOS" &&
                effectiveSectionId !== "ESC_05_IDIOMAS" &&
                !isContinuingEducationSection &&
                !isFacilitiesSection &&
                selectedSectionComments.length > 0 && (
                  <div className="rounded-2xl border bg-card p-6 shadow-sm">
                    <div className="flex flex-col gap-5 lg:flex-row lg:items-start lg:justify-between">
                      <div>
                        <p className="sia-eyebrow">
                          PREGUNTAS ABIERTAS
                        </p>

                        <h3 className="mt-2 text-2xl font-semibold">
                          Respuestas abiertas por sección
                        </h3>

                        <p className="mt-2 max-w-3xl text-sm text-muted-foreground">
                          Respuestas abiertas relacionadas exclusivamente con la
                          sección y los filtros que estás consultando.
                        </p>
                      </div>

                      <button
                        type="button"
                        onClick={() => setAssistantOpen(true)}
                        className="rounded-xl bg-primary px-5 py-2.5 text-sm font-medium text-primary-foreground transition hover:opacity-95"
                      >
                        Preguntar a SIA sobre esta sección
                      </button>
                    </div>

                    <div className="mt-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
                      <div className="rounded-xl border bg-background p-4">
                        <p className="text-xs text-muted-foreground">
                          Comentarios útiles
                        </p>

                        <p className="mt-2 text-2xl font-semibold">
                          {selectedSectionQualitativeSummary?.total ??
                            selectedSectionComments.length}
                        </p>
                      </div>

                      <div className="rounded-xl border bg-background p-4">
                        <p className="text-xs text-muted-foreground">
                          Positivos explícitos
                        </p>

                        <p className="mt-2 text-2xl font-semibold">
                          {selectedSectionQualitativeSummary?.positivos ?? 0}
                        </p>
                      </div>

                      <div className="rounded-xl border bg-background p-4">
                        <p className="text-xs text-muted-foreground">
                          Oportunidades / incidencias
                        </p>

                        <p className="mt-2 text-2xl font-semibold">
                          {(selectedSectionQualitativeSummary?.oportunidades ??
                            0) +
                            (selectedSectionQualitativeSummary?.incidencias ??
                              0)}
                        </p>
                      </div>

                      <div className="rounded-xl border bg-background p-4">
                        <p className="text-xs text-muted-foreground">
                          Comentarios libres
                        </p>

                        <p className="mt-2 text-2xl font-semibold">
                          {selectedSectionQualitativeSummary?.sin_clasificar ??
                            0}
                        </p>

                        <p className="mt-1 text-[11px] text-muted-foreground">
                          Pendientes de interpretación semántica
                        </p>
                      </div>
                    </div>

                    {selectedSectionCommentSamples.length > 0 && (
                      <div className="mt-6">
                        <div className="flex items-center justify-between gap-4">
                          <div>
                            <p className="text-sm font-semibold">
                              Respuestas abiertas de la selección
                            </p>

                            <p className="mt-1 text-xs text-muted-foreground">
                              Muestra de respuestas abiertas registradas.
                            </p>
                          </div>

                          <button
                            type="button"
                            onClick={() => setActiveTab("comentarios")}
                            className="text-sm font-medium text-primary hover:underline"
                          >
                            Ver todos los comentarios
                          </button>
                        </div>

                        <div className="mt-4 grid gap-3 md:grid-cols-2">
                          {selectedSectionCommentSamples.map(
                            (comment, index) => (
                              <div
                                key={`${comment.fila_origen}-${comment.pregunta_id}-${index}`}
                                className="rounded-xl border bg-muted/20 p-4"
                              >
                                <p className="text-xs font-medium text-muted-foreground">
                                  {comment.texto_corto}
                                </p>

                                <p className="mt-2 text-sm leading-6">
                                  {comment.texto}
                                </p>

                                {comment.persona_evaluada && (
                                  <p className="mt-3 text-xs text-muted-foreground">
                                    Sobre: {comment.persona_evaluada}
                                    {comment.rol_evaluado
                                      ? ` · ${comment.rol_evaluado}`
                                      : ""}
                                  </p>
                                )}
                              </div>
                            ),
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                )}
            </div>
          )}

          {summary && !loading && activeTab === "comparativos" && (
  <div className="mt-8 space-y-6">
    <div className="rounded-2xl border bg-card p-6 shadow-sm">
      <p className="sia-eyebrow">COMPARATIVO HISTÓRICO</p>

      <h2 className="mt-2 text-2xl font-semibold">
        Encuesta de Calidad 2025 vs 2026
      </h2>

      <p className="mt-2 max-w-3xl text-sm text-muted-foreground">
        La comparación considera únicamente los reactivos marcados como
        comparables históricamente en ambos instrumentos. Los indicadores
        incorporados a partir de 2026 no se mezclan con la medición 2025.
      </p>
    </div>

    {!historico2025 || !historico2026 ? (
      <div className="rounded-2xl border bg-card p-8 shadow-sm">
        <p className="text-sm text-muted-foreground">
          Cargando comparativo histórico...
        </p>
      </div>
    ) : (
      <>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <div className="rounded-2xl border bg-card p-6 shadow-sm">
            <p className="text-sm text-muted-foreground">
              Índice histórico 2025
            </p>

            <p className="mt-3 text-4xl font-semibold tracking-tight">
              {historico2025.indice_general?.toFixed(2) ?? "—"}
            </p>

            <div className="mt-3">
              <span
                className={`rounded-full border px-3 py-1 text-xs font-medium ${getNivelClasses(
                  historico2025.indice_general,
                )}`}
              >
                {getNivel(historico2025.indice_general)}
              </span>
            </div>
          </div>

          <div className="rounded-2xl border bg-card p-6 shadow-sm">
            <p className="text-sm text-muted-foreground">
              Índice histórico 2026
            </p>

            <p className="mt-3 text-4xl font-semibold tracking-tight">
              {historico2026.indice_general?.toFixed(2) ?? "—"}
            </p>

            <div className="mt-3">
              <span
                className={`rounded-full border px-3 py-1 text-xs font-medium ${getNivelClasses(
                  historico2026.indice_general,
                )}`}
              >
                {getNivel(historico2026.indice_general)}
              </span>
            </div>
          </div>

          <div className="rounded-2xl border bg-card p-6 shadow-sm">
            <p className="text-sm text-muted-foreground">
              Variación
            </p>

            <p
              className={`mt-3 text-4xl font-semibold tracking-tight ${
                historico2025.indice_general !== null &&
                historico2026.indice_general !== null &&
                historico2026.indice_general -
                  historico2025.indice_general >=
                  0
                  ? "text-emerald-700"
                  : "text-red-700"
              }`}
            >
              {historico2025.indice_general !== null &&
              historico2026.indice_general !== null
                ? `${
                    historico2026.indice_general -
                      historico2025.indice_general >=
                    0
                      ? "+"
                      : ""
                  }${(
                    historico2026.indice_general -
                    historico2025.indice_general
                  ).toFixed(2)}`
                : "—"}
            </p>

            <p className="mt-2 text-xs text-muted-foreground">
              Puntos de diferencia 2026 vs 2025
            </p>
          </div>

          <div className="rounded-2xl border bg-card p-6 shadow-sm">
            <p className="text-sm text-muted-foreground">
              Reactivos comparables
            </p>

            <p className="mt-3 text-4xl font-semibold tracking-tight">
              {Math.min(
                historico2025.preguntas_con_datos,
                historico2026.preguntas_con_datos,
              )}
            </p>

            <p className="mt-2 text-xs text-muted-foreground">
              Reactivos históricos con información disponible
            </p>
          </div>
        </div>

        <div className="rounded-2xl border bg-card p-6 shadow-sm">
          <div>
            <p className="sia-eyebrow">POR SECCIÓN</p>

            <h3 className="mt-2 text-xl font-semibold">
              Evolución de los indicadores comparables
            </h3>

            <p className="mt-1 text-sm text-muted-foreground">
              Se comparan únicamente secciones que cuentan con índice
              histórico en ambos periodos.
            </p>
          </div>

          <div className="mt-6 space-y-3">
            {historico2026.secciones
              .filter(
                (seccion2026) =>
                  seccion2026.indice !== null &&
                  historico2025.secciones.some(
                    (seccion2025) =>
                      seccion2025.seccion_nombre ===
                        seccion2026.seccion_nombre &&
                      seccion2025.indice !== null,
                  ),
              )
              .map((seccion2026) => {
                const seccion2025 =
                  historico2025.secciones.find(
                    (item) =>
                      item.seccion_nombre ===
                      seccion2026.seccion_nombre,
                  )

                if (
                  !seccion2025 ||
                  seccion2025.indice === null ||
                  seccion2026.indice === null
                ) {
                  return null
                }

                const diferencia =
                  seccion2026.indice - seccion2025.indice

                return (
                  <div
                    key={seccion2026.seccion_nombre}
                    className="rounded-xl border bg-background px-5 py-4"
                  >
                    <div className="grid gap-5 xl:grid-cols-[240px_1fr_280px] xl:items-center">
                      <div>
                        <p className="text-sm font-semibold">
                          {seccion2026.seccion_nombre}
                        </p>

                        <p className="mt-1 text-xs text-muted-foreground">
                          Comparación del índice de la sección
                        </p>
                      </div>

                      <div className="space-y-2">
                        <div className="grid grid-cols-[42px_1fr_48px] items-center gap-2">
                          <span className="text-[11px] text-muted-foreground">
                            2025
                          </span>

                          <div className="h-2.5 overflow-hidden rounded-full bg-slate-100">
                            <div
                              className="h-full rounded-full bg-slate-400"
                              style={{
                                width: `${Math.max(
                                  0,
                                  Math.min(
                                    100,
                                    seccion2025.indice,
                                  ),
                                )}%`,
                              }}
                            />
                          </div>

                          <span className="text-right text-xs font-semibold">
                            {seccion2025.indice.toFixed(2)}
                          </span>
                        </div>

                        <div className="grid grid-cols-[42px_1fr_48px] items-center gap-2">
                          <span className="text-[11px] text-muted-foreground">
                            2026
                          </span>

                          <div className="h-2.5 overflow-hidden rounded-full bg-emerald-50">
                            <div
                              className="h-full rounded-full bg-emerald-600"
                              style={{
                                width: `${Math.max(
                                  0,
                                  Math.min(
                                    100,
                                    seccion2026.indice,
                                  ),
                                )}%`,
                              }}
                            />
                          </div>

                          <span className="text-right text-xs font-semibold">
                            {seccion2026.indice.toFixed(2)}
                          </span>
                        </div>
                      </div>

                      <div className="grid grid-cols-3 gap-4 text-right">
                        <div>
                          <p className="text-[11px] text-muted-foreground">
                            2025
                          </p>

                          <p className="mt-1 text-lg font-semibold">
                            {seccion2025.indice.toFixed(2)}
                          </p>
                        </div>

                        <div>
                          <p className="text-[11px] text-muted-foreground">
                            2026
                          </p>

                          <p className="mt-1 text-lg font-semibold">
                            {seccion2026.indice.toFixed(2)}
                          </p>
                        </div>

                        <div>
                          <p className="text-[11px] text-muted-foreground">
                            Variación
                          </p>

                          <p
                            className={`mt-1 text-lg font-semibold ${
                              diferencia >= 0
                                ? "text-emerald-700"
                                : "text-red-600"
                            }`}
                          >
                            {diferencia >= 0 ? "+" : ""}
                            {diferencia.toFixed(2)}
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                )
              })}
          </div>
        </div>
      </>
    )}
  </div>
)}

          {summary && !loading && activeTab === "comentarios" && (
            <div className="mt-8 rounded-2xl border bg-card p-8 shadow-sm">
              <p className="sia-eyebrow">PREGUNTAS ABIERTAS</p>
              <h2 className="mt-2 text-2xl font-semibold">
                Comentarios
              </h2>
              <p className="mt-2 text-sm text-muted-foreground">
                Aquí concentraremos respuestas abiertas, búsquedas,
                temas recurrentes y análisis cualitativo.
              </p>
            </div>
          )}


        </div>
      </main>

      {summary && !loading && (
            <>
              <button
                type="button"
                onClick={() => setAssistantOpen(true)}
                className="fixed bottom-8 right-8 z-[100] rounded-2xl border bg-primary px-5 py-4 text-left text-primary-foreground shadow-2xl transition hover:opacity-95"
              >
                <span className="block text-xs font-semibold uppercase tracking-[0.14em] opacity-80">
                  SIA Intelligence
                </span>

                <span className="mt-1 block font-semibold">
                  Asistente SIA · En construcción
                </span>
              </button>

              {assistantOpen && (
                <div className="fixed inset-0 z-50 flex justify-end bg-black/20">
                  <button
                    type="button"
                    aria-label="Cerrar Asistente SIA"
                    onClick={() => setAssistantOpen(false)}
                    className="absolute inset-0"
                  />

                  <aside className="relative z-10 flex h-full w-full max-w-md flex-col border-l bg-background shadow-2xl">
                    <div className="border-b p-6">
                      <div className="flex items-start justify-between gap-4">
                        <div>
                          <p className="sia-eyebrow">
                            INTELIGENCIA INSTITUCIONAL
                          </p>

                          <div className="mt-2 flex flex-wrap items-center gap-2">
                            <h2 className="text-2xl font-semibold">
                              Asistente SIA
                            </h2>
                            <span className="rounded-full border bg-muted/30 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                              En construcción
                            </span>
                          </div>

                          <p className="mt-2 text-sm text-muted-foreground">
                            Vista conceptual del asistente de inteligencia institucional. La conexión con IA y las consultas en lenguaje natural se habilitarán en una fase posterior.
                          </p>
                        </div>

                        <button
                          type="button"
                          onClick={() => setAssistantOpen(false)}
                          className="rounded-xl border px-3 py-2 text-sm"
                        >
                          Cerrar
                        </button>
                      </div>
                    </div>

                    <div className="border-b bg-muted/30 p-5">
                      <p className="text-xs font-semibold uppercase tracking-[0.12em] text-muted-foreground">
                        Contexto activo
                      </p>

                      <div className="mt-3 space-y-1 text-sm">
                        <p>
                          <span className="text-muted-foreground">
                            Periodo:
                          </span>{" "}
                          {filters.periodo || "Todos"}
                        </p>

                        <p>
                          <span className="text-muted-foreground">
                            Modalidad:
                          </span>{" "}
                          {filters.modalidad || "Todas"}
                        </p>

                        <p>
                          <span className="text-muted-foreground">
                            Servicio:
                          </span>{" "}
                          {filters.servicio || "Todos"}
                        </p>

                        {activeTab === "secciones" && selectedSection && (
                          <p>
                            <span className="text-muted-foreground">
                              Sección:
                            </span>{" "}
                            {selectedSection.seccion_id === "ESC_00_PERFIL"
                              ? "Perfil de respuestas"
                              : selectedSection.seccion_nombre}
                          </p>
                        )}
                      </div>
                    </div>

                    <div className="flex flex-1 flex-col p-5">
                      <div className="flex-1 overflow-y-auto">
                        <div className="rounded-2xl border bg-muted/20 p-4">
                          <p className="text-xs font-semibold uppercase tracking-[0.12em] text-muted-foreground">
                            Respuesta
                          </p>
                          <p className="mt-2 text-sm leading-6">
                            Módulo en construcción. La versión final permitirá consultar KPIs, comentarios, comparativos y hallazgos de la selección activa mediante lenguaje natural.
                          </p>
                        </div>
                      </div>

                      <div className="mt-4 border-t pt-4">
                        <textarea
                          value={siaQuestion}
                          onChange={(event) => setSiaQuestion(event.target.value)}
                          onKeyDown={(event) => {
                            if (event.key === "Enter" && !event.shiftKey) {
                              event.preventDefault()
                              askSia()
                            }
                          }}
                          rows={3}
                          placeholder="Funcionalidad disponible en una próxima fase"
                          disabled
                          className="w-full resize-none rounded-xl border bg-muted/30 p-3 text-sm text-muted-foreground outline-none disabled:cursor-not-allowed"
                        />
                        <button
                          type="button"
                          onClick={askSia}
                          disabled
                          className="mt-3 w-full cursor-not-allowed rounded-xl border bg-muted/40 px-4 py-2.5 text-sm font-medium text-muted-foreground"
                        >
                          SIA · En construcción
                        </button>
                      </div>
                    </div>
                  </aside>
                </div>
              )}
            </>
          )}
    </div>
  )
}
