import { downloadResultsReportPdf } from "@/services/resultsReportPdf"
import { useEffect, useState } from "react"
import "@/components/calidad/qualityPresentation.css"
import { GlobalSectionSummary, MetricComparison, MetricChange, MetricCard, MetricSkeleton, MetricState } from "@/components/common/CalidadMetrics"
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

import { DirectionCareerList, DirectionCriteria, DirectionGlobal, useDirectionHistory } from "@/components/calidad/DirectionResults"
import { AcademicServices } from "@/components/calidad/AcademicServices"
import { Facilities } from "@/components/calidad/Facilities"
import { SchoolEnvironment } from "@/components/calidad/SchoolEnvironment"
import { Satisfaction } from "@/components/calidad/Satisfaction"
import { LanguageCenter } from "@/components/calidad/LanguageCenter"
import { ContinuingEducation } from "@/components/calidad/ContinuingEducation"
import { InstitutionalServices } from "@/components/calidad/InstitutionalServices"
import { currentPopulations, populationKey } from "@/components/calidad/directionModel"

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
  question: Pick<CalidadQuestion, "texto_corto" | "pregunta_id">,
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


function downloadQualityReportPdf(
  _summary: CalidadSummary,
  filters: CalidadFilterSelection,
  _historico2025: CalidadSummary | null,
  _historico2026: CalidadSummary | null,
) {
  void downloadResultsReportPdf({ ...filters }, true).catch(error => {
    window.alert(error instanceof Error ? error.message : "No fue posible generar el informe de resultados.")
  })
}

function getNivel(indice: number | null) {
  if (indice === null) return "Sin respuestas en este periodo"
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
              {question.n} respuestas a criterios
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
              : "Resultado actual"}
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
  const [historyStatus, setHistoryStatus] = useState<"loading" | "ready" | "error">("loading")
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
    const controller = new AbortController()
    let active = true
    async function cargarDatos() {
      try {
        setLoading(true)
        setError(null)

        const data = await getCalidadSummary(filters, controller.signal)

        if (active) setSummary(data)
      } catch (err) {
        if (!active) return
        setError(
          err instanceof Error
            ? err.message
            : "No fue posible cargar Encuesta de Calidad.",
        )
      } finally {
        if (active) setLoading(false)
      }
    }

    cargarDatos()
    return () => { active = false; controller.abort() }
  }, [
    filters.periodo,
    filters.modalidad,
    filters.servicio,
  ])

  useEffect(() => {
    const controller = new AbortController()
    let active = true
    setHistoryStatus("loading")
    setHistorico2025(null)
    setHistorico2026(null)
    async function cargarHistorico(
      periodo: "2025" | "2026",
    ): Promise<CalidadSummary> {
      const buildUrl = () => {
        const params = new URLSearchParams()
        params.set("periodo", periodo)
        params.set("historico", "true")

        if (filters.modalidad) {
          params.set("modalidad", filters.modalidad)
        }
        if (filters.servicio) {
          params.set("servicio", filters.servicio)
        }

        return `http://127.0.0.1:8000/api/encuesta-calidad/preview/resumen?${params.toString()}`
      }

      const request = async () => {
        const response = await fetch(buildUrl(), { signal: controller.signal })
        if (!response.ok) {
          throw new Error(
            `No fue posible cargar el histórico ${periodo}.`,
          )
        }
        return (await response.json()) as CalidadSummary
      }

      return request()
    }

    async function cargarComparativosHistoricos() {
      try {
        const [data2025, data2026] = await Promise.all([
          cargarHistorico("2025"),
          cargarHistorico("2026"),
        ])

        if (!active) return
        setHistorico2025(data2025)
        setHistorico2026(data2026)
        setHistoryStatus("ready")
      } catch (err) {
        if (!active) return
        console.error(
          "No fue posible cargar los comparativos históricos.",
          err,
        )
        setHistorico2025(null)
        setHistorico2026(null)
        setHistoryStatus("error")
      }
    }

    cargarComparativosHistoricos()
    return () => { active = false; controller.abort() }
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

  const selectedSectionPeople = Array.from(new Map([
    ...(summary?.comparativos_personal?.asignaciones ?? []).map(item => ({
      seccion_id: item.seccion_id, seccion_nombre: "Asistentes/coordinadores",
      persona_evaluada: item.persona_evaluada, rol_evaluado: item.rol_evaluado,
      indice: null, preguntas_con_datos: 0, evaluaciones: 0,
    })),
    ...(summary?.personas ?? []),
  ].filter(
      (person) =>
        canonicalSectionId(person.seccion_id) ===
        effectiveSectionId,
    ).map(person => [`${person.persona_evaluada}|||${person.rol_evaluado}`, person])).values())

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
        ?.asignaciones ?? summary?.comparativos_personal?.personas_por_carrera_modalidad ?? [])
        .filter(
          (item) =>
            Boolean(effectiveStaffOverviewPerson) &&
            normalizeAcademicLabel(item.persona_evaluada) ===
              normalizeAcademicLabel(
                effectiveStaffOverviewPerson?.persona_evaluada ?? "",
              ),
        )
        .map((item) => [
          populationKey(item),
          {
            carrera_base: item.carrera_base,
            modalidad_academica: item.modalidad_academica,
            evaluaciones: item.evaluaciones,
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

  const directionCareerModalities = currentPopulations(summary)
  const directionHistory = useDirectionHistory(
    effectiveSectionId === "ESC_02_DIRECCION" && !loading && filters.periodo === "2026",
    filters.modalidad,
    filters.servicio,
  )

  const selectedDirectionPopulation =
    directionCareerModalities.find(
      (item) =>
        item.carrera_base ===
          selectedDirectionCareer &&
        item.modalidad_academica ===
          selectedDirectionModality,
    ) ??
    null

  const effectiveDirectionCareer =
    selectedDirectionPopulation
      ?.carrera_base ?? ""

  const effectiveDirectionModality =
    selectedDirectionPopulation
      ?.modalidad_academica ?? ""

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

  const isContinuingEducationSection = effectiveSectionId === "ESC_06_EDUC_CONT"

  const isFacilitiesSection = effectiveSectionId === "ESC_07_INSTALACIONES"
  const isEnvironmentSection = effectiveSectionId === "ESC_08_AMBIENTE"
  const isSatisfactionSection = effectiveSectionId === "ESC_09_SATISFACCION"
  const isLanguageCenterSection = effectiveSectionId === "ESC_05_IDIOMAS"

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

  return (
    <div className="calidad-page flex min-h-screen bg-background">
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
              <p className="sr-only">
                Cargando resultados de Encuesta de Calidad...
              </p>
              <MetricSkeleton />
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
                        <div className="flex flex-wrap items-start justify-between gap-4">
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
                        <div className="flex flex-wrap items-start justify-between gap-4">
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

              {selectedSection && effectiveSectionId === "ESC_01_PERSONAL" && (
                <GlobalSectionSummary title="Asistentes/coordinadores">
                  <div className="grid gap-5 sm:grid-cols-2 xl:grid-cols-4">
                    <div><p className="text-sm text-muted-foreground">Resultado general</p><p className="mt-1 text-3xl font-semibold">{selectedSection.indice?.toFixed(2) ?? "—"}</p></div>
                    <div><p className="text-sm text-muted-foreground">Respuestas a criterios</p><p className="mt-1 text-3xl font-semibold">{selectedSectionQuestions.reduce((total, question) => total + question.n, 0).toLocaleString()}</p></div>
                    <div><p className="text-sm text-muted-foreground">Encuestas analizadas</p><p className="mt-1 text-3xl font-semibold">{(staffSummary?.evaluaciones ?? 0).toLocaleString()}</p></div>
                    <div><p className="text-sm text-muted-foreground">Responsables evaluados</p><p className="mt-1 text-3xl font-semibold">{staffSummary?.personas_evaluadas ?? selectedSectionPeople.length}</p></div>
                  </div>
                  <p className="text-xs text-muted-foreground">{staffSummary?.criterios_evaluados ?? staffUniqueCriteriaCount} criterios evaluados · Una encuesta puede aportar respuestas a varios criterios.</p>
                  {selectedSection.indice === null && <MetricState empty />}
                  <div className="space-y-3 border-t pt-4">
                    <div className="flex flex-wrap items-baseline justify-between gap-2"><h4 className="font-semibold">Evolución global comparable</h4>{historyStatus === "ready" && <MetricChange delta={selectedSectionHistoricalDelta} />}</div>
                    <p className="text-xs text-muted-foreground">Solo información metodológicamente equivalente · {selectedSectionComparableQuestions} reactivos comparables.</p>
                    {historyStatus === "error" ? <p className="text-xs text-muted-foreground">No fue posible cargar el comparativo histórico</p> : historyStatus === "loading" ? <MetricSkeleton /> : selectedSection2025?.indice != null && selectedSection2026?.indice != null ? <MetricComparison previous={selectedSection2025.indice} current={selectedSection2026.indice} /> : <MetricState />}
                  </div>
                </GlobalSectionSummary>
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
                          Responsables · Asistentes y coordinadores
                        </h3>

                        <p className="mt-2 max-w-3xl text-sm text-muted-foreground">
                          Consulta una carrera o un responsable y abre su detalle
                          con criterios y comentarios.
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
                              <div className="flex flex-wrap items-start justify-between gap-4">
                                <div className="min-w-0 flex-1">
                                  <p className="font-semibold">
                                    {career.carrera_base}
                                  </p>

                                  <div className="mt-2 flex flex-wrap items-center gap-2">
                                    <span className="rounded-full border px-2.5 py-1 text-[10px] font-medium text-muted-foreground">
                                      {career.modalidad_academica}
                                    </span>

                                    <span className="text-xs text-muted-foreground">
                                      {career.evaluaciones.toLocaleString()} encuestas analizadas
                                    </span>

                                    <span className="text-xs text-muted-foreground">
                                      · {career.personas_evaluadas} {career.personas_evaluadas === 1 ? "responsable evaluado" : "responsables evaluados"}
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
                                      Sin respuestas en este periodo
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
                                <div className="flex flex-wrap items-start justify-between gap-4">
                                  <div className="min-w-0">
                                    <p className="text-xs text-muted-foreground">
                                      {person.rol_evaluado}
                                    </p>
                                    <p className="mt-1 font-semibold">
                                      {person.persona_evaluada}
                                    </p>
                                    <p className="mt-1 text-xs text-muted-foreground">
                                      {person.evaluaciones.toLocaleString()} encuestas analizadas
                                      {" · "}
                                      {person.preguntas_con_datos} criterios evaluados
                                    </p>
                                  </div>
                                  <div className="shrink-0 text-right">
                                    <p className="text-xs text-muted-foreground">
                                      Resultado actual
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
                          <div className="flex flex-wrap items-start justify-between gap-4">
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
                          <div className="grid gap-3 sm:grid-cols-2">
                            <div className="rounded-xl border bg-card p-4">
                              <p className="text-xs text-muted-foreground">
                                Resultado actual
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
                                Encuestas analizadas
                              </p>
                              <p className="mt-1 text-2xl font-semibold">
                                {effectiveStaffOverviewPerson.evaluaciones.toLocaleString()}
                              </p>
                            </div>

                            <div className="rounded-xl border bg-card p-4">
                              <p className="text-xs text-muted-foreground">
                                Criterios evaluados
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
                                    {!assignment.evaluaciones && " · Sin respuestas en este periodo"}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}

                          <div className="mt-7">
                            <p className="text-sm font-semibold">
                              Resultado por criterio
                            </p>
                            <p className="mt-1 text-xs text-muted-foreground">
                              Los cuatro componentes de atención se muestran por separado
                              para evitar que el promedio oculte diferencias.
                            </p>

                            <div className="mt-4 space-y-3">
                              {staffOverviewQuestions.map((question) => (<MetricCard key={question.pregunta_id} title={question.texto_corto} current={question.indice} previous={null} responses={question.n} distribution={question} currentYear={filters.periodo || "2026"} />))}
                            </div>
                          </div>

                          <div className="mt-7">
                            <div className="flex items-end justify-between gap-4">
                              <div>
                                <p className="text-sm font-semibold">
                                  Comentarios de estudiantes
                                </p>
                                <p className="mt-2 text-xs text-muted-foreground">Análisis cualitativo con SIA · En construcción</p>
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
                                  de las respuestas a criterios registradas.
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
                          <div className="flex flex-wrap items-start justify-between gap-4">
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
                          <div className="grid gap-3 sm:grid-cols-2">
                            <div className="rounded-xl border bg-card p-4">
                              <p className="text-xs text-muted-foreground">Resultado actual</p>
                              <p className="mt-1 text-2xl font-semibold">
                                {selectedStaffModalitySummary.indice?.toFixed(2) ?? "—"}
                              </p>
                            </div>
                            <div className="rounded-xl border bg-card p-4">
                              <p className="text-xs text-muted-foreground">Encuestas analizadas</p>
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
                              <p className="text-xs text-muted-foreground">Responsables evaluados</p>
                              <p className="mt-1 text-2xl font-semibold">
                                {selectedStaffModalitySummary.personas_evaluadas}
                              </p>
                            </div>
                          </div>

                          {staffPeopleForCareer.length > 0 ? (
                            <div className="mt-7">
                              <p className="text-sm font-semibold">
                                Responsables evaluados
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
                                            {person.evaluaciones} encuestas analizadas
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
                                Sin respuestas en este periodo
                              </p>
                              <p className="mt-1 text-xs text-muted-foreground">
                                Esta población existe, pero no tiene respuestas válidas
                                asociadas a personal en los filtros actuales.
                              </p>
                            </div>
                          )}

                          {effectiveStaffPerson && (
                            <div className="mt-7 border-t pt-6">
                              <div className="flex flex-wrap items-start justify-between gap-4">
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
                                    Resultado actual
                                  </p>
                                  <p className="mt-1 text-2xl font-semibold">
                                    {effectiveStaffPerson.indice?.toFixed(2) ?? "—"}
                                  </p>
                                </div>
                              </div>

                              <p className="mt-6 text-xs text-muted-foreground">Sin comparable histórico</p>

                              <div className="mt-5 space-y-3">
                                {staffQuestionsByCareer.map((question) => (<MetricCard key={question.pregunta_id} title={question.texto_corto} current={question.indice} previous={null} responses={question.n} distribution={question} currentYear={filters.periodo || "2026"} />))}
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
                      <DirectionGlobal current={summary} historical={directionHistory.data} year={filters.periodo || "2026"}
                        loading={directionHistory.loading} error={directionHistory.error} filtered={Boolean(filters.servicio)} />

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
                        <DirectionCareerList current={summary} historical={directionHistory.data} year={filters.periodo || "2026"}
                          loading={directionHistory.loading} error={directionHistory.error}
                          onSelect={population => {
                            setSelectedDirectionCareer(population.carrera_base)
                            setSelectedDirectionModality(population.modalidad_academica)
                            setDirectionCareerDrawerOpen(true)
                          }} />
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
                                    <div className="flex flex-wrap items-start justify-between gap-4">
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
                                          {director.evaluaciones.toLocaleString()} encuestas analizadas
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
                                            Sin respuestas en este periodo
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
                              <div className="flex flex-wrap items-start justify-between gap-4">
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
                                  <p className="text-xs text-muted-foreground">Resultado actual</p>
                                  <p className="mt-1 text-2xl font-semibold">
                                    {selectedDirectionPopulation.indice?.toFixed(2) ?? "—"}
                                  </p>
                                </div>
                                <div className="rounded-xl border bg-card p-4">
                                  <p className="text-xs text-muted-foreground">Encuestas analizadas</p>
                                  <p className="mt-1 text-2xl font-semibold">
                                    {selectedDirectionPopulation.encuestas.toLocaleString()}
                                  </p>
                                </div>
                                <div className="rounded-xl border bg-card p-4">
                                  <p className="text-xs text-muted-foreground">Criterios evaluados</p>
                                  <p className="mt-1 text-2xl font-semibold">
                                    {selectedDirectionPopulation.preguntas_con_datos}
                                  </p>
                                </div>
                              </div>

                              {selectedDirectionPopulation.estado !== "con_datos" && (
                                <div className="mt-7 rounded-2xl border bg-muted/30 p-5">
                                  <p className="text-sm font-semibold">
                                    Sin respuestas en este periodo
                                  </p>
                                  <p className="mt-2 text-sm leading-6 text-muted-foreground">
                                    La carrera forma parte del catálogo institucional, pero no cuenta con evaluaciones válidas de Dirección para el periodo y filtros seleccionados.
                                  </p>
                                </div>
                              )}

                              <div className="mt-7">
                                <p className="text-sm font-semibold">Criterios evaluados</p>
                                <div className="mt-4">
                                  <DirectionCriteria current={summary} historical={directionHistory.data}
                                    population={selectedDirectionPopulation} year={filters.periodo || "2026"} loading={directionHistory.loading} error={directionHistory.error} />
                                </div>
                              </div>

                              <div className="mt-7">
                                <p className="text-sm font-semibold">Comentarios asociados</p>
                                <p className="mt-2 text-xs text-muted-foreground">Análisis cualitativo con SIA · En construcción</p>
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
                              <div className="flex flex-wrap items-start justify-between gap-4">
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
                                  <p className="text-xs text-muted-foreground">Resultado general</p>
                                  <p className="mt-1 text-2xl font-semibold">
                                    {effectiveDirectionDirector.indice?.toFixed(2) ?? "—"}
                                  </p>
                                </div>
                                <div className="rounded-xl border bg-card p-4">
                                  <p className="text-xs text-muted-foreground">Encuestas analizadas</p>
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
                                        <p className="mt-1 text-xs text-muted-foreground">{population.modalidad_academica} · {population.encuestas} encuestas analizadas</p>
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
                                  {directionDirectorQuestions.map((question) => (<MetricCard key={question.pregunta_id} title={question.texto_corto} current={question.indice} previous={null} responses={question.n} distribution={question} currentYear={filters.periodo || "2026"} />))}
                                </div>
                              </div>

                                  <div className="mt-7">
                                <p className="text-sm font-semibold">Comentarios asociados</p>
                                <p className="mt-2 text-xs text-muted-foreground">Análisis cualitativo con SIA · En construcción</p>
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

            {effectiveSectionId === "ESC_03_SERVICIOS" && (
              <InstitutionalServices
                key={`${filters.periodo}|${filters.modalidad}|${filters.servicio}`}
                periodo={filters.periodo || "2026"}
                modalidad={filters.modalidad || "Escolarizadas"}
                servicio={filters.servicio || ""}
                relatedComments={(service, comments) => commentsForInstitutionalServiceQuestion(
                  { pregunta_id: service.pregunta_id, texto_corto: service.nombre }, comments,
                )}
              />
            )}

            {effectiveSectionId === "ESC_04_ACADEMICOS" && (
              <AcademicServices
                key={`${filters.periodo}|${filters.modalidad}|${filters.servicio}`}
                periodo={filters.periodo || "2026"}
                modalidad={filters.modalidad || "Escolarizadas"}
                servicio={filters.servicio || ""}
              />
            )}

            {isLanguageCenterSection && (
              <LanguageCenter
                key={`${filters.periodo}|${filters.modalidad}|${filters.servicio}`}
                periodo={filters.periodo || "2026"}
                modalidad={filters.modalidad || "Escolarizadas"}
                servicio={filters.servicio || ""}
              />
            )}

            {isContinuingEducationSection && (
              <ContinuingEducation
                key={`${filters.periodo}|${filters.modalidad}|${filters.servicio}`}
                periodo={filters.periodo || "2026"}
                modalidad={filters.modalidad || "Escolarizadas"}
                servicio={filters.servicio || ""}
              />
            )}

            {isFacilitiesSection && (
              <Facilities
                key={`${filters.periodo}|${filters.modalidad}|${filters.servicio}`}
                periodo={filters.periodo || "2026"}
                modalidad={filters.modalidad || "Escolarizadas"}
                servicio={filters.servicio || ""}
              />
            )}

            {isEnvironmentSection && (
              <SchoolEnvironment
                key={`${filters.periodo}|${filters.modalidad}|${filters.servicio}`}
                periodo={filters.periodo || "2026"}
                modalidad={filters.modalidad || "Escolarizadas"}
                servicio={filters.servicio || ""}
              />
            )}

            {isSatisfactionSection && (
              <Satisfaction
                key={`${filters.periodo}|${filters.modalidad}|${filters.servicio}`}
                periodo={filters.periodo || "2026"}
                modalidad={filters.modalidad || "Escolarizadas"}
                servicio={filters.servicio || ""}
              />
            )}

            {effectiveSectionId !== "ESC_02_DIRECCION" &&
              effectiveSectionId !== "ESC_03_SERVICIOS" &&
              effectiveSectionId !== "ESC_04_ACADEMICOS" &&
              effectiveSectionId !== "ESC_05_IDIOMAS" &&
              !isContinuingEducationSection &&
              !isFacilitiesSection &&
              !isEnvironmentSection &&
              !isSatisfactionSection &&
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
              !isEnvironmentSection &&
              !isSatisfactionSection &&
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

    {historyStatus === "error" ? (
      <p className="text-sm text-muted-foreground">No fue posible cargar el comparativo histórico</p>
    ) : historyStatus === "loading" ? (
      <div className="rounded-2xl border bg-card p-8 shadow-sm">
        <p className="sr-only">
          Cargando comparativo histórico...
        </p>
        <MetricSkeleton />
      </div>
    ) : !historico2025?.encuestas || !historico2026?.encuestas ? (
      <p className="text-sm text-muted-foreground">Sin comparable histórico</p>
    ) : (
      <>
        <GlobalSectionSummary title="Índice histórico comparable">
          <div className="flex flex-wrap items-baseline justify-between gap-2"><p className="text-sm text-muted-foreground">2025 → 2026</p><MetricChange delta={historico2025.indice_general !== null && historico2026.indice_general !== null ? historico2026.indice_general - historico2025.indice_general : null} /></div>
          <MetricComparison previous={historico2025.indice_general} current={historico2026.indice_general} />
          <p className="text-xs text-muted-foreground">{Math.min(historico2025.preguntas_con_datos, historico2026.preguntas_con_datos)} reactivos comparables · Reactivos históricos con información disponible</p>
        </GlobalSectionSummary>

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

                return (<article key={seccion2026.seccion_nombre} className="space-y-3 rounded-xl border bg-background p-4"><div className="flex flex-wrap items-baseline justify-between gap-2"><h4 className="font-semibold">{seccion2026.seccion_nombre}</h4><MetricChange delta={diferencia} /></div><p className="text-xs text-muted-foreground">Comparación del índice de la sección</p><MetricComparison previous={seccion2025.indice} current={seccion2026.indice} /></article>)
              })}
          </div>
        </div>
      </>
    )}
  </div>
)}

          {summary && !loading && activeTab === "comentarios" && (
            <div className="mt-8 rounded-xl border bg-card p-4 text-sm text-muted-foreground">
              <p>
                Consulta los comentarios en el detalle de cada responsable,
                carrera, servicio, criterio o indicador bajo los filtros activos.
              </p>
            </div>
          )}


        </div>
      </main>

    </div>
  )
}
