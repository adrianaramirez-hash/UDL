// Presentation keys use stable normalizer/KPI aliases and degree-prefix formatting.
// The source list is always the current endpoint, never the historical endpoint.
export type DirectionMetric = {
  pregunta_id: string; texto_corto: string; indice: number | null; n: number;
  favorable_pct: number | null; neutral_pct: number | null; desfavorable_pct: number | null;
}
export type DirectionPopulation = {
  carrera_base: string; modalidad_academica: string; indice: number | null; encuestas: number;
  preguntas_con_datos: number; estado: string;
}
export type DirectionData = {
  comparativos_direccion: {
    por_carrera_modalidad: DirectionPopulation[];
    preguntas_por_carrera_modalidad: (DirectionMetric & { carrera_base: string; modalidad_academica: string })[];
  };
  preguntas: (DirectionMetric & { seccion_id: string })[];
  secciones: { seccion_id: string; indice: number | null }[];
}

function textKey(value: string) {
  return value.normalize("NFD").replace(/[\u0300-\u036f]/g, "").replace(/\p{Cf}/gu, "").toLowerCase().replace(/\s+/g, " ").trim()
}
export function careerKey(value: string) {
  const key = textKey(value).replace(/^maestria\s*:\s*/, "maestria en ")
  const aliases: Record<string, string> = {
    "cine y television digital": "cine y tv digital",
    "cine television digital": "cine y tv digital",
    "informatica": "tecnologias de la informacion",
    "tecnologias de informacion": "tecnologias de la informacion",
    "tecnologia de la informacion": "tecnologias de la informacion",
  }
  return aliases[key] ?? key
}
export function populationKey(population: Pick<DirectionPopulation, "carrera_base" | "modalidad_academica">) {
  return `${careerKey(population.carrera_base)}|||${textKey(population.modalidad_academica)}`
}
export function criterionId(value: string) { return value.replace(/^ESC25_/, "ESC_") }
export const directionCriteria = [
  ["ESC_DIR_01", "Disponibilidad"],
  ["ESC_DIR_02", "Atención a necesidades"],
  ["ESC_DIR_03", "Solución de problemas"],
  ["ESC_DIR_04", "Seguimiento / desempeño docente"],
] as const

export function currentPopulations(data: DirectionData | null) {
  const groups = new Map<string, DirectionPopulation[]>()
  for (const row of data?.comparativos_direccion.por_carrera_modalidad ?? []) {
    if (!careerKey(row.carrera_base)) continue
    const key = populationKey(row)
    groups.set(key, [...(groups.get(key) ?? []), row])
  }
  return [...groups.values()].map(rows => {
    const withData = rows.filter(row => row.indice !== null && row.encuestas > 0)
    // Do not average separate metrics or invent a merged population.
    const conflict = withData.length > 1
    const row = withData[0] ?? rows[0]!
    const display = rows.find(item => textKey(item.carrera_base) === careerKey(item.carrera_base)) ?? row
    return { ...row, carrera_base: display.carrera_base.trim(), conflict, ...(conflict ? { indice: null } : {}) }
  }).sort((a, b) => a.carrera_base.localeCompare(b.carrera_base, "es") || a.modalidad_academica.localeCompare(b.modalidad_academica, "es"))
}

export function populationMetric(data: DirectionData | null, population: DirectionPopulation) {
  const matches = currentPopulations(data).filter(row => populationKey(row) === populationKey(population))
  return matches.length === 1 && !matches[0]!.conflict ? matches[0]! : null
}

export function criteriaFor(data: DirectionData | null, population?: DirectionPopulation) {
  const questions = population
    ? (data?.comparativos_direccion.preguntas_por_carrera_modalidad ?? []).filter(row => populationKey(row) === populationKey(population))
    : (data?.preguntas ?? []).filter(row => criterionId(row.seccion_id) === "ESC_02_DIRECCION")
  return directionCriteria.map(([id, label]) => {
    const matches = questions.filter(row => criterionId(row.pregunta_id) === id)
    // Ambiguous matches are unavailable, not averaged or matched by fuzzy text.
    const question = matches.length === 1 ? matches[0] : null
    return { pregunta_id: id, texto_corto: question?.texto_corto || label, indice: question?.indice ?? null, n: question?.n ?? 0,
      favorable_pct: question?.favorable_pct ?? null, neutral_pct: question?.neutral_pct ?? null, desfavorable_pct: question?.desfavorable_pct ?? null }
  })
}

export function directionAudit(current: DirectionData, historical: DirectionData | null) {
  return currentPopulations(current).map(row => ({
    carrera: row.carrera_base, modalidad: row.modalidad_academica, respuestas_actuales: row.encuestas,
    historico: populationMetric(historical, row)?.indice !== null && populationMetric(historical, row)?.indice !== undefined,
    estado: row.conflict ? "Identidad ambigua: revisar origen" : row.indice === null || row.encuestas === 0 ? "Sin respuestas en este periodo" : "Con resultados",
  }))
}
