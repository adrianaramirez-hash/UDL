import assert from "node:assert/strict"
import { readFileSync, existsSync } from "node:fs"
import path from "node:path"
import { createRequire } from "node:module"
import { fileURLToPath } from "node:url"
import vm from "node:vm"
import test from "node:test"
import ts from "typescript"
import React from "react"
import { renderToStaticMarkup } from "react-dom/server"

const root = fileURLToPath(new URL("../", import.meta.url))
const require = createRequire(new URL("../package.json", import.meta.url))
const modules = new Map()
function load(name) {
  const filename = [name, `${name}.ts`, `${name}.tsx`].find(existsSync)
  if (modules.has(filename)) return modules.get(filename).exports
  const module = { exports: {} }
  modules.set(filename, module)
  const code = ts.transpileModule(readFileSync(filename, "utf8"), { compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, target: ts.ScriptTarget.ES2023 } }).outputText
  const localRequire = id => id.startsWith("@/") ? load(path.join(root, "src", id.slice(2))) : id.startsWith(".") ? load(path.resolve(path.dirname(filename), id)) : require(id)
  vm.runInThisContext(`(function(require,module,exports){${code}\n})`, { filename })(localRequire, module, module.exports)
  return module.exports
}
const { ContinuingEducationView } = load(path.join(root, "src/components/calidad/ContinuingEducation.tsx"))
const { fetchContinuingEducation } = load(path.join(root, "src/services/continuingEducation.ts"))
const names = ["Calidad de cursos y diplomados", "Relevancia de los temas para la formación profesional", "Horarios y modalidades", "Recomendación de programas de Educación Continua", "Conocimiento de cursos, talleres y diplomados", "Conocimiento de dónde consultar la oferta académica"]
const ids = ["EC_CALIDAD", "EC_TEMAS", "EC_HORARIOS", "EC_RECOMIENDA", "EC_CONOCE", "EC_SABE_DONDE"]
const binary = { si: 3, no: 1, total_valido: 4, si_pct: 75, no_pct: 25 }
const current = {
  criterios: names.map((nombre, i) => ({ id: ids[i], nombre, resultado: 75, n: 4, encuestas_validas: 4, historico_disponible: false,
    tipo_dato: i < 3 ? "LIKERT" : "BINARIA", unidad: i < 3 ? "indice" : "porcentaje", binaria: i < 3 ? null : binary,
    favorable_pct: 75, neutral_pct: 25, desfavorable_pct: 0 })),
  encuestas_analizadas: 4, respuestas: 24, participacion: { si: 2, no: 2, total_valido: 4, si_pct: 50, no_pct: 50 },
  medios: { n: 2, valores: [{ nombre: "Correo institucional", n: 2, porcentaje: 100 }] },
  comentarios: [{ texto: "Comentario solo del detalle", texto_informativo: true }],
  opciones: { carreras: ["Derecho"], modalidades: ["Presencial"], participacion: ["si", "no"] },
}
const props = { current, year: "2026", scope: { carrera: "", modalidad_academica: "", participacion: "" }, onScopeChange() {} }
const render = extra => renderToStaticMarkup(React.createElement(ContinuingEducationView, { ...props, ...extra }))

test("Vista global muestra seis indicadores independientes y contadores sin exigir carrera", () => {
  const html = render()
  assert.equal((html.match(/<article\b/g) ?? []).length, 6)
  for (const name of names) assert.ok(html.includes(name))
  for (const label of ["Encuestas analizadas", "Respuestas a criterios", "Todas / Vista global"]) assert.ok(html.includes(label))
  assert.doesNotMatch(html, /Resultado general|Evaluaciones|Selecciona una carrera/)
  assert.match(html, /Una encuesta puede aportar respuestas a varios criterios de Educación Continua/)
})

test("Participación y medios son descripciones separadas de los indicadores", () => {
  const html = render()
  const participation = html.match(/<section aria-label="Participación"[\s\S]*?<\/section>/)[0]
  assert.match(participation, /Sí: 50\.00% \(2\)/)
  assert.match(participation, /No: 50\.00% \(2\)/)
  const media = html.match(/<section aria-label="Medios de difusión"[\s\S]*?<\/section>/)[0]
  assert.match(media, /2 respuestas válidas/)
  assert.match(media, /Correo institucional/)
  assert.match(media, /100\.00%/)
  assert.doesNotMatch(media, /Resultado actual|Otro/)
})

test("Recomendación usa distribución Sí/No y porcentaje real sin NPS", () => {
  const card = render().match(/<article\b[\s\S]*?<\/article>/g)[3]
  assert.match(card, /<strong>75\.00%<\/strong>/)
  assert.match(card, /Sí: 75\.00% \(3\)/)
  assert.match(card, /No: 25\.00% \(1\)/)
  assert.match(card, /Recomendaría los programas de Educación Continua a otros estudiantes/)
  assert.doesNotMatch(card, /NPS|promotores|pasivos|detractores|Favorable|Escala 0–100/)
})

test("Seis especializados no muestran barras de 2025, históricos ni deltas", () => {
  const html = render()
  assert.doesNotMatch(html, /Sin comparable histórico/)
  assert.equal((html.match(/aria-label="2026"/g) ?? []).length, 3)
  assert.doesNotMatch(html, /2025|↑|↓|pts|puntos porcentuales/)
})

test("Sin respuestas no se representa como cero y cero válido se conserva", () => {
  const emptyBinary = { si: 0, no: 0, total_valido: 0, si_pct: null, no_pct: null }
  const empty = { ...current, criterios: current.criterios.map(row => ({ ...row, resultado: null, n: 0, binaria: row.binaria ? emptyBinary : null })),
    participacion: emptyBinary, medios: { n: 0, valores: [] } }
  const html = render({ current: empty })
  assert.equal((html.match(/Sin respuestas en este periodo/g) ?? []).length, 8)
  assert.doesNotMatch(html, /0\.00%|<strong>0\.00<\/strong>/)
  const zero = { ...current, criterios: current.criterios.map(row => ({ ...row, resultado: 0, binaria: row.binaria ? { si: 0, no: 4, total_valido: 4, si_pct: 0, no_pct: 100 } : null })) }
  assert.match(render({ current: zero }), /<strong>0\.00%<\/strong>/)
  assert.match(render({ current: zero }), /<strong>0\.00<\/strong>/)
})

test("Comentarios no aparecen ni se clasifican globalmente", () => {
  assert.doesNotMatch(render(), /Comentario solo del detalle|Análisis cualitativo|sentimiento|Fortalezas|Áreas de mejora/)
})

test("El cliente envía filtros opcionales y solicita únicamente la fuente actual", async () => {
  const original = globalThis.fetch, urls = []
  globalThis.fetch = async url => { urls.push(new URL(url, 'http://localhost')); return { ok: true, json: async () => ({}) } }
  try {
    const scope = { periodo: "2026", modalidad: "Escolarizadas", servicio: "Programa", carrera: "Derecho", modalidad_academica: "Presencial", participacion: "si" }
    await fetchContinuingEducation(scope)
    assert.equal(urls.length, 1)
    assert.equal(urls[0].pathname, "/api/encuesta-calidad/educacion-continua/actual")
    for (const [key, value] of Object.entries(scope)) assert.equal(urls[0].searchParams.get(key), value)
  } finally { globalThis.fetch = original }
})
