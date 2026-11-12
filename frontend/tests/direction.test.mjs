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
const model = load(path.join(root, "src/components/calidad/directionModel.ts"))
const components = load(path.join(root, "src/components/calidad/DirectionResults.tsx"))
const metrics = load(path.join(root, "src/components/common/CalidadMetrics.tsx"))
const row = (name, extra = {}) => ({ carrera_base: name, modalidad_academica: "Presencial", indice: null, encuestas: 0, preguntas_con_datos: 0, estado: "sin_encuestas", ...extra })
const question = (id, extra = {}) => ({ pregunta_id: id, texto_corto: "Disponibilidad", indice: 76.12, n: 10, favorable_pct: 60, neutral_pct: 20, desfavorable_pct: 20, ...extra })
const data = (rows = [], questions = [], global = []) => ({ comparativos_direccion: { por_carrera_modalidad: rows, preguntas_por_carrera_modalidad: questions }, preguntas: global, secciones: [{ seccion_id: "ESC_02_DIRECCION", indice: 76.12 }] })
const render = (component, props) => renderToStaticMarkup(React.createElement(component, props))

test("Global muestra índice y cuatro criterios sin elegir carrera", () => {
  const html = render(components.DirectionGlobal, { current: data([], [], [question("ESC_DIR_01", { seccion_id: "ESC_02_DIRECCION" })]), historical: null, year: "2026", loading: false, error: false, filtered: false })
  assert.match(html, /Todas \/ Vista global/)
  assert.match(html, /76\.12/)
  for (const label of ["Disponibilidad", "Atención a necesidades", "Solución de problemas", "Seguimiento / desempeño docente"]) assert.ok(html.includes(label))
})

test("Carrera sin respuestas permanece visible y permite consultar cuatro criterios", () => {
  const current = data([row("Gastronomía")])
  const html = render(components.DirectionCareerList, { current, historical: null, year: "2026", onSelect() {} })
  assert.match(html, /Gastronomía/)
  assert.match(html, /Sin respuestas en este periodo/)
  assert.equal(model.criteriaFor(current, row("Gastronomía")).length, 4)
})

test("Carrera con resultados sin histórico no desaparece", () => {
  const current = data([row("Pedagogía", { indice: 50, encuestas: 1, estado: "con_datos" })])
  const html = render(components.DirectionCareerList, { current, historical: data(), year: "2026", onSelect() {} })
  assert.match(html, /Pedagogía/)
  assert.match(html, /50\.00/)
  assert.match(html, /Sin comparable histórico/)
  assert.doesNotMatch(html, /Sin respuestas en este periodo/)
})

test("Histórico no agrega carreras ni duplica alias del catálogo actual", () => {
  const current = data([row("Cine y TV Digital", { indice: 76.12, encuestas: 10 }), row(" Cine y Televisión Digital ")])
  const history = data([row("Cine y Televisión Digital", { indice: 78.24, encuestas: 10 }), row("Carrera solo histórica", { indice: 90, encuestas: 1 })])
  const html = render(components.DirectionCareerList, { current, historical: history, year: "2026", onSelect() {} })
  assert.equal(model.currentPopulations(current).length, 1)
  assert.doesNotMatch(html, /Carrera solo histórica/)
  assert.match(html, /↓ -2\.12 pts/)
})

test("Nombres vacíos e invisibles no se renderizan", () => {
  assert.equal(model.currentPopulations(data([row(""), row("  "), row("\u200b\u2060"), row("Derecho")])).length, 1)
})

test("Una carrera con datos muestra sus cuatro resultados y distribuciones", () => {
  const population = row("Diseño Gráfico", { indice: 76.12, encuestas: 10 })
  const current = data([population], model.directionCriteria.map(([id]) => ({ ...question(id), ...population })))
  const html = render(components.DirectionCriteria, { current, historical: null, population, year: "2026" })
  assert.equal((html.match(/role="meter"/g) ?? []).length, 4)
  assert.match(html, /Favorable 60\.0%/)
})

test("Carrera futura recibe resultados al cambiar la fuente, sin cambios de código", () => {
  const population = row("Programa futuro")
  assert.equal(model.directionAudit(data([population]), null)[0].estado, "Sin respuestas en este periodo")
  const updated = row("Programa futuro", { indice: 82, encuestas: 3, estado: "con_datos" })
  const audit = model.directionAudit(data([updated]), null)
  assert.equal(audit[0].estado, "Con resultados")
  assert.equal(audit[0].respuestas_actuales, 3)
})

test("Barras mantienen escala 0–100 y delta junto al título", () => {
  const html = render(metrics.MetricCard, { title: "Disponibilidad", previous: 78.24, current: 76.12, responses: 10 })
  assert.match(html, /width:78\.24%/)
  assert.match(html, /width:76\.12%/)
  assert.match(html, /aria-valuemin="0" aria-valuemax="100"/)
  assert.match(html, /Disponibilidad<\/h4><span[^>]*>↓ -2\.12 pts/)
  assert.match(render(metrics.MetricDelta, { previous: 50, current: 50 }), /→ estable/)
  assert.match(render(metrics.MetricDelta, { previous: 50, current: 50.68 }), /↑ \+0\.68 pts/)
})

test("Un cero válido se muestra como resultado; histórico no suplanta resultado ausente", () => {
  const zero = render(metrics.MetricComparison, { previous: 20, current: 0 })
  assert.match(zero, /0\.00/)
  assert.doesNotMatch(zero, /Sin respuestas/)
  const absent = render(metrics.MetricComparison, { previous: 78.24, current: null })
  assert.match(absent, /Sin respuestas en este periodo/)
  assert.match(absent, /78\.24/)
})

test("Modalidades distintas siguen separadas, y métricas ambiguas no se promedian", () => {
  const first = row("Derecho", { indice: 70, encuestas: 5 })
  const executive = row("Derecho", { modalidad_academica: "Ejecutiva", indice: 60, encuestas: 3 })
  assert.equal(model.currentPopulations(data([first, executive])).length, 2)
  const ambiguous = model.currentPopulations(data([first, { ...first, carrera_base: " DERECHO " }]))[0]
  assert.equal(ambiguous.conflict, true)
  assert.equal(ambiguous.indice, null)
})

test("Solo se comparan IDs de criterios estables; no hay coincidencia por texto aproximado", () => {
  const current = data([], [], [question("OTRO_ID", { seccion_id: "ESC_02_DIRECCION" })])
  assert.equal(model.criteriaFor(current)[0].indice, null)
  const history = data([], [], [question("ESC25_DIR_01", { seccion_id: "ESC25_02_DIRECCION", indice: 69.24 })])
  assert.equal(model.criteriaFor(history)[0].indice, 69.24)
})

test("Carga y error histórico mantienen resultados actuales sin afirmar ausencia de comparable", () => {
  const props = { title: "Disponibilidad", previous: null, current: 76.12, responses: 10 }
  const pending = render(metrics.MetricCard, { ...props, loading: true })
  assert.match(pending, /76\.12/)
  assert.match(pending, /Cargando comparativo/)
  assert.doesNotMatch(pending, /Sin comparable histórico/)
  const error = render(metrics.MetricCard, { ...props, comparisonError: true })
  assert.match(error, /No fue posible cargar el comparativo histórico/)
  assert.doesNotMatch(error, /Sin comparable histórico/)
})
