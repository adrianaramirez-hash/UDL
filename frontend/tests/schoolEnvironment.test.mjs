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
const { SchoolEnvironmentView } = load(path.join(root, "src/components/calidad/SchoolEnvironment.tsx"))
const { fetchSchoolEnvironment } = load(path.join(root, "src/services/schoolEnvironment.ts"))
const names = ["Relaci\u00f3n con profesores / asesores", "Relaci\u00f3n con compa\u00f1eros", "Ambiente escolar en mi servicio"]
const current = { incidencia: { si: 2, no: 8, total_valido: 10, si_pct: 20, no_pct: 80 }, indice: 75, respuestas: 30, encuestas_analizadas: 10,
  distribucion: { favorable_pct: 70, neutral_pct: 20, desfavorable_pct: 10 },
  criterios: names.map((nombre, i) => ({ id: `AMB_${i}`, pregunta_id: `ESC_AMB_${i}`, nombre, indice: 75, n: 10, encuestas_validas: 10, favorable_pct: 70, neutral_pct: 20, desfavorable_pct: 10 })),
  comentarios: [{ incluir_resumen: true, texto_informativo: true, texto: "Comentario exclusivo del detalle" }],
  opciones: { carreras: ["Derecho", "Nutrición"], modalidades: ["Presencial", "Ejecutiva"] }, poblacion: { carrera: null, modalidad_academica: null } }
const props = { current, historical: null, year: "2026", scope: { carrera: "", modalidad_academica: "" }, onScopeChange() {} }
const render = extra => renderToStaticMarkup(React.createElement(SchoolEnvironmentView, { ...props, ...extra }))

test("Vista global sin carrera muestra los tres contadores y exactamente tres criterios", () => {
  const html = render()
  for (const label of ["Resultado general", "Respuestas a criterios", "Encuestas analizadas", "Todas / Vista global"]) assert.ok(html.includes(label))
  assert.equal((html.match(/Ver detalle de/g) ?? []).length, 3)
  assert.doesNotMatch(html, /Selecciona una carrera|evaluaciones/)
  assert.match(html, /Una encuesta puede aportar respuestas a varios criterios de ambiente escolar/)
})

test("Se conservan por separado los tres criterios sin criterios duplicados", () => {
  const html = render()
  for (const name of names) assert.equal((html.match(new RegExp(`<h4[^>]*>${name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}<\\/h4>`, "g")) ?? []).length, 1)
})

test("Sin respuestas no se muestra como cero; sin histórico conserva el resultado actual", () => {
  const html = render({ current: { ...current, criterios: current.criterios.map((row, i) => i === 2 ? { ...row, indice: null, n: 0 } : row) } })
  assert.match(html, /Sin respuestas en este periodo/)
  assert.match(html, /Sin comparable histórico/)
  assert.match(html, /Resultado actual: <strong>—<\/strong>/)
  assert.match(html, /75\.00/)
})

test("Comparables muestran barras a escala real y delta junto al nombre", () => {
  const historical = { indice_2025: 70, indice_2026: 75, ids_comparables: ["AMB_0"], criterios: [{ id: "AMB_0", indice_2025: 78.24, indice_2026: 75 }] }
  const html = render({ historical })
  assert.match(html, /Relación con profesores \/ asesores<\/h4><span[^>]*>↓ -3\.24 pts/)
  assert.match(html, /width:78\.24%/)
  assert.match(html, /aria-valuemin="0" aria-valuemax="100"/)
  assert.equal((html.match(/Ver detalle de/g) ?? []).length, 3)
})

test("Error histórico informa una sola vez y mantiene visibles los tres resultados", () => {
  const html = render({ historyError: true })
  assert.equal((html.match(/No fue posible cargar el comparativo histórico/g) ?? []).length, 1)
  assert.equal((html.match(/Ver detalle de/g) ?? []).length, 3)
  assert.match(html, /75\.00/)
  assert.doesNotMatch(html, /Sin comparable histórico/)
})

test("Carga histórica usa skeleton sin ocultar el resultado actual", () => {
  const html = render({ loadingHistory: true })
  assert.match(html, /Cargando comparativo/)
  assert.match(html, /75\.00/)
})

test("Comentarios permanecen fuera de la vista global", () => {
  const html = render()
  assert.doesNotMatch(html, /Comentario exclusivo del detalle|Respuestas abiertas por sección|Positivos|Incidencias/)
})

test("El cliente mantiene los mismos filtros de carrera y modalidad en ambos años", async () => {
  const original = globalThis.fetch
  const urls = []
  globalThis.fetch = async url => { urls.push(new URL(url)); return { ok: true, json: async () => ({}) } }
  try {
    const scope = { periodo: "2026", modalidad: "Escolarizadas", servicio: "Programa vigente", carrera: "Derecho", modalidad_academica: "Presencial" }
    await fetchSchoolEnvironment("actual", scope)
    await fetchSchoolEnvironment("historico", scope)
    for (const key of ["modalidad", "servicio", "carrera", "modalidad_academica"]) {
      assert.equal(urls[0].searchParams.get(key), scope[key])
      assert.equal(urls[1].searchParams.get(key), scope[key])
    }
    assert.equal(urls[0].searchParams.get("periodo"), "2026")
    assert.equal(urls[1].searchParams.has("periodo"), false)
  } finally { globalThis.fetch = original }
})

test("Cero válido en ambos años muestra barras y estado estable", () => {
  const historical = { indice_2025: 0, indice_2026: 0, ids_comparables: ["AMB_0"],
    criterios: [{ id: "AMB_0", indice_2025: 0, indice_2026: 0 }] }
  const html = render({ historical, current: { ...current, criterios: current.criterios.map((row, i) => i === 0 ? { ...row, indice: 0 } : row) } })
  const card = html.match(/<article\b[\s\S]*?<\/article>/)[0]
  assert.match(card, /Resultado actual: <strong>0\.00<\/strong>/)
  assert.match(card, /→ estable/)
  assert.equal((card.match(/aria-valuenow="0"/g) ?? []).length, 2)
  assert.doesNotMatch(card, /Sin comparable histórico|Sin respuestas en este periodo/)
})

test("Incidencia aparece separada y su variación usa puntos porcentuales", () => {
  const historical = { indice_2025: 70, indice_2026: 75, ids_comparables: [], criterios: [],
    incidencia: { comparable: true, porcentaje_2025: 30, porcentaje_2026: 20, delta_pp: -10 } }
  const html = render({ historical })
  const incidence = html.match(/<section aria-label="Incidencia reportada"[\s\S]*?<\/section>/)[0]
  assert.match(incidence, /No forma parte del Resultado general/)
  assert.match(incidence, /Sí: 20\.00% \(2 respuestas\)/)
  assert.match(incidence, /No: 80\.00% \(8 respuestas\)/)
  assert.match(incidence, /aria-label="2025" aria-valuemin="0" aria-valuemax="100" aria-valuenow="30"/)
  assert.match(incidence, /aria-label="2026" aria-valuemin="0" aria-valuemax="100" aria-valuenow="20"/)
  assert.match(incidence, /↓ −10\.00 pp/)
  assert.match(incidence, /title="puntos porcentuales"/)
  assert.doesNotMatch(incidence, /pts|Resultado actual|Favorable|Desfavorable/)
  assert.equal((html.match(/<article\b/g) ?? []).length, 3)
})

test("Incidencia sin respuestas no se convierte en 0%, pero cero real se conserva", () => {
  const empty = { si: 0, no: 0, total_valido: 0, si_pct: null, no_pct: null }
  const incidenceHtml = incidence => render({ current: { ...current, incidencia: incidence } }).match(/<section aria-label="Incidencia reportada"[\s\S]*?<\/section>/)[0]
  assert.match(incidenceHtml(empty), /Sin respuestas en este periodo/)
  assert.doesNotMatch(incidenceHtml(empty), /0\.00%/)
  assert.match(incidenceHtml({ si: 0, no: 10, total_valido: 10, si_pct: 0, no_pct: 100 }), /Sí: 0\.00%/)
})
