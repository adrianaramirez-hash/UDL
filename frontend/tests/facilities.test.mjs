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
const { FacilitiesView } = load(path.join(root, "src/components/calidad/Facilities.tsx"))
const { fetchFacilities } = load(path.join(root, "src/services/facilities.ts"))
const names = ["Laboratorio de c\u00f3mputo", "Salones de clase", "Sanitarios", "Equipo did\u00e1ctico HyFlex", "Instalaciones del plantel", "Instalaciones espec\u00edficas para el aprendizaje", "Biblioteca", "Cafeter\u00eda", "Espacios al aire libre", "WiFi para estudiantes"]
const current = { indice: 75, respuestas: 100, encuestas_analizadas: 10,
  distribucion: { favorable_pct: 70, neutral_pct: 20, desfavorable_pct: 10 },
  criterios: names.map((nombre, i) => ({ id: `INS_${i}`, pregunta_id: `ESC_INS_${i}`, nombre, indice: 75, n: 10, encuestas_validas: 10, favorable_pct: 70, neutral_pct: 20, desfavorable_pct: 10 })),
  comentarios: [{ incluir_resumen: true, texto_informativo: true, texto: "Comentario exclusivo del detalle" }],
  opciones: { carreras: ["Derecho", "Nutrición"], modalidades: ["Presencial", "Ejecutiva"] }, poblacion: { carrera: null, modalidad_academica: null } }
const props = { current, historical: null, year: "2026", scope: { carrera: "", modalidad_academica: "" }, onScopeChange() {} }
const render = extra => renderToStaticMarkup(React.createElement(FacilitiesView, { ...props, ...extra }))

test("Vista global sin carrera muestra los tres contadores y exactamente diez criterios", () => {
  const html = render()
  for (const label of ["Resultado general", "Respuestas a criterios", "Encuestas analizadas", "Todas / Vista global"]) assert.ok(html.includes(label))
  assert.equal((html.match(/Ver detalle de/g) ?? []).length, 10)
  assert.doesNotMatch(html, /Selecciona una carrera|evaluaciones/)
  assert.match(html, /Una encuesta puede aportar respuestas a varios criterios de instalaciones/)
})

test("Se conservan por separado los diez criterios sin criterios duplicados", () => {
  const html = render()
  for (const name of names) assert.equal((html.match(new RegExp(`<h4[^>]*>${name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}<\\/h4>`, "g")) ?? []).length, 1)
})

test("Sin respuestas no se muestra como cero; sin histórico conserva el resultado actual", () => {
  const html = render({ current: { ...current, criterios: current.criterios.map((row, i) => i === 3 ? { ...row, indice: null, n: 0 } : row) } })
  assert.match(html, /Sin respuestas en este periodo/)
  assert.match(html, /Sin comparable histórico/)
  assert.match(html, /Resultado actual: <strong>—<\/strong>/)
  assert.match(html, /75\.00/)
})

test("Comparables muestran barras a escala real y delta junto al nombre", () => {
  const historical = { indice_2025: 70, indice_2026: 75, ids_comparables: ["INS_0"], criterios: [{ id: "INS_0", indice_2025: 78.24, indice_2026: 75 }] }
  const html = render({ historical })
  assert.match(html, /Laboratorio de cómputo<\/h4><span[^>]*>↓ -3\.24 pts/)
  assert.match(html, /width:78\.24%/)
  assert.match(html, /aria-valuemin="0" aria-valuemax="100"/)
  assert.equal((html.match(/Ver detalle de/g) ?? []).length, 10)
})

test("Error histórico informa una sola vez y mantiene visibles los diez resultados", () => {
  const html = render({ historyError: true })
  assert.equal((html.match(/No fue posible cargar el comparativo histórico/g) ?? []).length, 1)
  assert.equal((html.match(/Ver detalle de/g) ?? []).length, 10)
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
  globalThis.fetch = async url => { urls.push(new URL(url, 'http://localhost')); return { ok: true, json: async () => ({}) } }
  try {
    const scope = { periodo: "2026", modalidad: "Escolarizadas", servicio: "Programa vigente", carrera: "Derecho", modalidad_academica: "Presencial" }
    await fetchFacilities("actual", scope)
    await fetchFacilities("historico", scope)
    for (const key of ["modalidad", "servicio", "carrera", "modalidad_academica"]) {
      assert.equal(urls[0].searchParams.get(key), scope[key])
      assert.equal(urls[1].searchParams.get(key), scope[key])
    }
    assert.equal(urls[0].searchParams.get("periodo"), "2026")
    assert.equal(urls[1].searchParams.has("periodo"), false)
  } finally { globalThis.fetch = original }
})

test("Biblioteca y Cafetería conservan sus propios históricos aunque cambie el orden", () => {
  const historical = { indice_2025: 60, indice_2026: 75, ids_comparables: ["INS_6", "INS_7"],
    criterios: [{ id: "INS_7", indice_2025: 25, indice_2026: 75 },
                { id: "INS_6", indice_2025: 90, indice_2026: 75 }] }
  const cards = render({ historical }).match(/<article\b[\s\S]*?<\/article>/g)
  const library = cards.find(card => card.includes("Biblioteca</h4>"))
  const cafeteria = cards.find(card => card.includes("Cafetería</h4>"))
  assert.match(library, /aria-valuenow="90"/)
  assert.doesNotMatch(library, /aria-valuenow="25"/)
  assert.match(library, /↓ -15\.00 pts/)
  assert.match(cafeteria, /aria-valuenow="25"/)
  assert.doesNotMatch(cafeteria, /aria-valuenow="90"/)
  assert.match(cafeteria, /↑ \+50\.00 pts/)
})

test("Cero válido en ambos años muestra barras y estado estable", () => {
  const historical = { indice_2025: 0, indice_2026: 0, ids_comparables: ["INS_0"],
    criterios: [{ id: "INS_0", indice_2025: 0, indice_2026: 0 }] }
  const html = render({ historical, current: { ...current, criterios: current.criterios.map((row, i) => i === 0 ? { ...row, indice: 0 } : row) } })
  const card = html.match(/<article\b[\s\S]*?<\/article>/)[0]
  assert.match(card, /Resultado actual: <strong>0\.00<\/strong>/)
  assert.match(card, /→ estable/)
  assert.equal((card.match(/aria-valuenow="0"/g) ?? []).length, 2)
  assert.doesNotMatch(card, /Sin comparable histórico|Sin respuestas en este periodo/)
})
