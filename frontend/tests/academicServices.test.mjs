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
const { AcademicServicesView } = load(path.join(root, "src/components/calidad/AcademicServices.tsx"))
const { fetchAcademicServices } = load(path.join(root, "src/services/academicServices.ts"))
const names = ["Profesores / Asesores", "Plan de estudios", "Clases presenciales", "Clases en Zoom", "Prácticas académicas", "Certificaciones", "Plataforma SEAC en general", "SEAC · Apps móviles", "SEAC · Facilidad para ingresar", "SEAC · Trámites administrativos", "SEAC · Tareas y trabajos", "Contenidos de las Aulas Virtuales"]
const current = { indice: 75, respuestas: 120, encuestas_analizadas: 10,
  distribucion: { favorable_pct: 70, neutral_pct: 20, desfavorable_pct: 10 },
  criterios: names.map((nombre, i) => ({ id: `ACA_${i}`, pregunta_id: `ESC_ACA_${i}`, nombre, indice: 75, n: 10, encuestas_validas: 10, favorable_pct: 70, neutral_pct: 20, desfavorable_pct: 10 })),
  comentarios: [{ incluir_resumen: true, texto_informativo: true, texto: "Comentario exclusivo del detalle" }],
  opciones: { carreras: ["Derecho", "Nutrición"], modalidades: ["Presencial", "Ejecutiva"] }, poblacion: { carrera: null, modalidad_academica: null } }
const props = { current, historical: null, year: "2026", scope: { carrera: "", modalidad_academica: "" }, onScopeChange() {} }
const render = extra => renderToStaticMarkup(React.createElement(AcademicServicesView, { ...props, ...extra }))

test("Vista global sin carrera muestra los tres contadores y exactamente doce criterios", () => {
  const html = render()
  for (const label of ["Resultado general", "Respuestas a criterios", "Encuestas analizadas", "Todas / Vista global"]) assert.ok(html.includes(label))
  assert.equal((html.match(/Ver detalle de/g) ?? []).length, 12)
  assert.doesNotMatch(html, /Selecciona una carrera|evaluaciones/)
  assert.match(html, /Una encuesta puede aportar respuestas a varios criterios académicos/)
})

test("Se conservan por separado los seis conceptos SEAC sin criterios duplicados", () => {
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
  const historical = { indice_2025: 70, indice_2026: 75, ids_comparables: ["ACA_0"], criterios: [{ id: "ACA_0", indice_2025: 78.24, indice_2026: 75 }] }
  const html = render({ historical })
  assert.match(html, /Profesores \/ Asesores<\/h4><span[^>]*>↓ -3\.24 pts/)
  assert.match(html, /width:78\.24%/)
  assert.match(html, /aria-valuemin="0" aria-valuemax="100"/)
  assert.equal((html.match(/Ver detalle de/g) ?? []).length, 12)
})

test("Error histórico informa una sola vez y mantiene visibles los doce resultados", () => {
  const html = render({ historyError: true })
  assert.equal((html.match(/No fue posible cargar el comparativo histórico/g) ?? []).length, 1)
  assert.equal((html.match(/Ver detalle de/g) ?? []).length, 12)
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
    await fetchAcademicServices("actual", scope)
    await fetchAcademicServices("historico", scope)
    for (const key of ["modalidad", "servicio", "carrera", "modalidad_academica"]) {
      assert.equal(urls[0].searchParams.get(key), scope[key])
      assert.equal(urls[1].searchParams.get(key), scope[key])
    }
    assert.equal(urls[0].searchParams.get("periodo"), "2026")
    assert.equal(urls[1].searchParams.has("periodo"), false)
  } finally { globalThis.fetch = original }
})
