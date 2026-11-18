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
const { InstitutionalServicesView } = load(path.join(root, "src/components/calidad/InstitutionalServices.tsx"))
const { fetchInstitutionalServices } = load(path.join(root, "src/services/institutionalServices.ts"))
const service = (id, extra = {}) => ({ id, pregunta_id: `ESC_${id}`, nombre: id, indice: 75, n: 10, favorable_pct: 70, neutral_pct: 20, desfavorable_pct: 10, ...extra })
const current = { indice: 75, respuestas: 20, encuestas_con_evaluacion: 10, distribucion: { favorable_pct: 70, neutral_pct: 20, desfavorable_pct: 10 }, servicios: [service("Admisiones"), service("Becas")], comentarios: [], opciones: { carreras: ["Derecho", "Nutrición"], modalidades: ["Presencial", "Ejecutiva"] }, poblacion: { carrera: null, modalidad_academica: null } }
const props = { current, historical: null, year: "2026", scope: { carrera: "", modalidad_academica: "" }, onScopeChange() {}, relatedComments() { return [] } }
const render = extra => renderToStaticMarkup(React.createElement(InstitutionalServicesView, { ...props, ...extra }))

test("Global carga sin carrera y conserva todos los servicios actuales", () => {
  const html = render()
  assert.match(html, /Todas \/ Vista global/)
  assert.match(html, /Admisiones/)
  assert.match(html, /Becas/)
  assert.match(html, /Respuestas a criterios/)
  assert.match(html, /Encuestas analizadas/)
  assert.doesNotMatch(html, /Selecciona una carrera|Respuestas abiertas por sección/)
})

test("Servicio sin histórico mantiene su resultado y distingue ausencia de respuestas", () => {
  const html = render({ current: { ...current, servicios: [service("Sin histórico"), service("Sin respuestas", { indice: null, n: 0 })] } })
  assert.match(html, /Sin comparable histórico/)
  assert.match(html, /Sin respuestas en este periodo/)
  assert.match(html, /75\.00/)
  assert.match(html, /Resultado actual: <strong>—<\/strong>/)
})

test("Histórico no crea servicios; comparación usa barras 0–100 y delta junto al nombre", () => {
  const history = { indice_2025: 70, indice_2026: 75, criterios: ["Admisiones"], servicios: [{ id: "Admisiones", indice_2025: 78.24, indice_2026: 75 }, { id: "Servicio solo histórico", indice_2025: 90, indice_2026: null }] }
  const html = render({ historical: history })
  assert.doesNotMatch(html, /Servicio solo histórico/)
  assert.match(html, /Admisiones<\/h4><span[^>]*>↓ -3\.24 pts/)
  assert.match(html, /width:78\.24%/)
  assert.match(html, /aria-valuemin="0" aria-valuemax="100"/)
  assert.equal((html.match(/Ver detalle de/g) ?? []).length, 2)
})

test("Error histórico no bloquea datos actuales y se informa una sola vez", () => {
  const html = render({ historyError: true })
  assert.match(html, /75\.00/)
  assert.match(html, /Ver detalle de Admisiones/)
  assert.equal((html.match(/No fue posible cargar el comparativo histórico/g) ?? []).length, 1)
  assert.doesNotMatch(html, /Sin comparable histórico/)
})

test("Carga histórica muestra skeleton y conserva resultados actuales", () => {
  const html = render({ loadingHistory: true })
  assert.match(html, /Cargando comparativo/)
  assert.match(html, /75\.00/)
  assert.doesNotMatch(html, /Sin comparable histórico/)
})

test("El cliente envía exactamente carrera, modalidad y programa en ambos años", async () => {
  const original = globalThis.fetch
  const urls = []
  globalThis.fetch = async url => { urls.push(new URL(url, 'http://localhost')); return { ok: true, json: async () => ({}) } }
  try {
    const scope = { periodo: "2026", modalidad: "Escolarizadas", servicio: "Licenciatura: Derecho", carrera: "Derecho", modalidad_academica: "Presencial" }
    await fetchInstitutionalServices("actual", scope)
    await fetchInstitutionalServices("historico", scope)
    for (const key of ["modalidad", "servicio", "carrera", "modalidad_academica"]) {
      assert.equal(urls[0].searchParams.get(key), scope[key])
      assert.equal(urls[1].searchParams.get(key), scope[key])
    }
    assert.equal(urls[0].searchParams.get("periodo"), "2026")
    assert.equal(urls[1].searchParams.has("periodo"), false)
  } finally { globalThis.fetch = original }
})
