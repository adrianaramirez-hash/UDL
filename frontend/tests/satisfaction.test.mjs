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
const { SatisfactionView } = load(path.join(root, "src/components/calidad/Satisfaction.tsx"))
const { fetchSatisfaction } = load(path.join(root, "src/services/satisfaction.ts"))
const current = {
  satisfaccion: { id: "SAT_GENERAL", nombre: "Satisfacción general", indice: 75, n: 4, favorable_pct: 75, neutral_pct: 25, desfavorable_pct: 0 },
  recomendacion: { id: "REC_UDL", nombre: "Recomendación", si: 3, no: 1, total_valido: 4, si_pct: 75, no_pct: 25 },
  encuestas_analizadas: 4, comentarios: [{ texto: "Motivo solo del detalle", texto_informativo: true }],
  opciones: { carreras: ["Derecho", "Nutrición"], modalidades: ["Presencial", "Ejecutiva"] },
}
const historical = {
  satisfaccion: { id: "SAT_GENERAL", comparable: true, indice_2025: 50, indice_2026: 75, delta: 25 },
  recomendacion: { id: "REC_UDL", comparable: true, porcentaje_2025: 90, porcentaje_2026: 75, delta_pp: -15 },
}
const props = { current, historical, year: "2026", scope: { carrera: "", modalidad_academica: "" }, onScopeChange() {} }
const render = extra => renderToStaticMarkup(React.createElement(SatisfactionView, { ...props, ...extra }))

test("Global muestra dos indicadores independientes, encuestas y filtros opcionales", () => {
  const html = render()
  assert.equal((html.match(/<article\b/g) ?? []).length, 2)
  assert.equal((html.match(/Respuestas a criterios:/g) ?? []).length, 2)
  for (const label of ["Satisfacción general", "Recomendación", "Encuestas analizadas", "Todas / Vista global"]) assert.ok(html.includes(label))
  assert.doesNotMatch(html, /Resultado general|NPS|promotores|pasivos|detractores|Selecciona una carrera/)
})

test("Históricos independientes usan puntos de índice y puntos porcentuales", () => {
  const cards = render().match(/<article\b[\s\S]*?<\/article>/g)
  assert.match(cards[0], /↑ \+25\.00 pts/)
  assert.match(cards[0], /aria-valuenow="50"/)
  assert.match(cards[0], /Favorable 75\.0%/)
  assert.doesNotMatch(cards[0], /90\.00|puntos porcentuales/)
  assert.match(cards[1], /↓ −15\.00 pp/)
  assert.match(cards[1], /title="puntos porcentuales"/)
  assert.match(cards[1], /aria-label="2025" aria-valuemin="0" aria-valuemax="100" aria-valuenow="90"/)
  assert.match(cards[1], /aria-label="2026" aria-valuemin="0" aria-valuemax="100" aria-valuenow="75"/)
  assert.match(cards[1], /Sí: 75\.00% \(3\)/)
  assert.match(cards[1], /No: 25\.00% \(1\)/)
  assert.doesNotMatch(cards[1], /pts|Favorable|Neutral|Desfavorable/)
})

test("Sin histórico conserva ambos resultados actuales", () => {
  const html = render({ historical: null })
  assert.equal((html.match(/Sin comparable histórico/g) ?? []).length, 2)
  assert.match(html, /75\.00%/)
  assert.match(html, />75\.00<\/p>/)
})

test("Sin respuestas no muestra cero; cero válido en ambos indicadores se conserva", () => {
  const empty = { ...current, satisfaccion: { ...current.satisfaccion, indice: null, n: 0 },
    recomendacion: { ...current.recomendacion, si: 0, no: 0, total_valido: 0, si_pct: null, no_pct: null } }
  const html = render({ current: empty, historical: null })
  assert.equal((html.match(/Sin respuestas en este periodo/g) ?? []).length, 2)
  assert.doesNotMatch(html, /0\.00%/)
  const zero = render({ current: { ...current, satisfaccion: { ...current.satisfaccion, indice: 0 },
    recomendacion: { ...current.recomendacion, si: 0, no: 4, si_pct: 0, no_pct: 100 } }, historical: null })
  assert.match(zero, /Sí: 0\.00%/)
  assert.match(zero, />0\.00<\/p>/)
})

test("Fallo o carga histórica conserva los datos actuales sin afirmar ausencia de comparable", () => {
  for (const extra of [{ historyError: true }, { loadingHistory: true }]) {
    const html = render(extra)
    assert.match(html, /75\.00%/)
    assert.doesNotMatch(html, /Sin comparable histórico/)
    assert.match(html, extra.historyError ? /No fue posible cargar el comparativo histórico/ : /Cargando comparativo/)
  }
})

test("Motivos no aparecen en la vista global ni se clasifican", () => {
  assert.doesNotMatch(render(), /Motivo solo del detalle|Análisis cualitativo|sentimiento|Conclusiones automáticas/)
})

test("El cliente envía exactamente la misma población a actual e histórico", async () => {
  const original = globalThis.fetch, urls = []
  globalThis.fetch = async url => { urls.push(new URL(url, 'http://localhost')); return { ok: true, json: async () => ({}) } }
  try {
    const scope = { periodo: "2026", modalidad: "Escolarizadas", servicio: "Programa vigente", carrera: "Derecho", modalidad_academica: "Presencial" }
    await fetchSatisfaction("actual", scope)
    await fetchSatisfaction("historico", scope)
    for (const key of ["modalidad", "servicio", "carrera", "modalidad_academica"]) assert.equal(urls[0].searchParams.get(key), urls[1].searchParams.get(key))
    assert.equal(urls[0].searchParams.get("periodo"), "2026")
    assert.equal(urls[1].searchParams.has("periodo"), false)
  } finally { globalThis.fetch = original }
})
