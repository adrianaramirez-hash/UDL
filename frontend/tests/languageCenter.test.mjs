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
const { LanguageCenterView } = load(path.join(root, "src/components/calidad/LanguageCenter.tsx"))
const { fetchLanguageCenter } = load(path.join(root, "src/services/languageCenter.ts"))
const names = ["¿En general te gusta tu clase?", "¿En general estás satisfecho con tu docente?", "¿En general es satisfactorio para ti el servicio que te ofrecen en la oficina del Centro de Idiomas?"]
const current = {
  criterios: names.map((nombre, i) => ({ id: `IDI_${i}`, nombre, indice: 75, n: 2, encuestas_validas: 2, historico_disponible: false, favorable_pct: 100, neutral_pct: 0, desfavorable_pct: 0 })),
  encuestas_analizadas: 4, respuestas: 6, uso: { si: 2, no: 2, total_valido: 4, si_pct: 50, no_pct: 50 },
  idiomas: { n: 2, valores: [{ nombre: "Inglés", n: 2, porcentaje: 100 }] },
  tipos_curso: { n: 2, valores: [{ nombre: "Intensivo", n: 2, porcentaje: 100 }] },
  profesores: { n: 2, nombres: ["Docente descriptivo"] },
  comentarios: [{ pregunta_id: "IDI_COM", texto: "Comentario solo del detalle", texto_informativo: true }],
  preguntas_comentarios: [{ pregunta_id: "IDI_COM", nombre: "¿Qué es lo que más te gusta?" }],
  opciones: { carreras: ["Derecho"], modalidades: ["Presencial"], idiomas: ["Inglés"], tipos_curso: ["Intensivo"] },
}
const props = { current, year: "2026", scope: { carrera: "", modalidad_academica: "", idioma: "", tipo_curso: "" }, onScopeChange() {} }
const render = extra => renderToStaticMarkup(React.createElement(LanguageCenterView, { ...props, ...extra }))

test("Vista global muestra tres criterios y contexto separado sin exigir filtros", () => {
  const html = render()
  assert.equal((html.match(/<article\b/g) ?? []).length, 3)
  for (const name of names) assert.ok(html.includes(name))
  for (const label of ["Encuestas analizadas", "Respuestas a criterios", "Uso del servicio", "Idiomas", "Tipos de curso", "Todas / Vista global"]) assert.ok(html.includes(label))
  assert.doesNotMatch(html, /Resultado general|Evaluaciones|Selecciona una carrera/)
  assert.match(html, /Una encuesta puede aportar respuestas a varios criterios de Centro de Idiomas/)
})

test("Especializados solo muestran barra actual, sin 2025 ni delta", () => {
  const html = render()
  assert.doesNotMatch(html, /Sin comparable histórico/)
  assert.equal((html.match(/aria-label="2026"/g) ?? []).length, 3)
  assert.doesNotMatch(html, /2025|pts|puntos porcentuales|↑|↓/)
  assert.match(html, /width:75%/)
})

test("Uso, idiomas y curso muestran sus propios denominadores", () => {
  const html = render()
  assert.match(html, /Sí: 50\.00% \(2\)/)
  assert.match(html, /No: 50\.00% \(2\)/)
  const languages = html.match(/<section aria-label="Idiomas"[\s\S]*?<\/section>/)[0]
  assert.match(languages, /2 respuestas válidas/)
  assert.match(languages, /Inglés/)
  assert.match(languages, /100\.00%/)
  assert.doesNotMatch(languages, /Otro/)
})

test("Ausencia de datos no es cero; cero válido se mantiene", () => {
  const empty = { ...current, criterios: current.criterios.map(row => ({ ...row, indice: null, n: 0 })),
    uso: { si: 0, no: 0, total_valido: 0, si_pct: null, no_pct: null }, idiomas: { n: 0, valores: [] }, tipos_curso: { n: 0, valores: [] } }
  const html = render({ current: empty })
  assert.equal((html.match(/Sin respuestas en este periodo/g) ?? []).length, 6)
  assert.doesNotMatch(html, /0\.00%|<strong>0\.00<\/strong>/)
  assert.match(render({ current: { ...current, criterios: current.criterios.map(row => ({ ...row, indice: 0 })) } }), /<strong>0\.00<\/strong>/)
})

test("Profesor y comentarios quedan en detalle, sin clasificaciones ni ranking global", () => {
  assert.doesNotMatch(render(), /Docente descriptivo|Comentario solo del detalle|Análisis cualitativo|ranking|fortaleza|sentimiento/)
})

test("El cliente envía filtros opcionales y no solicita histórico", async () => {
  const original = globalThis.fetch, urls = []
  globalThis.fetch = async url => { urls.push(new URL(url)); return { ok: true, json: async () => ({}) } }
  try {
    const scope = { periodo: "2026", modalidad: "Escolarizadas", servicio: "Programa", carrera: "Derecho", modalidad_academica: "Presencial", idioma: "Inglés", tipo_curso: "Intensivo" }
    await fetchLanguageCenter(scope)
    assert.equal(urls.length, 1)
    assert.equal(urls[0].pathname, "/api/encuesta-calidad/centro-idiomas/actual")
    for (const [key, value] of Object.entries(scope)) assert.equal(urls[0].searchParams.get(key), value)
  } finally { globalThis.fetch = original }
})
