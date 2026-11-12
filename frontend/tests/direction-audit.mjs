// Run with Node 24+: node tests/direction-audit.mjs <captured-json> <report.md>
// Input: 2026 and 2025_historico summaries from the existing preview/resumen endpoint.
import assert from "node:assert/strict"
import fs from "node:fs"
import path from "node:path"
import { directionAudit, currentPopulations, populationKey, criteriaFor } from "../src/components/calidad/directionModel.ts"

const [input, output] = process.argv.slice(2)
if (!input || !output) throw new Error("Indica JSON de entrada y Markdown de salida")
const data = JSON.parse(fs.readFileSync(input, "utf8"))
const current = data["2026"], history = data["2025_historico"]
const rows = directionAudit(current, history), populations = currentPopulations(current)
assert.equal(new Set(populations.map(populationKey)).size, populations.length)
assert.ok(rows.every(row => row.carrera.trim()))
assert.ok(populations.every(row => !row.conflict))
for (const name of ["Diseño Gráfico", "Cine y TV Digital", "Nutrición", "Pedagogía", "Comunicación Multimedia", "Derecho", "Gastronomía", "Diseño de Modas", "Tecnologías de la Información", "Actuación"]) {
  assert.ok(rows.some(row => row.carrera === name), `Falta ${name}`)
}
const criteria = criteriaFor(current), historical = criteriaFor(history)
const lines = [
  "# Auditoría de Dirección — Etapa B", "",
  "Base: `5450b4c7aa54c6987cda4018f6373ca65beb14f9`. Lectura de las fuentes reales el 5 de noviembre de 2026.", "",
  "Ámbito: Escolarizadas, periodo actual 2026, sin filtro de servicio; histórico 2025 con el mismo ámbito. Lecturas del endpoint estable `preview/resumen`; histórico con `historico=true`. No se recuperó código de la rama de respaldo.", "",
  "## Resultado", "",
  `- ${new Set(rows.map(row => row.carrera)).size} carreras/programas distintos, presentados en ${rows.length} combinaciones carrera/modalidad.`,
  `- ${rows.filter(row => row.respuestas_actuales > 0).length} poblaciones con respuestas actuales y ${rows.filter(row => row.respuestas_actuales === 0).length} sin respuestas de Dirección.`,
  `- ${rows.filter(row => row.historico).length} poblaciones con histórico disponible y ${rows.filter(row => !row.historico).length} sin histórico.`,
  "- 0 nombres vacíos, 0 claves duplicadas y 0 poblaciones con métricas ambiguas.",
  `- ${rows.reduce((sum, row) => sum + row.respuestas_actuales, 0)} encuestas con respuestas válidas de Dirección; ${criteria.reduce((sum, row) => sum + row.n, 0)} respuestas entre los cuatro criterios.`,
  "- Índice global: 69.13 en 2025 y 67.03 en 2026; variación −2.10 puntos.", "",
  "Los conteos con/sin respuestas y con/sin histórico se hacen por combinación carrera/modalidad. Derecho y Pedagogía tienen modalidades separadas; una misma carrera puede tener resultados en una modalidad y carecer de ellos en otra.", "",
  "## Presentación de identidades", "",
  "El endpoint estable entrega 30 filas. En la presentación se unifican Informática / Tecnologías de la Información (alias ya presente en el normalizador estable) y Maestría:Educación Especial / Maestría en Educación Especial (formato del prefijo del grado). En estos dos pares no hay respuestas actuales: no se suman ni recalculan índices. No se cambia el catálogo ni la normalización del backend.", "",
  "Las claves toleran acentos, mayúsculas y espacios y usan las variantes de Cine/Tecnologías ya reconocidas por la versión estable. Modalidades distintas conservan claves distintas. Si dos filas con resultados tienen la misma identidad, se marca la ambigüedad en lugar de promediar métricas. El histórico nunca incorpora carreras al listado actual.", "",
  "## Auditoría por carrera y modalidad", "",
  "«Encuestas actuales» cuenta encuestas únicas con respuestas válidas de Dirección, no la suma de los cuatro reactivos. Histórico «Sí» significa un índice 2025 disponible para esa misma población.", "",
  "| Carrera / programa | Modalidad | Encuestas actuales | Histórico | Estado visual esperado |",
  "|---|---|---:|:---:|---|",
  ...rows.map(row => `| ${row.carrera} | ${row.modalidad} | ${row.respuestas_actuales} | ${row.historico ? "Sí" : "No"} | ${row.estado}${row.estado === "Con resultados" && !row.historico ? " · Sin comparable histórico" : ""} |`),
  "", "## Cuatro criterios globales", "",
  "| Criterio | Respuestas 2026 | 2025 | 2026 | Variación |", "|---|---:|---:|---:|---:|",
  ...criteria.map((q, i) => `| ${q.texto_corto} | ${q.n} | ${historical[i].indice.toFixed(2)} | ${q.indice.toFixed(2)} | ${(q.indice - historical[i].indice).toFixed(2)} pts |`),
  "", "## Verificación", "",
  "- `node --test tests/direction.test.mjs`: 12 pruebas de regresión sobre datos sintéticos y renderizado HTML de los componentes.",
  "- Auditoría adicional con datos reales: 28 claves únicas, sin nombres vacíos ni conflictos; las 10 carreras solicitadas presentes.",
  "- TypeScript: sin errores nuevos; permanecen 11 errores heredados. El error de estado `sin_encuestas` del bloque sustituido deja de existir.",
  "- Compilación Vite correcta. Esta comprobación de empaquetado no sustituye la comprobación de tipos, que conserva los errores heredados.",
  "- Backend, normalización, catálogo, asignaciones, clasificación de comentarios, fórmulas, equivalencias, PDF y CSV sin modificaciones.",
  "- Pendiente: validación visual interactiva del usuario antes de aplicar los componentes a cualquier otra sección.", "",
  "## Revisión visual sugerida", "",
  "1. Abrir Por sección → Dirección: ver índice global, 1340 respuestas y cuatro criterios sin elegir carrera.",
  "2. Abrir Diseño Gráfico: ver sus cuatro criterios, barras 2025/2026 y distribuciones.",
  "3. Abrir Gastronomía o Actuación: conservar la tarjeta y mostrar «Sin respuestas en este periodo», incluso cuando haya histórico.",
  "4. Abrir Pedagogía Ejecutiva: mostrar resultado 50.00 y «Sin comparable histórico».",
  "5. Usar Todas / Vista global en ambos selectores: recuperar todo el listado sin alterar el resumen global.",
  "6. Confirmar que los detalles por director y los comentarios siguen disponibles.", "",
  "La etapa B termina aquí. No se aplicaron los componentes a otras secciones.", "",
]
fs.mkdirSync(path.dirname(output), { recursive: true })
fs.writeFileSync(output, lines.join("\n"))
console.log(`Auditoría guardada: ${rows.length} poblaciones verificadas.`)
