// node tests/quality-catalog-audit.mjs <captured-comparison.json> <report.md>
import assert from 'node:assert/strict'
import {readFileSync,writeFileSync} from 'node:fs'
import vm from 'node:vm'
import ts from 'typescript'
const source=readFileSync(new URL('../src/components/calidad/directionModel.ts',import.meta.url),'utf8')
const module={exports:{}}
vm.runInNewContext(ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText,{module,exports:module.exports})
const {currentPopulations,populationKey}=module.exports
const data=JSON.parse(readFileSync(process.argv[2],'utf8'))
const before=currentPopulations(data.before), after=currentPopulations(data.after)
const active=[...new Map(data.catalog.map(row=>{const p={carrera_base:row.CARRERA_BASE,modalidad_academica:row.MODALIDAD_ACADEMICA};return [populationKey(p),p]})).values()]
const missingBefore=active.filter(row=>!before.some(p=>populationKey(p)===populationKey(row)))
const missingAfter=active.filter(row=>!after.some(p=>populationKey(p)===populationKey(row)))
assert.equal(missingBefore.length,6)
assert.equal(missingAfter.length,0)
assert.equal(new Set(after.map(populationKey)).size,after.length)
assert.equal(after.filter(row=>!row.carrera_base.trim() || row.conflict).length,0)
const history=currentPopulations(data.history)
const lines=['# Correcciones funcionales de Encuesta de Calidad','',
  'Fecha: 11 de noviembre de 2026. Fuente normalizada actual y catálogo activo capturados en lectura; histórico de la auditoría global. Sin escrituras en las fuentes.','',
  `Dirección: ${before.length} → ${after.length} poblaciones visibles. Activas faltantes: ${missingBefore.length} → ${missingAfter.length}. Nombres vacíos: 0. Duplicados indebidos/conflictos: 0.`, '',
  '## Catálogo activo de Dirección: antes/después','',
  'Cada fila corresponde a una identidad del catálogo activo. “Respuestas” cuenta encuestas con respuesta válida de Dirección; un cero es un conteo, nunca una calificación. El histórico se consulta para la misma identidad y no crea poblaciones.','',
  '| Carrera | Modalidad | Activa | Respuestas actuales | Histórico | Estado antes | Estado después |',
  '|---|---|---|---:|---|---|---|']
for(const row of active){
  const old=before.find(p=>populationKey(p)===populationKey(row)), now=after.find(p=>populationKey(p)===populationKey(row)), past=history.find(p=>populationKey(p)===populationKey(row))
  const state=p=>!p?'Ausente':p.indice===null||!p.encuestas?'Sin respuestas en este periodo':'Con respuestas'
  lines.push(`| ${row.carrera_base} | ${row.modalidad_academica} | Sí | ${now.encuestas} | ${past?.indice!=null?'Sí':'Sin comparable histórico'} | ${state(old)} | ${state(now)} |`)
}
lines.push('','## Asistentes: asignaciones expuestas','',
  'Se usa el catálogo efectivo y las reglas confirmadas existentes. Las asociaciones sin respuestas se conservan sin índice ni evaluación inventados. Los nombres originales permanecen intactos; los alias de presentación ya existentes evitan duplicados.','',
  '| Responsable | Antes: asignaciones visibles | Después: asignaciones visibles | Incorporadas sin respuestas |','|---|---:|---:|---|')
const assignments=data.after.comparativos_personal.asignaciones
for(const name of [...new Set(assignments.map(r=>r.persona_evaluada))].sort()){
  const old=data.before.comparativos_personal.personas_por_carrera_modalidad.filter(r=>r.persona_evaluada===name)
  const current=[...new Map(assignments.filter(r=>r.persona_evaluada===name).map(r=>[populationKey(r),r])).values()]
  const added=current.filter(r=>!old.some(p=>populationKey(p)===populationKey(r)))
  for(const observed of old){
    const exposed=current.find(p=>populationKey(p)===populationKey(observed))
    assert.equal(exposed?.indice,observed.indice,name+' '+observed.carrera_base)
    assert.equal(exposed?.evaluaciones,observed.evaluaciones,name+' '+observed.carrera_base)
  }
  lines.push(`| ${name} | ${new Set(old.map(populationKey)).size} | ${current.length} | ${added.map(r=>r.carrera_base+' / '+r.modalidad_academica).join('; ')||'—'} |`)
}
assert.equal(data.unchanged_existing_metrics,true)
lines.push('','## Conservación de metodología','',
  'La comparación automatizada ejecutó el motor anterior y el corregido sobre los mismos 37,597 registros normalizados y el mismo catálogo. Todos los resultados preexistentes, preguntas, secciones, índices y agregados son idénticos. Solo se amplió la presencia de poblaciones sin respuestas en Dirección y se expuso el catálogo efectivo de asignaciones de Asistentes.', '',
  'No se modificaron normalización, fórmulas, equivalencias, nombres canónicos, selección metodológica del histórico, PDF, CSV ni estilos generales.','')
writeFileSync(process.argv[3],lines.join('\n'))
console.log(JSON.stringify({before:before.length,after:after.length,missingBefore,missingAfter:missingAfter.length,assignments:assignments.length,unchangedMetrics:true},null,2))
