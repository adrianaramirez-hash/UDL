import assert from 'node:assert/strict'
import {readFileSync} from 'node:fs'
import vm from 'node:vm'
import test from 'node:test'
import ts from 'typescript'

function harness(response, labels=[]) {
  const events=[],requests=[],module={exports:{}}
  const anchor={href:'',download:'',click(){events.push('download')},remove(){events.push('remove')}}
  const context={module,exports:module.exports,URLSearchParams,Error,
    URL:{createObjectURL(){events.push('blob');return 'blob:report'},revokeObjectURL(){events.push('revoke')}},
    document:{querySelectorAll(){return labels},createElement(tag){assert.equal(tag,'a');return anchor},body:{appendChild(){events.push('append')}}},
    fetch:async url=>{requests.push(new URL(url, 'http://localhost'));return response},setTimeout(callback){callback()},
  }
  const source=readFileSync(new URL('../src/services/resultsReportPdf.ts',import.meta.url),'utf8')
  vm.runInNewContext(ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText,context)
  return {download:module.exports.downloadResultsReportPdf,events,requests,anchor}
}

test('PDF conserva la población y descarga el nombre indicado por el servidor',async()=>{
  const h=harness({ok:true,headers:new Headers({'Content-Type':'application/pdf','Content-Disposition':'attachment; filename="Informe_de_resultados_Derecho_Presencial.pdf"'}),blob:async()=>new Blob(['%PDF-1.4'],{type:'application/pdf'})})
  const scope={periodo:'2026',modalidad:'Escolarizadas',servicio:'Programa activo',carrera:'Derecho',modalidad_academica:'Presencial'}
  await h.download(scope)
  assert.equal(h.requests.length,1)
  assert.equal(h.requests[0].pathname,'/api/encuesta-calidad/informe-resultados.pdf')
  for(const [key,value] of Object.entries(scope))assert.equal(h.requests[0].searchParams.get(key),value)
  assert.equal(h.anchor.download,'Informe_de_resultados_Derecho_Presencial.pdf')
  assert.deepEqual(h.events,['blob','append','download','remove','revoke'])
})
test('PDF con fallo de fuente no descarga un archivo ni vuelve a consultar global',async()=>{
  const h=harness({ok:false,json:async()=>({detail:'Fuente temporalmente no disponible'})})
  await assert.rejects(h.download({carrera:'Derecho',modalidad_academica:'Presencial'}),/Fuente temporalmente/)
  assert.equal(h.requests.length,1)
  assert.equal(h.requests[0].searchParams.get('carrera'),'Derecho')
  assert.equal(h.events.length,0)
})
test('PDF rechaza una respuesta que no sea PDF aunque el servidor responda 200',async()=>{
  const h=harness({ok:true,headers:new Headers({'Content-Type':'application/json'})})
  await assert.rejects(h.download({periodo:'2026'}),/no es un PDF/)
  assert.equal(h.events.length,0)
})
test('PDF lee carrera y modalidad vigentes sin modificar selectores ni el filtro superior',async()=>{
  const labels=[['Carrera / programa','Derecho'],['Modalidad','Presencial'],['Idioma','Inglés']].map(([textContent,value])=>({textContent,querySelector(){return Object.freeze({value})}}))
  const h=harness({ok:true,headers:new Headers({'Content-Type':'application/pdf'}),blob:async()=>new Blob(['%PDF-1.4'])},labels)
  const scope=Object.freeze({periodo:'2026',modalidad:'Escolarizadas'})
  await h.download(scope,true)
  assert.equal(h.requests[0].searchParams.get('carrera'),'Derecho')
  assert.equal(h.requests[0].searchParams.get('modalidad_academica'),'Presencial')
  assert.equal(h.requests[0].searchParams.get('modalidad'),'Escolarizadas')
  assert.equal(h.requests[0].searchParams.get('idioma'),'Inglés')
  assert.equal(Object.keys(scope).length,2)
})
