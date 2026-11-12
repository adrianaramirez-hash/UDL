import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import vm from 'node:vm'
import test from 'node:test'
import ts from 'typescript'

// Exercise the production effects, including cleanup, with controlled transport.
const source = readFileSync(new URL('../src/pages/EncuestaCalidad.tsx', import.meta.url), 'utf8')
const ast = ts.createSourceFile('page.tsx', source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)
const page = ast.statements.find(node => ts.isFunctionDeclaration(node) && node.name?.text === 'EncuestaCalidad')
const effects = page.body.statements.filter(node => ts.isExpressionStatement(node) && ts.isCallExpression(node.expression) && node.expression.expression.getText(ast) === 'useEffect')
const deferred = () => { let resolve, reject; const promise = new Promise((a,b) => {resolve=a;reject=b}); return {promise,resolve,reject} }
const settle = () => new Promise(resolve => setImmediate(resolve))
function harness() {
  const state = {}, calls = [], pending = []
  const context = { AbortController, URLSearchParams, console: {error(){}}, filters: {},
    getCalidadSummary(filters, signal) { const task=deferred(); pending.push(task); calls.push({filters,signal});return task.promise },
    fetch(url, options) { const task=deferred(); pending.push(task);calls.push({url,signal:options.signal});return task.promise },
  }
  for (const name of ['Summary','Error','Loading','Historico2025','Historico2026','HistoryStatus']) context['set'+name]=value=>{state[name]=value}
  function run(index, filters) {
    context.filters=filters
    const code=ts.transpileModule(`(${effects[index].expression.arguments[0].getText(ast)})`, {compilerOptions:{target:ts.ScriptTarget.ES2022}}).outputText
    return vm.runInNewContext(code,context)()
  }
  return {state,calls,pending,run}
}
const response = data => ({ok:true,json:async()=>data})

test('Histórico vacío conserva carrera y modalidad y nunca consulta global', async () => {
  const h=harness(); h.run(1,{servicio:'Derecho',modalidad:'Presencial'})
  for(const task of h.pending) task.resolve(response({encuestas:0,preguntas:[]}))
  await settle()
  assert.equal(h.calls.length,2)
  for(const call of h.calls){const url=new URL(call.url);assert.equal(url.searchParams.get('servicio'),'Derecho');assert.equal(url.searchParams.get('modalidad'),'Presencial')}
  assert.equal(h.state.HistoryStatus,'ready');assert.equal(h.state.Historico2025.encuestas,0)
})
test('Respuesta actual tardía no sobrescribe el filtro vigente aunque el transporte ignore abort', async () => {
  const h=harness(); const cleanup=h.run(0,{servicio:'Derecho'});cleanup();h.run(0,{servicio:'Pedagogía'})
  h.pending[1].resolve({career:'Pedagogía'});await settle();h.pending[0].resolve({career:'Derecho'});await settle()
  assert.equal(h.state.Summary.career,'Pedagogía');assert.equal(h.state.Loading,false);assert.equal(h.calls[0].signal.aborted,true)
})
test('Error actual obsoleto no reemplaza el éxito vigente', async () => {
  const h=harness();h.run(0,{servicio:'Derecho'})();h.run(0,{servicio:'Pedagogía'})
  h.pending[1].resolve({career:'Pedagogía'});await settle();h.pending[0].reject(new Error('anterior'));await settle()
  assert.equal(h.state.Error,null);assert.equal(h.state.Summary.career,'Pedagogía')
})
test('Histórico tardío no reemplaza la población vigente', async () => {
  const h=harness();h.run(1,{servicio:'Derecho'})();h.run(1,{servicio:'Pedagogía'})
  h.pending.slice(2).forEach(task=>task.resolve(response({career:'Pedagogía'})));await settle()
  h.pending.slice(0,2).forEach(task=>task.resolve(response({career:'Derecho'})));await settle()
  assert.equal(h.state.Historico2025.career,'Pedagogía');assert.equal(h.state.Historico2026.career,'Pedagogía')
})
test('Error histórico termina loading y mantiene resultados actuales', async () => {
  const h=harness();h.state.Summary={indice:73};h.run(1,{})
  h.pending[0].resolve({ok:false});h.pending[1].resolve(response({encuestas:10}));await settle()
  assert.equal(h.state.HistoryStatus,'error');assert.equal(h.state.Historico2025,null);assert.equal(h.state.Summary.indice,73)
  assert.match(source,/historyStatus === "error"[\s\S]*?No fue posible cargar el comparativo histórico/)
})
test('Fallo histórico obsoleto no cambia el estado listo vigente', async () => {
  const h=harness();h.run(1,{servicio:'Derecho'})();h.run(1,{servicio:'Pedagogía'})
  h.pending.slice(2).forEach(task=>task.resolve(response({encuestas:2})));await settle()
  h.pending[0].reject(new Error('anterior'));h.pending[1].resolve(response({encuestas:0}));await settle()
  assert.equal(h.state.HistoryStatus,'ready');assert.equal(h.state.Historico2025.encuestas,2)
})
test('Quitar filtros consulta global explícitamente sin recuperar la población anterior', async () => {
  const h=harness();h.run(1,{servicio:'Derecho',modalidad:'Presencial'})();h.run(1,{})
  for(const call of h.calls.slice(2)){const url=new URL(call.url);assert.equal(url.searchParams.has('servicio'),false);assert.equal(url.searchParams.has('modalidad'),false)}
  h.pending.forEach(task=>task.resolve(response({encuestas:1})));await settle();assert.equal(h.state.HistoryStatus,'ready')
})
test('Detalles no muestran conclusiones o clasificaciones heurísticas', () => {
  for(const label of ['Interpretación general','Lectura del resultado','Resumen preliminar de comentarios','Positivos explícitos','Oportunidades / incidencias']) assert.equal(source.includes(label),false,label)
  assert.match(source,/Análisis cualitativo con SIA · En construcción/)
})

for (const component of ['InstitutionalServices','AcademicServices','Facilities','SchoolEnvironment','Satisfaction','LanguageCenter','ContinuingEducation','DirectionResults']) {
  const content=readFileSync(new URL(`../src/components/calidad/${component}.tsx`,import.meta.url),'utf8')
  const parsed=ts.createSourceFile('component.tsx',content,ts.ScriptTarget.Latest,true,ts.ScriptKind.TSX)
  let effect
  function visit(node){
    if(ts.isCallExpression(node) && node.expression.getText(parsed)==='useEffect') effect=node.arguments[0]
    ts.forEachChild(node,visit)
  }
  visit(parsed)
  const code=ts.transpileModule(`(${effect.getText(parsed)})`,{compilerOptions:{target:ts.ScriptTarget.ES2022}}).outputText
  test(`${component}: actual, histórico y detalle descartan respuestas y errores obsoletos`,async()=>{
    const state={}, pending=[]
    function run(career){
      const scope={periodo:'2026',carrera:career,modalidad_academica:'Presencial'}
      const context={AbortController,JSON,key:JSON.stringify(scope),scope:JSON.stringify(scope),enabled:true,modalidad:'Escolarizadas',servicio:career}
      for(const name of ['Current','History','Result']) context['set'+name]=value=>{state[name]=value}
      for(const name of [...content.matchAll(/\b(fetch\w+|getCalidadSummary)\s*(?:<[^>]+>)?\(/g)].map(m=>m[1])) {
        context[name]=()=>{const task=deferred();pending.push({career,...task});return task.promise}
      }
      return vm.runInNewContext(code,context)()
    }
    const cleanup=run('Derecho');cleanup();run('Pedagogía')
    pending.filter(t=>t.career==='Pedagogía').forEach(t=>t.resolve({career:'Pedagogía'}));await settle()
    pending.filter(t=>t.career==='Derecho').forEach((t,i)=>i%2?t.reject(new Error('anterior')):t.resolve({career:'Derecho'}));await settle()
    assert.ok(Object.keys(state).length)
    for(const value of Object.values(state)){assert.equal(value.data.career,'Pedagogía');assert.ok(!value.error);if('loading' in value)assert.equal(value.loading,false)}
  })
  if(!['LanguageCenter','ContinuingEducation'].includes(component)) test(`${component}: error histórico finaliza carga sin ocultar actual`,async()=>{
    const state={}, scope={periodo:'2026',carrera:'Derecho'}, context={AbortController,JSON,key:JSON.stringify(scope),scope:JSON.stringify(scope),enabled:true,modalidad:'Escolarizadas',servicio:'Derecho'}
    for(const name of ['Current','History','Result'])context['set'+name]=value=>{state[name]=value}
    for(const name of [...content.matchAll(/\b(fetch\w+|getCalidadSummary)\s*(?:<[^>]+>)?\(/g)].map(m=>m[1]))context[name]=kind=>kind==='actual'?Promise.resolve({indice:75}):Promise.reject(new Error('histórico caído'))
    vm.runInNewContext(code,context)();await settle()
    if(component==='DirectionResults'){assert.equal(state.Result.loading,false);assert.equal(state.Result.error,true)}
    else {assert.equal(state.Current.data.indice,75);assert.equal(state.History.error,true);assert.equal(state.History.key,context.key)}
  })
}
