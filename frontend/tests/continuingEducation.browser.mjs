// Offline interaction test: real React component + API fixtures calculated by Python.
// Run: node --test tests/continuingEducation.browser.mjs
import assert from "node:assert/strict"
import { spawn, execFileSync } from "node:child_process"
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs"
import { tmpdir } from "node:os"
import path from "node:path"
import { fileURLToPath } from "node:url"
import test from "node:test"
import { createServer } from "vite"
import react from "@vitejs/plugin-react"

const root = fileURLToPath(new URL("../", import.meta.url))
const repo = path.dirname(root.replace(/[\\/]$/, ""))
const chrome = [process.env.CHROME_PATH, "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"].find(file => file && existsSync(file))
const pause = ms => new Promise(resolve => setTimeout(resolve, ms))

test("Educación Continua: navegación, filtros, detalle y carga en navegador", { skip: !chrome, timeout: 60000 }, async t => {
  const fixtures = JSON.parse(execFileSync(path.join(repo, ".venv/Scripts/python.exe"),
    ["-m", "backend.TESTS.continuing_education_browser_fixture"], { cwd: repo, encoding: "utf8" }))
  const directory = mkdtempSync(path.join(tmpdir(), "udl-continuing-education-browser-"))
  let browser, socket, server
  try {
    const entry = `
        import React from 'react';
        import { createRoot } from 'react-dom/client';
        import { ContinuingEducation } from '/src/components/calidad/ContinuingEducation.tsx';
        const fixtures = ${JSON.stringify(fixtures)};
        window.continuingEducationTest = { requests: [], fail: false, delay: 0 };
        window.fetch = async (input) => {
          const url = new URL(String(input), location.href);
          const kind = url.pathname.split('/').pop();
          const key = (url.searchParams.get('carrera') || '') + '|' + (url.searchParams.get('modalidad_academica') || '') + '|' + (url.searchParams.get('participacion') || '');
          continuingEducationTest.requests.push({kind, key});
          const fail = continuingEducationTest.fail;
          await new Promise(resolve => setTimeout(resolve, continuingEducationTest.delay));
          return new Response(JSON.stringify(fixtures[key][kind]), {status: fail ? 503 : 200});
        };
        createRoot(document.getElementById('root')).render(React.createElement(ContinuingEducation,
          {periodo: '2026', modalidad: 'Escolarizadas', servicio: ''}));
        `
    server = await createServer({ root, configFile: false, cacheDir: path.join(directory, "vite-cache"), plugins: [react(), {
      name: "continuing-education-browser-fixture",
      resolveId(id) { if (id === "/__continuing_education_entry.js") return id },
      load(id) { if (id === "/__continuing_education_entry.js") return entry },
      configureServer(vite) {
        vite.middlewares.use("/__continuing_education_test", async (_req, res) => {
          const html = '<!doctype html><html><body><div id="root"></div><script type="module" src="/__continuing_education_entry.js"></script></body></html>'
          res.setHeader("Content-Type", "text/html")
          res.end(await vite.transformIndexHtml("/__continuing_education_test", html))
        })
      },
    }], resolve: { alias: { "@": path.join(root, "src") } },
    logLevel: "error", server: { host: "127.0.0.1", port: 0, open: false } })
    await server.listen()
    const port = server.httpServer.address().port
    browser = spawn(chrome, ["--headless=new", "--no-sandbox", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
      "--remote-debugging-port=0", `--user-data-dir=${directory}`, "about:blank"], { stdio: "ignore", windowsHide: true })
    const activePort = path.join(directory, "DevToolsActivePort")
    for (let i = 0; !existsSync(activePort) && i < 150; i++) await pause(100)
    assert.ok(existsSync(activePort), "El navegador debe iniciar DevTools")
    const debugPort = readFileSync(activePort, "utf8").split("\n")[0]
    const pages = await fetch(`http://127.0.0.1:${debugPort}/json/list`, { signal: AbortSignal.timeout(5000) }).then(response => response.json())
    socket = new WebSocket(pages.find(page => page.type === "page").webSocketDebuggerUrl)
    await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject })
    let nextId = 0
    const pending = new Map()
    socket.onmessage = event => {
      const message = JSON.parse(event.data)
      if (message.method === "Runtime.exceptionThrown") console.error(JSON.stringify(message.params.exceptionDetails))
      const callback = pending.get(message.id)
      if (callback) { pending.delete(message.id); message.error ? callback.reject(message.error) : callback.resolve(message.result) }
    }
    const send = (method, params = {}) => new Promise((resolve, reject) => {
      const id = ++nextId
      const timeout = setTimeout(() => { pending.delete(id); reject(new Error(`DevTools timeout: ${method}`)) }, 10000)
      pending.set(id, { resolve: value => { clearTimeout(timeout); resolve(value) }, reject: error => { clearTimeout(timeout); reject(error) } })
      socket.send(JSON.stringify({ id, method, params }))
    })
    const evaluate = async expression => {
      const result = await send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true })
      assert.equal(result.exceptionDetails, undefined, JSON.stringify(result.exceptionDetails))
      return result.result.value
    }
    const until = async expression => {
      for (let i = 0; i < 150; i++) { if (await evaluate(expression)) return; await pause(50) }
      const body = await evaluate("document.body.innerText")
      console.error(`No se cumplió: ${expression}\n${body}`)
      assert.fail(`No se cumplió: ${expression}\n${body}`)
    }
    const select = (index, value) => evaluate(`(() => {
      const select = document.querySelectorAll('select')[${index}];
      Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(select, ${JSON.stringify(value)});
      select.dispatchEvent(new Event('change', {bubbles: true}));
    })()`)
    const ready = key => until(`window.continuingEducationTest && continuingEducationTest.requests.some(r => r.key === ${JSON.stringify(key)}) && document.querySelectorAll('article').length === 6 && !document.body.innerText.includes('Cargando comparativo')`)
    const checkMetrics = async key => {
      const expected = fixtures[key].actual
      assert.deepEqual(await evaluate("[...document.querySelectorAll('.text-3xl')].map(el => el.textContent)"),
        [expected.encuestas_analizadas.toLocaleString("es-MX"), expected.respuestas.toLocaleString("es-MX")])
      assert.deepEqual(await evaluate("[...document.querySelectorAll('article strong')].map(el => el.textContent)"), expected.criterios.map(row => row.resultado === null ? "—" : `${row.resultado.toFixed(2)}${row.unidad === 'porcentaje' ? '%' : ''}`))
      const participation = await evaluate("document.querySelector('[aria-label=Participación]').innerText")
      if (expected.participacion.si_pct === null) assert.match(participation, /Sin respuestas en este periodo/)
      else assert.ok(participation.includes(`Sí: ${expected.participacion.si_pct.toFixed(2)}% (${expected.participacion.si})`))
    }
    await send("Runtime.enable")
    await send("Page.navigate", { url: `http://127.0.0.1:${port}/__continuing_education_test` })
    await ready("||")
    await t.test("Global muestra seis indicadores y contexto separado sin histórico ni NPS", async () => {
      await checkMetrics("||")
      const text = await evaluate("document.body.innerText")
      assert.doesNotMatch(text, /Sin comparable histórico/)
      assert.doesNotMatch(text, /2025|Resultado general|Evaluaciones|Comentario de|NPS|promotores|pasivos|detractores/)
      assert.equal(await evaluate("document.querySelectorAll('select').length"), 3)
      assert.equal(await evaluate("continuingEducationTest.requests.every(r => r.kind === 'actual')"), true)
      const recommendation = await evaluate("document.querySelectorAll('article')[3].innerText")
      assert.match(recommendation, /Sí: 50\.00% \(1\)/)
      assert.match(recommendation, /No: 50\.00% \(1\)/)
    })
    await t.test("Carrera, modalidad y participación filtran toda la encuesta", async () => {
      await select(0, "Derecho"); await ready("Derecho||"); await checkMetrics("Derecho||")
      await select(1, "Ejecutiva"); await ready("Derecho|Ejecutiva|"); await checkMetrics("Derecho|Ejecutiva|")
      await select(2, "si"); await ready("Derecho|Ejecutiva|si"); await checkMetrics("Derecho|Ejecutiva|si")
      const media = await evaluate("document.querySelector('[aria-label=\"Medios de difusión\"]').innerText")
      assert.match(media, /Redes sociales/)
      assert.doesNotMatch(media, /Correo institucional/)
    })
    await t.test("Recomendación y comentarios permanecen en su detalle contextual", async () => {
      await evaluate("document.querySelectorAll('article button')[3].click()")
      await until("!!document.querySelector('[role=dialog]')")
      const text = await evaluate("document.querySelector('[role=dialog]').innerText")
      assert.match(text, /Sí: 0\.00% \(0\)/)
      assert.match(text, /Comentario de Derecho \/ Ejecutiva/)
      assert.doesNotMatch(text, /Comentario de Nutrición|Comentario de Derecho \/ Presencial|NPS|Fortalezas|sentimiento|2025/)
      assert.match(text, /¿Qué podríamos hacer para mejorar Educación Continua\?/)
      assert.match(text, /Análisis cualitativo con SIA · En construcción/)
      await evaluate("document.querySelector('[aria-label=\"Cerrar detalle del indicador\"]').click()")
      assert.equal(await evaluate("document.body.innerText.includes('Comentario de')"), false)
    })
    await t.test("Quitar filtros restaura el global", async () => {
      await select(0, ""); await ready("|Ejecutiva|si"); await checkMetrics("|Ejecutiva|si")
      await select(1, ""); await ready("||si"); await checkMetrics("||si")
      await select(2, ""); await ready("||"); await checkMetrics("||")
    })
    await t.test("No participantes conservan conocimiento válido sin inventar satisfacción", async () => {
      await select(2, "no"); await ready("||no"); await checkMetrics("||no")
      assert.equal(await evaluate("[...document.querySelectorAll('article')].filter(el => el.innerText.includes('Sin respuestas en este periodo')).length"), 4)
      assert.equal(await evaluate("document.querySelectorAll('article')[4].innerText.includes('Sí: 0.00%')"), true)
    })
    await t.test("Una población sin datos no muestra resultados cero", async () => {
      await select(0, "Derecho"); await ready("Derecho||no"); await checkMetrics("Derecho||no")
      assert.equal(await evaluate("[...document.querySelectorAll('article')].filter(el => el.innerText.includes('Sin respuestas en este periodo')).length"), 6)
    })
  } finally {
    socket?.close()
    if (browser && browser.exitCode === null && browser.signalCode === null) {
      const closed = new Promise(resolve => browser.once("exit", resolve))
      browser.kill()
      await Promise.race([closed, pause(2000)])
    }
    await server?.close()
    // Only remove the exact temporary profile created by this test.
    assert.ok(path.resolve(directory).startsWith(path.resolve(tmpdir()) + path.sep) && path.basename(directory).startsWith("udl-continuing-education-browser-"))
    rmSync(directory, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 })
  }
})
