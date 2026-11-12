// Offline interaction test: real React component + API fixtures calculated by Python.
// Run: node --test tests/schoolEnvironment.browser.mjs
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

test("Ambiente escolar: navegación, filtros, detalle y fallos históricos en navegador", { skip: !chrome, timeout: 60000 }, async t => {
  const fixtures = JSON.parse(execFileSync(path.join(repo, ".venv/Scripts/python.exe"),
    ["-m", "backend.TESTS.school_environment_browser_fixture"], { cwd: repo, encoding: "utf8" }))
  const directory = mkdtempSync(path.join(tmpdir(), "udl-environment-browser-"))
  let browser, socket, server
  try {
    const entry = `
        import React from 'react';
        import { createRoot } from 'react-dom/client';
        import { SchoolEnvironment } from '/src/components/calidad/SchoolEnvironment.tsx';
        const fixtures = ${JSON.stringify(fixtures)};
        window.environmentTest = { requests: [], failHistory: false, delay: 0 };
        window.fetch = async (input) => {
          const url = new URL(String(input), location.href);
          const kind = url.pathname.split('/').pop();
          const key = (url.searchParams.get('carrera') || '') + '|' + (url.searchParams.get('modalidad_academica') || '');
          environmentTest.requests.push({kind, key});
          const fail = environmentTest.failHistory;
          if (kind === 'historico') await new Promise(resolve => setTimeout(resolve, environmentTest.delay));
          return new Response(JSON.stringify(fixtures[key][kind]), {status: kind === 'historico' && fail ? 503 : 200});
        };
        createRoot(document.getElementById('root')).render(React.createElement(SchoolEnvironment,
          {periodo: '2026', modalidad: 'Escolarizadas', servicio: ''}));
        `
    server = await createServer({ root, configFile: false, cacheDir: path.join(directory, "vite-cache"), plugins: [react(), {
      name: "environment-browser-fixture",
      resolveId(id) { if (id === "/__environment_entry.js") return id },
      load(id) { if (id === "/__environment_entry.js") return entry },
      configureServer(vite) {
        vite.middlewares.use("/__environment_test", async (_req, res) => {
          const html = '<!doctype html><html><body><div id="root"></div><script type="module" src="/__environment_entry.js"></script></body></html>'
          res.setHeader("Content-Type", "text/html")
          res.end(await vite.transformIndexHtml("/__environment_test", html))
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
    const ready = key => until(`window.environmentTest && environmentTest.requests.filter(r => r.key === ${JSON.stringify(key)}).length >= 2 && document.querySelectorAll('article').length === 3 && !document.body.innerText.includes('Cargando comparativo')`)
    const checkMetrics = async key => {
      const expected = fixtures[key].actual
      const actual = await evaluate("[...document.querySelectorAll('.text-3xl')].map(el => el.textContent)")
      assert.deepEqual(actual, [expected.indice?.toFixed(2) ?? "—", expected.respuestas.toLocaleString("es-MX"), expected.encuestas_analizadas.toLocaleString("es-MX")])
    }
    const checkHistory = async key => {
      const expected = fixtures[key].historico
      const meters = await evaluate("[...document.querySelectorAll('section')[0].querySelectorAll('[role=meter]')].map(el => Number(el.getAttribute('aria-valuenow')))")
      assert.deepEqual(meters, [expected.indice_2025, expected.indice_2026])
      const requests = await evaluate(`environmentTest.requests.filter(r => r.key === ${JSON.stringify(key)}).map(r => r.kind)`)
      assert.ok(requests.includes("actual") && requests.includes("historico"))
    }
    await send("Runtime.enable")
    await send("Page.navigate", { url: `http://127.0.0.1:${port}/__environment_test` })
    await ready("|")
    await t.test("Vista global inmediata y tres criterios distintos", async () => {
      await ready("|")
      await checkMetrics("|")
      await checkHistory("|")
      assert.equal(await evaluate("document.querySelector('select').value"), "")
      const names = await evaluate("[...document.querySelectorAll('article h4')].map(el => el.textContent)")
      assert.equal(new Set(names).size, 3)
      assert.ok(names.every(Boolean))
      assert.equal(await evaluate("document.body.innerText.includes('Comentario de')"), false)
    })
    await t.test("Incidencia tiene detalle independiente con conteos y puntos porcentuales", async () => {
      await evaluate("[...document.querySelectorAll('button')].find(el => el.textContent === 'Ver incidencia reportada').click()")
      await until("!!document.querySelector('[role=dialog]')")
      const detail = await evaluate("document.querySelector('[role=dialog]').innerText")
      assert.match(detail, /Incidencia reportada/)
      assert.match(detail, /Sí: 33\.33% \(1 respuestas\)/)
      assert.match(detail, /No: 66\.67% \(2 respuestas\)/)
      assert.match(detail, /→ estable/)
      assert.match(detail, /Análisis cualitativo con SIA · En construcción/)
      assert.match(detail, /Comentario de Derecho/)
      await evaluate("document.querySelector('[aria-label=\"Cerrar detalle del criterio\"]').click()")
    })
    await t.test("Carrera y modalidad recalculan contadores, histórico y comentarios", async () => {
      await select(0, "Derecho"); await ready("Derecho|"); await checkMetrics("Derecho|")
      await select(1, "Ejecutiva"); await ready("Derecho|Ejecutiva"); await checkMetrics("Derecho|Ejecutiva")
      await checkHistory("Derecho|Ejecutiva")
      const incidence = await evaluate("document.querySelector('[aria-label=\"Incidencia reportada\"]').innerText")
      assert.match(incidence, /Sí: 0\.00%/)
      assert.match(incidence, /↓ −100\.00 pp/)
      assert.equal(await evaluate("[...document.querySelectorAll('article')].filter(el => el.innerText.includes('Sin respuestas en este periodo')).length"), 1)
      await evaluate("[...document.querySelectorAll('article button')][0].click()")
      await until("!!document.querySelector('[role=dialog]')")
      const detail = await evaluate("document.querySelector('[role=dialog]').innerText")
      assert.match(detail, /Comentario de Derecho \/ Ejecutiva/)
      assert.doesNotMatch(detail, /Comentario de Nutrición|Comentario de Derecho \/ Presencial/)
      assert.match(detail, /Análisis cualitativo con SIA · En construcción/)
      await evaluate("document.querySelector('[aria-label=\"Cerrar detalle del criterio\"]').click()")
    })
    await t.test("Modalidad independiente y regreso a global", async () => {
      await select(0, ""); await ready("|Ejecutiva"); await checkMetrics("|Ejecutiva")
      await select(1, ""); await ready("|"); await checkMetrics("|")
    })
    await t.test("Fallo histórico conserva los tres resultados actuales", async () => {
      await evaluate("environmentTest.failHistory = true; environmentTest.delay = 100")
      await select(0, "Nutrición")
      await until("document.body.innerText.includes('No fue posible cargar el comparativo histórico')")
      await checkMetrics("Nutrición|")
      assert.equal(await evaluate("document.querySelectorAll('article').length"), 3)
      assert.equal(await evaluate("document.body.innerText.includes('Sin comparable histórico')"), false)
    })
    await t.test("Sin población no se convierten los resultados a cero", async () => {
      await evaluate("environmentTest.failHistory = false")
      await select(1, "Ejecutiva"); await ready("Nutrición|Ejecutiva"); await checkMetrics("Nutrición|Ejecutiva")
      assert.equal(await evaluate("[...document.querySelectorAll('article')].filter(el => el.innerText.includes('Sin respuestas en este periodo')).length"), 3)
    })
    await t.test("Cambios rápidos descartan el histórico de la población anterior", async () => {
      await evaluate("environmentTest.delay = 350")
      await select(0, "Derecho")
      await until("document.querySelectorAll('article').length === 3 && document.body.innerText.includes('Cargando comparativo')")
      await select(0, "")
      await ready("|Ejecutiva"); await checkMetrics("|Ejecutiva"); await checkHistory("|Ejecutiva")
      await select(1, ""); await ready("|"); await checkMetrics("|"); await checkHistory("|")
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
    assert.ok(path.resolve(directory).startsWith(path.resolve(tmpdir()) + path.sep) && path.basename(directory).startsWith("udl-environment-browser-"))
    rmSync(directory, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 })
  }
})
