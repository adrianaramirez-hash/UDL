// Offline interaction test: real React component + API fixtures calculated by Python.
// Run: node --test tests/languageCenter.browser.mjs
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

test("Centro de Idiomas: navegación, filtros, detalle y carga en navegador", { skip: !chrome, timeout: 60000 }, async t => {
  const fixtures = JSON.parse(execFileSync(path.join(repo, ".venv/Scripts/python.exe"),
    ["-m", "backend.TESTS.language_center_browser_fixture"], { cwd: repo, encoding: "utf8" }))
  const directory = mkdtempSync(path.join(tmpdir(), "udl-language-center-browser-"))
  let browser, socket, server
  try {
    const entry = `
        import React from 'react';
        import { createRoot } from 'react-dom/client';
        import { LanguageCenter } from '/src/components/calidad/LanguageCenter.tsx';
        const fixtures = ${JSON.stringify(fixtures)};
        window.languageCenterTest = { requests: [], fail: false, delay: 0 };
        window.fetch = async (input) => {
          const url = new URL(String(input), location.href);
          const kind = url.pathname.split('/').pop();
          const key = (url.searchParams.get('carrera') || '') + '|' + (url.searchParams.get('modalidad_academica') || '') + '|' + (url.searchParams.get('idioma') || '') + '|' + (url.searchParams.get('tipo_curso') || '');
          languageCenterTest.requests.push({kind, key});
          const fail = languageCenterTest.fail;
          await new Promise(resolve => setTimeout(resolve, languageCenterTest.delay));
          return new Response(JSON.stringify(fixtures[key][kind]), {status: fail ? 503 : 200});
        };
        createRoot(document.getElementById('root')).render(React.createElement(LanguageCenter,
          {periodo: '2026', modalidad: 'Escolarizadas', servicio: ''}));
        `
    server = await createServer({ root, configFile: false, cacheDir: path.join(directory, "vite-cache"), plugins: [react(), {
      name: "language-center-browser-fixture",
      resolveId(id) { if (id === "/__language_center_entry.js") return id },
      load(id) { if (id === "/__language_center_entry.js") return entry },
      configureServer(vite) {
        vite.middlewares.use("/__language_center_test", async (_req, res) => {
          const html = '<!doctype html><html><body><div id="root"></div><script type="module" src="/__language_center_entry.js"></script></body></html>'
          res.setHeader("Content-Type", "text/html")
          res.end(await vite.transformIndexHtml("/__language_center_test", html))
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
    const ready = key => until(`window.languageCenterTest && languageCenterTest.requests.some(r => r.key === ${JSON.stringify(key)}) && document.querySelectorAll('article').length === 3 && !document.body.innerText.includes('Cargando comparativo')`)
    const checkMetrics = async key => {
      const expected = fixtures[key].actual
      assert.deepEqual(await evaluate("[...document.querySelectorAll('.text-3xl')].map(el => el.textContent)"),
        [expected.encuestas_analizadas.toLocaleString("es-MX"), expected.respuestas.toLocaleString("es-MX")])
      assert.deepEqual(await evaluate("[...document.querySelectorAll('article strong')].map(el => el.textContent)"), expected.criterios.map(row => row.indice?.toFixed(2) ?? "—"))
      const usage = await evaluate("document.querySelector('[aria-label=\"Uso del servicio\"]').innerText")
      if (expected.uso.si_pct === null) assert.match(usage, /Sin respuestas en este periodo/)
      else assert.ok(usage.includes(`Sí: ${expected.uso.si_pct.toFixed(2)}% (${expected.uso.si})`))
    }
    await send("Runtime.enable")
    await send("Page.navigate", { url: `http://127.0.0.1:${port}/__language_center_test` })
    await ready("|||")
    await t.test("Global muestra tres criterios y contextos separados sin histórico", async () => {
      await checkMetrics("|||")
      const text = await evaluate("document.body.innerText")
      assert.doesNotMatch(text, /Sin comparable histórico/)
      assert.doesNotMatch(text, /2025|Resultado general|Evaluaciones|Comentario de|ranking/)
      assert.equal(await evaluate("document.querySelectorAll('select').length"), 4)
      assert.equal(await evaluate("languageCenterTest.requests.every(r => r.kind === 'actual')"), true)
    })
    await t.test("Carrera, modalidad, idioma y curso filtran la misma encuesta", async () => {
      await select(0, "Derecho"); await ready("Derecho|||"); await checkMetrics("Derecho|||")
      await select(1, "Ejecutiva"); await ready("Derecho|Ejecutiva||"); await checkMetrics("Derecho|Ejecutiva||")
      await select(1, ""); await ready("Derecho|||")
      await select(2, "Francés"); await ready("Derecho||Francés|"); await checkMetrics("Derecho||Francés|")
      await select(3, "Sabatino"); await ready("Derecho||Francés|Sabatino"); await checkMetrics("Derecho||Francés|Sabatino")
      const language = await evaluate("document.querySelector('[aria-label=Idiomas]').innerText")
      assert.match(language, /Francés/)
      assert.doesNotMatch(language, /Inglés/)
    })
    await t.test("Detalle conserva las dos preguntas abiertas y profesor solo descriptivo", async () => {
      await evaluate("document.querySelector('article button').click()")
      await until("!!document.querySelector('[role=dialog]')")
      const text = await evaluate("document.querySelector('[role=dialog]').innerText")
      assert.match(text, /Comentario de Derecho \/ Francés \/ Sabatino/)
      assert.doesNotMatch(text, /Comentario de Nutrición|Comentario de Derecho \/ Inglés|Fortalezas|sentimiento|ranking|2025/)
      assert.match(text, /Docente descriptivo/)
      assert.match(text, /¿Qué es lo que menos te gusta/)
      assert.match(text, /¿Qué es lo que más te gusta/)
      assert.match(text, /Análisis cualitativo con SIA · En construcción/)
      await evaluate("document.querySelector('[aria-label=\"Cerrar detalle del criterio\"]').click()")
      assert.equal(await evaluate("document.body.innerText.includes('Comentario de')"), false)
    })
    await t.test("Quitar los filtros recupera el global", async () => {
      await select(0, ""); await ready("||Francés|Sabatino"); await checkMetrics("||Francés|Sabatino")
      await select(2, ""); await ready("|||Sabatino"); await checkMetrics("|||Sabatino")
      await select(3, ""); await ready("|||"); await checkMetrics("|||")
    })
    await t.test("Población sin respuestas no muestra calificaciones cero", async () => {
      await select(0, "Nutrición"); await ready("Nutrición|||")
      await select(2, "Francés"); await ready("Nutrición||Francés|"); await checkMetrics("Nutrición||Francés|")
      assert.equal(await evaluate("[...document.querySelectorAll('article')].filter(el => el.innerText.includes('Sin respuestas en este periodo')).length"), 3)
      assert.equal(await evaluate("document.querySelector('[aria-label=Idiomas]').innerText.includes('Otro')"), false)
    })
    await t.test("Error de carga se informa y permite recuperar el componente", async () => {
      await evaluate("languageCenterTest.fail = true")
      await select(3, "Sabatino")
      await until("!!document.querySelector('[role=alert]')")
      assert.match(await evaluate("document.querySelector('[role=alert]').innerText"), /No fue posible cargar Centro de Idiomas/)
      // Remount at the page level, as when the user reopens the section.
      await send("Page.reload")
      await ready("|||"); await checkMetrics("|||")
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
    assert.ok(path.resolve(directory).startsWith(path.resolve(tmpdir()) + path.sep) && path.basename(directory).startsWith("udl-language-center-browser-"))
    rmSync(directory, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 })
  }
})
