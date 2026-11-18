import assert from "node:assert/strict"
import { spawn } from "node:child_process"
import { existsSync } from "node:fs"
import { mkdtemp, readFile, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import path from "node:path"
import { setTimeout as pause } from "node:timers/promises"
import { fileURLToPath } from "node:url"
import test from "node:test"
import { createServer } from "vite"

const frontend = fileURLToPath(new URL("../", import.meta.url))
const chrome = [process.env.CHROME_PATH, "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
  "/usr/bin/google-chrome", "/usr/bin/chromium"].find(file => file && existsSync(file))

async function waitFor(check, message) {
  for (let i = 0; i < 120; i++) {
    if (await check()) return
    await pause(100)
  }
  assert.fail(message)
}

test("Acceso directo y módulos de Inicio con la SPA real", { skip: !chrome, timeout: 60000 }, async t => {
  const directory = await mkdtemp(path.join(tmpdir(), "udl-access-browser-"))
  let browser, socket, server
  let scenario, releaseSession, requests = []
  try {
    server = await createServer({
      root: frontend,
      cacheDir: path.join(directory, "vite"),
      logLevel: "error",
      server: { port: 0, host: "127.0.0.1" },
      plugins: [{
        name: "access-test-api",
        configureServer(vite) {
          vite.middlewares.use(async (req, res, next) => {
            if (!req.url?.startsWith("/api/")) return next()
            requests.push(req.url)
            res.setHeader("Content-Type", "application/json")
            if (req.url === "/api/auth/me") {
              const { status, user } = scenario
              await new Promise(resolve => {
                const previous = releaseSession
                releaseSession = () => { previous?.(); resolve() }
                res.once("close", resolve)
              })
              res.statusCode = status
              res.end(JSON.stringify(user ?? {}))
            } else if (req.url === "/api/auth/logout") {
              res.statusCode = 204
              res.end()
            } else if (req.url.startsWith("/api/observacion/")) {
              res.statusCode = 503
              res.end("{}")
            } else {
              const params = new URL(req.url, "http://localhost").searchParams
              res.end(JSON.stringify({
                encuestas: 1, respuestas_kpi_validas: 0, indice_general: null, qa_total: 0,
                fila_inicio: 0, fila_fin: 0, secciones: [], preguntas: [], comentarios: [], personas: [],
                comparativos_direccion: { por_carrera_modalidad: [], preguntas_por_carrera_modalidad: [] },
                filtros: {
                  periodos: ["2025", "2026"], modalidades: ["Escolarizadas"],
                  servicios: scenario.user?.alcance.tipo === "asignado" ? ["Derecho"] : ["Derecho", "Nutrición"],
                  seleccion: { periodo: params.get("periodo"), modalidad: "Escolarizadas", servicio: null },
                },
              }))
            }
          })
        },
      }],
    })
    await server.listen()
    const origin = `http://127.0.0.1:${server.httpServer.address().port}`
    browser = spawn(chrome, ["--headless=new", "--no-sandbox", "--disable-gpu", "--no-first-run",
      "--no-default-browser-check", "--remote-debugging-port=0", `--user-data-dir=${directory}`, "about:blank"],
    { stdio: "ignore", windowsHide: true })
    const activePort = path.join(directory, "DevToolsActivePort")
    await waitFor(() => existsSync(activePort), "Chrome debe iniciar DevTools")
    const debugPort = (await readFile(activePort, "utf8")).split("\n")[0]
    const pages = await fetch(`http://127.0.0.1:${debugPort}/json/list`).then(response => response.json())
    socket = new WebSocket(pages.find(page => page.type === "page").webSocketDebuggerUrl)
    await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject })
    let sequence = 0
    const pending = new Map()
    socket.onmessage = ({ data }) => {
      const message = JSON.parse(data)
      const callback = pending.get(message.id)
      if (callback) { pending.delete(message.id); callback(message) }
    }
    const send = (method, params = {}) => new Promise((resolve, reject) => {
      const id = ++sequence
      const timeout = setTimeout(() => reject(new Error(`DevTools: ${method}`)), 5000)
      pending.set(id, message => {
        clearTimeout(timeout)
        if (message.error) reject(new Error(message.error.message))
        else resolve(message.result)
      })
      socket.send(JSON.stringify({ id, method, params }))
    })
    const evaluate = async expression => {
      const result = await send("Runtime.evaluate", { expression, returnByValue: true })
      assert.equal(result.exceptionDetails, undefined)
      return result.result.value
    }
    const heading = () => evaluate("document.querySelector('h1')?.textContent.trim()")
    const quality = "encuesta_calidad.ver"
    const globalQuality = "encuesta_calidad.ver_global"
    const observation = "observacion_clases.ver"
    const user = (rol, permissions, scope) => ({
      nombre: rol === "DG" ? "Adriana de prueba" : "Rubí de prueba",
      email: "prueba@example.test", rol,
      nombre_rol: { DG: "Dirección General", DC: "Dirección de carrera", DF: "Dirección Finanzas" }[rol],
      modulos: ["encuesta_calidad", "observacion_clases"], permisos: permissions,
      alcance: { tipo: scope, servicios: scope === "asignado" ? ["Derecho"] : [] },
    })
    const cases = [
      { name: "sin sesión", status: 401, heading: "Iniciar sesión", allowed: false },
      { name: "DC sin módulo", status: 200, user: user("DC", [], "sin_acceso"), heading: "Acceso no autorizado", allowed: false },
      { name: "Observación sin permiso", path: "/observacion-clases", status: 200,
        user: user("DC", [quality], "asignado"), heading: "Acceso no autorizado", allowed: false },
      { name: "DC autorizado", status: 200, user: user("DC", [quality], "asignado"), heading: "Encuesta de Calidad", allowed: true },
      { name: "DG autorizado", status: 200, user: user("DG", [quality], "global"), heading: "Encuesta de Calidad", allowed: true },
    ]
    for (const [index, entry] of cases.entries()) {
      await t.test(entry.name, async () => {
        scenario = entry
        releaseSession = null
        requests = []
        // A distinct document for each direct browser navigation.
        await send("Page.navigate", { url: `${origin}${entry.path ?? "/encuesta-calidad"}?case=${index}` })
        await waitFor(() => releaseSession, "Debe consultar /api/auth/me")
        await waitFor(async () => await heading() === "Verificando sesión…", "Debe esperar a /me")
        assert.deepEqual([...new Set(requests)], ["/api/auth/me"])
        releaseSession()
        await waitFor(async () => await heading() === entry.heading, "La ruta debe mostrar el estado correcto")
        if (entry.allowed) {
          await waitFor(() => requests.some(url => url.startsWith("/api/encuesta-calidad/")), "El módulo autorizado debe cargar")
          assert.ok(await evaluate(`document.body.innerText.includes(${JSON.stringify(entry.user.nombre)})`))
          await waitFor(() => evaluate("Array.from(document.querySelectorAll('option')).some(option => option.value === 'Derecho')"),
            "Debe renderizar el catálogo recibido")
          if (entry.user.rol === "DC") {
            assert.ok(await evaluate("document.body.innerText.includes('Mis servicios asignados')"))
            assert.ok(!await evaluate("Array.from(document.querySelectorAll('option')).some(option => option.value === 'Nutrición')"))
          }
        } else {
          assert.ok(requests.every(url => url === "/api/auth/me"))
          if (entry.status === 401) {
            assert.equal(await evaluate("location.pathname"), "/login")
            assert.equal(await evaluate("document.querySelector('a[href=\"/api/auth/login\"]')?.getAttribute('href')"), "/api/auth/login")
          }
        }
      })
    }
    await evaluate("[...document.querySelectorAll('button')].find(b => b.textContent.trim() === 'Cerrar sesión').click()")
    await waitFor(async () => await heading() === "Iniciar sesión", "El logout debe volver al login")
    assert.ok(requests.includes("/api/auth/logout"))

    const modules = [
      { permission: quality, path: "/encuesta-calidad", title: "Encuesta de Calidad" },
      { permission: observation, path: "/observacion-clases", title: "Observación de Clases" },
    ]
    const homeLinks = () => evaluate("[...document.querySelectorAll('section[aria-labelledby=modules-title] a')].map(a => a.getAttribute('href'))")
    const sidebarLinks = () => evaluate("[...document.querySelectorAll('aside nav ul a')].map(a => a.getAttribute('href'))")
    const openHome = async (command, params = {}) => {
      releaseSession = null
      await send(command, params)
      await waitFor(() => releaseSession, "Inicio debe consultar /me")
      releaseSession()
      await waitFor(() => evaluate(`document.querySelector('header a[href="/"]')?.textContent.trim() === 'SIA / Inicio'`),
        "El Home autenticado debe estar visible")
    }
    const homeCases = [
      { name: "DG global", account: user("DG", [quality, globalQuality, observation], "global"),
        greeting: "Buenos días, Adriana.", institutional: true },
      { name: "DC con alcance asignado", account: user("DC", [quality], "asignado"),
        greeting: "Buenos días, Rubí.", assigned: true },
      { name: "DG sin permiso global", account: user("DG", [observation], "global"),
        greeting: "Buenos días, Adriana." },
      { name: "DF sin acceso institucional", account: user("DF", [], "sin_acceso"),
        greeting: "Buenos días, Rubí." },
      { name: "cuenta sin nombre", account: { ...user("DC", [], "sin_acceso"), nombre: "   " },
        greeting: "Buenos días." },
      { name: "cuenta con el correo como nombre",
        account: { ...user("DC", [], "sin_acceso"), nombre: "prueba@example.test" }, greeting: "Buenos días." },
    ]
    for (const { name, account, greeting, institutional = false, assigned = false } of homeCases) {
      await t.test(`Inicio con ${name}`, async () => {
        scenario = { status: 200, user: account }
        await openHome("Page.navigate", { url: origin + "/?modules=" + encodeURIComponent(name) })
        assert.equal(await heading(), greeting)
        assert.equal(await evaluate("document.querySelector('main h1 + p')?.textContent.trim()"), account.nombre_rol)
        assert.ok((await evaluate("[...document.querySelectorAll('header p')].map(p => p.textContent.trim())"))
          .includes(account.nombre_rol))
        const content = await evaluate("document.querySelector('main').innerText")
        for (const text of ["ASISTENTE SIA", "Métricas institucionales clave", "87.4%",
          "El Asesor SIA preparó un resumen estratégico de la situación académica institucional."]) {
          assert.equal(content.includes(text), institutional, text)
        }
        if (!institutional) {
          const introduction = assigned
            ? "Consulta los resultados y herramientas disponibles para las áreas que tienes asignadas."
            : "Consulta las herramientas disponibles para tu cuenta."
          assert.ok(content.includes(introduction))
        }
        const allowed = modules.filter(item => account.permisos.includes(item.permission))
        const paths = allowed.map(item => item.path)
        assert.deepEqual(await homeLinks(), paths)
        assert.deepEqual(await sidebarLinks(), paths)
        for (const { path, title } of allowed) {
          await evaluate(`document.querySelector('section[aria-labelledby=modules-title] a[href="${path}"]').click()`)
          await waitFor(async () => await evaluate("location.pathname") === path && await heading() === title,
            `La tarjeta debe abrir ${path}`)
          await evaluate(`document.querySelector('header a[href="/"]').click()`)
          await waitFor(() => evaluate("location.pathname === '/' && Boolean(document.querySelector('#modules-title'))"),
            "Inicio/SIA debe regresar al Home")
        }
        await openHome("Page.reload")
        assert.deepEqual(await homeLinks(), paths)
      })
    }
  } finally {
    releaseSession?.()
    socket?.close()
    if (browser && browser.exitCode === null && browser.signalCode === null) {
      const closed = new Promise(resolve => browser.once("exit", resolve))
      browser.kill()
      await Promise.race([closed, pause(2000)])
    }
    await server?.close()
    assert.equal(path.dirname(path.resolve(directory)), path.resolve(tmpdir()))
    assert.ok(path.basename(directory).startsWith("udl-access-browser-"))
    await rm(directory, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 })
  }
})
