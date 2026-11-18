# Empaquetado de producción

Referencia funcional `f5b8afc299fe58b86a4b82d5f956243920b566bc`.

## Runtime

| Componente | Requisito |
| --- | --- |
| Backend | CPython 3.12. Versión de referencia 3.12.3. |
| SQLite | 3.35 o posterior por `DELETE … RETURNING` en el flujo OIDC. Versión de referencia 3.45.1. |
| Build del frontend | Node 24.19.0 y npm 11.17.0 como versiones de referencia. Vite declara `^20.19.0 || >=22.12.0` en el lockfile. |

El servidor de producción sirve el frontend compilado desde el proxy inverso.
Node se usa durante el build.

## Instalación y pruebas en una copia limpia

Preparar una copia de las fuentes de `backend/` y `frontend/` con los manifiestos de
esta revisión. La copia empieza sin `.venv`, `node_modules` ni archivos de configuración
local. Desde su raíz, crear un entorno virtual nuevo con Python 3.12.3.

```sh
python -m venv .venv
```

Activar el entorno con `.venv/Scripts/Activate.ps1` en PowerShell o
`source .venv/bin/activate` en una shell POSIX. Después ejecutar lo siguiente.

```sh
python -m pip install --only-binary=:all: -r backend/requirements.txt
python -m pip check
python -m backend.TESTS.smoke_production
python -m pip install --only-binary=:all: -r backend/requirements-test.txt
python -m unittest discover -s backend/TESTS -p 'test_*.py'
```

El manifiesto de producción fija las dependencias directas y transitivas. El de
pruebas agrega el cliente HTTP que usa `TestClient`.

La prueba de arranque espera `200` en `/health`, `401` en las rutas privadas sin
sesión y `503` al iniciar login con Google sin configurar. También genera un PDF
sintético en memoria. Las pruebas automatizadas de autenticación y PDF usan
fixtures. Las utilidades de auditoría manual que requieren una captura de datos y
un renderer externo se ejecutan por separado.

Desde `frontend/` en esa misma copia limpia ejecutar lo siguiente.

```sh
npm ci
npm run build
node --test tests/*.test.mjs
node --test --test-concurrency=1 tests/*.browser.mjs
```

`npm run build` ejecuta `tsc -b` y `vite build` y escribe `frontend/dist/`.
Para la suite de navegador se necesita Chrome o Edge. Las pruebas existentes
que generan fixtures Python en esta suite usan `.venv/Scripts/python.exe`.
Ejecutar `git diff --check` en el árbol de desarrollo.

## Contenido de los paquetes

| Paquete | Incluir |
| --- | --- |
| Backend | Código Python activo bajo `backend/`, conservando su estructura, y `backend/requirements.txt`. |
| Frontend | Únicamente el contenido generado en `frontend/dist/`, incluidos sus assets. |

Excluir `.env`, `.venv/`, `.git/`, `.runtime/`, `artifacts/`, `node_modules/`,
fuentes del frontend, pruebas, manifiesto de pruebas, SQLite local, logs,
credenciales de Google, OAuth Client Secret, JSON de cuenta de servicio, cachés
y archivos temporales. El build del frontend se ejecuta sin el `.env` del backend.

## Arranque productivo

Instalar `backend/requirements.txt` en el entorno del servicio. Desde el directorio
que contiene `backend/`, iniciar una instancia detrás del proxy inverso.

```sh
python -m uvicorn backend.GATEWAY.main:app --host 127.0.0.1 --port 8000 --workers 1 --no-access-log
```

El proxy termina HTTPS, sirve el contenido de `frontend/dist/`, reenvía `/api/`
a FastAPI y resuelve las rutas React a `index.html`. Si el proxy y el backend
están en contenedores separados, usar una interfaz de la red privada de
contenedores y dejar el puerto de FastAPI sin publicar.

Inyectar las variables y credenciales en el backend al arrancar. Montar el JSON
de la cuenta de servicio fuera de la raíz pública y ubicar `AUTH_SESSION_DB`
en un directorio privado y persistente. Configurar el proxy para que omita del
registro los parámetros del callback. Usar `GET /health` para comprobar que
responde `200` con `{"status":"healthy"}`.
