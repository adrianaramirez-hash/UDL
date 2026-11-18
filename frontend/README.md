# React + TypeScript + Vite

## Acceso local

Configura el `.env` de la raíz usando `.env.example`. Registra
`http://localhost:5173/api/auth/callback` como URI de redirección del cliente
web de Google y comparte la pestaña `ACCESOS` con la cuenta de servicio del
backend mediante el spreadsheet que la contiene. Usa el mismo host y puerto
en el navegador y en `AUTH_PUBLIC_ORIGIN`.

Para el callback anterior, configura `AUTH_PUBLIC_ORIGIN=http://localhost:5173`
y `AUTH_ALLOW_HTTP_LOCALHOST=1`. Conserva los valores de conexión existentes al
usar la plantilla. Omite `ACCESOS_CACHE_SECONDS`, `AUTH_SESSION_SECONDS` y
`AUTH_SESSION_DB` para usar los valores predeterminados.

Desde la raíz inicia `python -m uvicorn backend.GATEWAY.main:app --no-access-log`.
En `frontend/`, ejecuta `npm run dev`. Vite envía `/api` al backend local.
Para producción sirve `npm run build` y FastAPI bajo el mismo origen HTTPS,
registra el callback HTTPS y deshabilita `AUTH_ALLOW_HTTP_LOCALHOST`. Usa un
directorio privado y persistente para `AUTH_SESSION_DB`, compartido por los
workers del mismo host. El proxy debe omitir parámetros del callback en logs.

This template provides a minimal setup to get React working in Vite with HMR and some Oxlint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/)

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the Oxlint configuration

If you are developing a production application, we recommend enabling type-aware lint rules by installing `oxlint-tsgolint` and editing `.oxlintrc.json`:

```json
{
  "$schema": "./node_modules/oxlint/configuration_schema.json",
  "plugins": ["react", "typescript", "oxc"],
  "options": {
    "typeAware": true
  },
  "rules": {
    "react/rules-of-hooks": "error",
    "react/only-export-components": ["warn", { "allowConstantExport": true }]
  }
}
```

See the [Oxlint rules documentation](https://oxc.rs/docs/guide/usage/linter/rules) for the full list of rules and categories.
