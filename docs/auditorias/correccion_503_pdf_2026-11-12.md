# Corrección del HTTP 503 del PDF — 2026-11-12

## Excepción original, identificada antes de modificar código

La solicitud HTTP real al endpoint devolvió 503. La instrumentación aislada de gspread capturó antes de su transformación:

```text
gspread.exceptions.APIError
APIError: [429]: Quota exceeded for quota metric 'Read requests' and limit 'Read requests per minute per user' of service 'sheets.googleapis.com' for consumer 'project_number:787091861072'.
```

Traza de la versión anterior, con sus líneas originales:

| Archivo | Función | Línea |
| --- | --- | ---: |
| backend/GATEWAY/results_report_api.py | download_report: carga de 2025 | 20 |
| backend/GATEWAY/institutional_services_api.py | _fuente | 19 |
| backend/GATEWAY/calidad_api.py | _obtener_base_preview | 400 |
| backend/GATEWAY/calidad_api.py | _calcular_base_preview | 323 |
| backend/FUENTES/calidad/google_sheets.py | leer_filas: worksheet.get | 159 |
| .venv/Lib/site-packages/gspread/http_client.py | request: raise APIError | 128 |

Fuente: `ESC_2025_ARCHIVE`, documento `Encuesta de calidad de servicios escolarizados y licenciaturas ejecutivas 2025 (Respuestas)`, pestaña `Respuestas de formulario 1`, rango `A2:BK1755`.

Fue cuota de lecturas de Sheets: no fue CORS, credenciales ni timeout. Esta transcripción procede de la salida de diagnóstico anterior a los cambios; no se conserva un JSON original de esa ejecución.

## Flujo y lecturas

Antes: endpoint → `_fuente` actual → `_obtener_base_preview` / repositorio / normalizador → Sheets; después `_fuente` histórico y otra carga de Servicios institucionales. La proyección reutilizaba filas, pero mapas, escalas, estado, metadatos y catálogo de personas se solicitaban repetidamente mediante clientes/documentos distintos.

La ejecución original hizo **40 solicitudes HTTP de Sheets antes de fallar**: 26 de metadatos y 14 de valores, incluida la última con 429. Registró 12 `open_by_key`, 26 `Spreadsheet.fetch_sheet_metadata`, 8 `get_all_records`, 13 `Worksheet.get`, 3 `row_values` y 1 `col_values`. Estas cuentas de métodos se superponen: no deben sumarse como solicitudes externas. Una generación completa anterior se estima en aproximadamente 66 solicitudes; no llegó a completarse en la medición original.

Ahora: endpoint → snapshot vivo sin filtros de población (actual e histórico) → aplicar filtros → proyección existente de nueve módulos → renderer existente. El transporte memoiza GET idénticos y reutiliza clientes/documentos durante la construcción. El catálogo se precarga mediante el mismo loader existente.

Una generación completamente fría medida usa **16 solicitudes HTTP reales: 5 metadatos y 11 valores**:

| Lectura | Solicitudes |
| --- | ---: |
| Metadatos MASTER, ESC_MAIN, VIR_MAIN, PRE_MAIN y archivo 2025 | 5 |
| Mapa de preguntas | 1 |
| Firmas de fuentes | 1 |
| Encabezados de tres fuentes, requeridos por la validación existente | 3 |
| Estado ETL | 1 |
| Columna para última fila actual | 1 |
| Filas actuales `A2134:EO2771` | 1 |
| Escalas | 1 |
| Histórico `A2:BK1755` | 1 |
| Catálogo de personas | 1 |

Hay 5 aperturas efectivas. Aún se invocan loaders de gspread desde las funciones existentes, pero sus GET repetidos se resuelven en memoria. **La proyección de los nueve módulos realizó cero solicitudes externas**. No existía una lectura completa de respuestas por cada módulo; el problema eran las lecturas redundantes compartidas y de configuración.

## Caché, consistencia y errores

- Fuente en memoria: clave `(periodo, fuente/modalidad, año histórico)`, contiene exclusivamente datos sin filtros. Se aplica servicio después de recuperarla.
- PDF: clave de ocho filtros: periodo, modalidad, servicio, carrera, modalidad académica, idioma, tipo de curso y participación. Global y Derecho nunca comparten el resultado renderizado.
- Vigencia máxima: los 60 segundos existentes, acotada además por el vencimiento más próximo de los datasets normalizados. No se prolonga su frescura ni se usa un PDF vencido si falla la recarga.
- Un bloqueo serializa construcción y uso de la sesión; solicitudes concurrentes comparten una carga completa. El contexto del transporte se restaura al salir. Fuera del PDF, el transporte mantiene su comportamiento anterior.
- Snapshot lógico: cada recurso se lee como máximo una vez por carga y se mantiene fijo durante la proyección; Sheets no ofrece una transacción atómica entre documentos.
- Solo 429 permite un reintento, tras 2 segundos; el segundo error se propaga. No se cachean errores. La excepción original queda registrada antes del 503 y se devuelve `Retry-After: 60`.
- No se usa captura de validación, archivo local ni mock como fuente productiva. Las pruebas offline conservan sus fixtures exclusivamente en pruebas.

## Descargas reales en Uvicorn, puerto 8000

| Caso | Intento | HTTP | Content-Type | Lecturas Sheets | Páginas |
| --- | ---: | ---: | --- | ---: | ---: |
| Escolarizadas, todas las carreras/modalidades | 1 | 200 | application/pdf | 16 | 14 |
| Escolarizadas, todas las carreras/modalidades | 2 | 200 | application/pdf | 0 | 14 |
| Escolarizadas, Derecho, Presencial | 1 | 200 | application/pdf | 0 | 11 |
| Escolarizadas, Derecho, Presencial | 2 | 200 | application/pdf | 0 | 11 |

Parámetros Global: `periodo=2026&modalidad=Escolarizadas`.

Parámetros Derecho: `periodo=2026&modalidad=Escolarizadas&carrera=Derecho&modalidad_academica=Presencial`.

Las cuatro peticiones llegaron al servidor real y descargaron archivos con `Content-Disposition: attachment`. Los cuatro abren con PyMuPDF sin reparación; se verificaron filtros en cada página y hashes idénticos entre repeticiones de cada caso. Derecho utilizó el mismo snapshot de datos, con su propio resultado filtrado. Una ejecución independiente de Derecho con caché vacía también obtuvo 200 y 16 lecturas, con el mismo contenido por página.

Se compararon todas las páginas con los PDF aprobados: mismas cifras y texto, salvo la etiqueta genérica `Servicio institucional 09`, ahora resuelta como `Paramédico` por el helper de encabezados ya existente. No se cambió el diseño ni el renderer.

Evidencia: `artifacts/calidad/validacion_descargas_pdf_2026-11-12.json`. Los PDF descargados durante esta prueba se guardaron en el directorio temporal de validación, sin sobrescribir los artefactos aprobados.

## Archivos de esta corrección

Código productivo:

- `backend/FUENTES/calidad/google_sheets.py`
- `backend/FUENTES/calidad/read_snapshot.py` (nuevo)
- `backend/SERVICES/calidad/results_report_source.py` (nuevo)
- `backend/GATEWAY/results_report_api.py`

Pruebas y evidencia:

- `backend/TESTS/test_pdf_source_snapshot.py` (nuevo)
- `backend/TESTS/test_results_report_pdf.py`
- `backend/TESTS/audit_results_report.py`
- Este informe y el JSON de evidencia.

El repositorio ya tenía cambios de etapas anteriores. Se verificaron hashes contra el inicio de esta tarea: ningún archivo del frontend, dashboard, CSV, normalizador, KPI, servicios de cálculo, proyección ni renderer del PDF cambió durante esta corrección. No se editó Google Sheets.

## Regresiones

- `python -m unittest discover -s backend/TESTS -p 'test_*.py'`: **134 pruebas aprobadas**, incluidas 24 relacionadas con PDF/snapshot.
- `node --test tests/resultsReportPdf.test.mjs`: **4 pruebas aprobadas**.
- `node node_modules/vite/bin/vite.js build`: aprobado; advertencia heredada de tamaño de bundle.
- `git diff --check`: aprobado.
- Cobertura adicional: claves de todos los filtros, periodos/fuentes, vencimiento, recarga fallida sin datos vencidos, concurrencia, no mutación de datasets, aislamiento fuera del PDF y reintento limitado de 429.

**DESCARGA PDF CORREGIDA**
