# Auditoría de Instalaciones y equipo tecnológico

Fuente: captura normalizada `udl-instalaciones-audit.json`. Fecha de lectura: 2026-11-11. Ámbito: Escolarizadas; todas las carreras y modalidades académicas; 2026 frente a 2025 del mismo ámbito.

- Resultado general: **67.86**.
- Respuestas a criterios: **5,428**.
- Encuestas analizadas: **611**.
- Criterios encontrados: 10; con histórico: 10; sin histórico: 0.
- Comparativo global: 65.24 → 67.86; delta: 2.62 puntos.
- El motor estable calcula índices y denominadores. Encuestas analizadas es la unión de formularios con al menos una respuesta válida. Vacíos y no aplicables no se convierten en cero.
- La comparación usa exclusivamente IDs canónicos y aprobación COMPARABLE_HISTORICO de ambos mapas y registros; los globales usan la misma intersección con respuestas válidas en ambos años.

| Nombre | ID canónico | Respuestas actuales | Encuestas con respuesta válida | Resultado 2026 | Resultado 2025 | Delta | Histórico disponible | Estado visual |
|---|---|---:|---:|---:|---:|---:|:---:|---|
| Laboratorio de cómputo | INS_LAB_COMPUTO | 385 | 385 | 63.12 | 58.58 | 4.54 | Sí | Resultado actual + barras 2025/2026 + delta |
| Salones de clase | INS_SALONES | 603 | 603 | 71.14 | 69.17 | 1.97 | Sí | Resultado actual + barras 2025/2026 + delta |
| Sanitarios | INS_SANITARIOS | 605 | 605 | 74.75 | 73.12 | 1.63 | Sí | Resultado actual + barras 2025/2026 + delta |
| Equipo didáctico HyFlex | INS_HYFLEX | 577 | 577 | 66.81 | 66.66 | 0.15 | Sí | Resultado actual + barras 2025/2026 + delta |
| Instalaciones del plantel | INS_PLANTEL | 600 | 600 | 69.12 | 67.75 | 1.37 | Sí | Resultado actual + barras 2025/2026 + delta |
| Instalaciones específicas para el aprendizaje | INS_ESPECIFICAS | 592 | 592 | 72.34 | 70.26 | 2.08 | Sí | Resultado actual + barras 2025/2026 + delta |
| Biblioteca | INS_BIBLIOTECA | 442 | 442 | 72.57 | 69.68 | 2.89 | Sí | Resultado actual + barras 2025/2026 + delta |
| Cafetería | INS_CAFETERIA | 463 | 463 | 61.07 | 57.64 | 3.43 | Sí | Resultado actual + barras 2025/2026 + delta |
| Espacios al aire libre | INS_AIRE_LIBRE | 565 | 565 | 66.95 | 65.12 | 1.83 | Sí | Resultado actual + barras 2025/2026 + delta |
| WiFi para estudiantes | INS_WIFI | 596 | 596 | 60.70 | 54.44 | 6.26 | Sí | Resultado actual + barras 2025/2026 + delta |

## Biblioteca y Cafetería: trazabilidad independiente

| Año | Criterio | ID canónico | Pregunta | Columna aprobada | Encabezado RAW verificado |
|---|---|---|---|---|---|
| 2026 | Biblioteca | INS_BIBLIOTECA | ESC_INS_BIBLIOTECA | DM | [Biblioteca] |
| 2026 | Cafetería | INS_CAFETERIA | ESC_INS_CAFETERIA | DN | [Cafetería] |
| 2025 | Biblioteca | INS_BIBLIOTECA | ESC25_INS_BIBLIOTECA | AR | [Biblioteca] |
| 2025 | Cafetería | INS_CAFETERIA | ESC25_INS_CAFETERIA | AS | [Cafetería] |

Biblioteca solo se compara con INS_BIBLIOTECA; Cafetería solo con INS_CAFETERIA. No se utilizan posiciones, nombres similares ni equivalencias inferidas.

## Verificación de población

- Derecho / Todas las modalidades: resultado 56.77; respuestas 231; encuestas 28; histórico 60.77 → 56.77.
- Todas las carreras / Presencial: resultado 67.23; respuestas 4702; encuestas 509; histórico 65.24 → 67.23.
- Derecho / Presencial: resultado 55.09; respuestas 221; encuestas 27; histórico 60.77 → 55.09.

Quitar filtros recupera el global. La auditoría es temporal, no se muestra en producción y no incluye comentarios personales. No se avanza a Ambiente escolar.

## Validación de implementación · 11 de noviembre de 2026

- Backend: `python -m unittest discover -s backend/TESTS -p 'test_*.py'` — 51 pruebas aprobadas.
- Frontend: `node --test --test-concurrency=1 tests/facilities.test.mjs tests/facilities.browser.mjs tests/academicServices.test.mjs tests/academicServices.browser.mjs tests/direction.test.mjs tests/institutionalServices.test.mjs` — 50 pruebas aprobadas. Los navegadores se ejecutan secuencialmente para evitar interferencias entre servidores de prueba.
- Navegador: global inicial, diez nombres únicos, filtros carrera/modalidad, regreso a global, detalle y comentarios filtrados, población sin respuestas, error histórico independiente y descarte de respuestas de solicitudes anteriores. Estos casos usan poblaciones de prueba calculadas por el motor estable.
- Biblioteca/Cafetería: pruebas con mapas reordenados y nombres engañosos, histórico ausente de un criterio, valores distintos por criterio y rechazo de encabezados RAW cruzados. La auditoría real verifica además DM/AR para Biblioteca y DN/AS para Cafetería.
- Vite: `node node_modules/vite/bin/vite.js build` — paquete de producción generado. Advertencia de tamaño de bundle superior a 500 kB.
- TypeScript: comparación de diagnósticos completos contra la instantánea anterior a esta implementación; 10 errores preexistentes, 0 nuevos. La compilación completa de TypeScript sigue pendiente de corregir esos errores fuera del alcance de esta sección.
- `git diff --check`: sin errores de espacios.
- Integridad: 87 archivos previos preservados byte por byte. Solo se modificaron los dos archivos de integración enumerados abajo. Los bloques JSX de Dirección, Servicios institucionales y Servicios académicos son idénticos a la instantánea inicial. Normalización, metodología, equivalencias, exportaciones PDF/CSV y las demás secciones no cambiaron.

## Archivos de esta implementación

Modificados:

- `backend/GATEWAY/main.py`: registro del router independiente de instalaciones.
- `frontend/src/pages/EncuestaCalidad.tsx`: integración de Instalaciones; reemplazo exclusivo de su vista anterior y exclusión de sus resúmenes redundantes.

Agregados:

- `backend/GATEWAY/facilities_api.py`
- `backend/SERVICES/calidad/facilities.py`
- `backend/TESTS/test_facilities.py`
- `backend/TESTS/facilities_browser_fixture.py`
- `backend/TESTS/audit_facilities.py`
- `frontend/src/components/calidad/Facilities.tsx`
- `frontend/src/services/facilities.ts`
- `frontend/tests/facilities.test.mjs`
- `frontend/tests/facilities.browser.mjs`
- `docs/auditorias/instalaciones.md`
