# Auditoría de Ambiente escolar

Lectura de fuentes: 2026-11-11. Captura: `udl-ambiente-audit.json`. Ámbito: Escolarizadas, todas las carreras y modalidades académicas, 2026 frente a 2025 de la misma población.

- Resultado general: **79.83**.
- Respuestas a criterios: **1,851**.
- Encuestas analizadas: **638**.
- Criterios con histórico: 3; sin histórico: 0.
- Comparativo global: 77.80 → 79.83; delta 2.03 puntos de índice.
- La incidencia NO entra al Resultado general, a Respuestas a criterios ni a la distribución cuantitativa. Se verificó retirando sus registros: el resultado, respuestas, tres criterios y distribución permanecen idénticos.
- Encuestas analizadas es la unión de filas de formulario con al menos una respuesta cuantitativa válida o una respuesta binaria válida de incidencia. No cuenta formularios con solo texto abierto ni suma los denominadores de las preguntas.
- Cada criterio utiliza exclusivamente sus respuestas válidas; vacíos y no aplicables no son cero. Se preserva el motor estable de índices y sus pesos.

| Nombre | ID canónico | Respuestas actuales | Encuestas con respuesta válida | Resultado 2026 | Resultado 2025 | Delta | Histórico | Estado visual |
|---|---|---:|---:|---:|---:|---:|:---:|---|
| Relación con profesores / asesores | AMB_REL_PROF | 633 | 633 | 82.07 | 80.96 | 1.11 | Sí | Barras 2025/2026 y delta |
| Relación con compañeros | AMB_REL_COMP | 634 | 634 | 78.39 | 75.60 | 2.79 | Sí | Barras 2025/2026 y delta |
| Ambiente escolar en mi servicio | AMB_AMBIENTE | 584 | 584 | 79.02 | 76.84 | 2.18 | Sí | Barras 2025/2026 y delta |

## Incidencia reportada · indicador independiente

| Año | Respuestas Sí | Respuestas No | Total válido | Porcentaje Sí | Porcentaje No |
|---|---:|---:|---:|---:|---:|
| 2026 | 65 | 573 | 638 | 10.19 | 89.81 |
| 2025 | 218 | 1536 | 1754 | 12.43 | 87.57 |

- Histórico aprobado disponible: Sí.
- Porcentaje comparable 2025: 12.43%; 2026: 10.19%; delta: **-2.24 puntos porcentuales**.
- Los porcentajes son tasas de respuestas Sí/No sobre su total válido. No constituyen puntajes de índice. Ausencia de respuestas se conserva como null, mientras que una tasa válida de 0% permanece visible.

## Equivalencias y población

| Año | ID canónico | Pregunta aprobada | Columna de origen |
|---|---|---|---|
| 2026 | AMB_REL_PROF | ESC_AMB_REL_PROF | DR |
| 2026 | AMB_REL_COMP | ESC_AMB_REL_COMP | DS |
| 2026 | AMB_AMBIENTE | ESC_AMB_AMBIENTE | DT |
| 2026 | AMB_PROBLEMA | ESC_AMB_PROBLEMA | DU |
| 2025 | AMB_REL_PROF | ESC25_AMB_REL_PROF | AW |
| 2025 | AMB_REL_COMP | ESC25_AMB_REL_COMP | AX |
| 2025 | AMB_AMBIENTE | ESC25_AMB_AMBIENTE | AY |
| 2025 | AMB_PROBLEMA | ESC25_AMB_PROBLEMA | AZ |

Solo se usan IDs canónicos aprobados y COMPARABLE_HISTORICO en ambos mapas y registros. El global comparable usa la misma intersección de criterios válidos en ambos años. No se usan nombres similares, posiciones ni aproximaciones.

- Derecho / Todas las modalidades: resultado 81.92; respuestas 83; encuestas 28; incidencia 10.71%; delta de incidencia -5.38 puntos porcentuales.
- Todas las carreras / Presencial: resultado 78.73; respuestas 1478; encuestas 509; incidencia 11.39%; delta de incidencia -1.04 puntos porcentuales.
- Derecho / Presencial: resultado 81.25; respuestas 80; encuestas 27; incidencia 11.11%; delta de incidencia -4.98 puntos porcentuales.

Quitar filtros restaura el global. Los comentarios y descripciones permanecen en el detalle contextual de la aplicación; esta auditoría no incluye textos personales y no se muestra en producción. No se avanza a Satisfacción y recomendación.

## Validación de implementación · 11 de noviembre de 2026

- Backend: `python -m unittest discover -s backend/TESTS -p 'test_*.py'` — 65 pruebas aprobadas, incluidas 14 de Ambiente escolar.
- Frontend: `node --test --test-concurrency=1 tests/schoolEnvironment.test.mjs tests/schoolEnvironment.browser.mjs tests/facilities.test.mjs tests/facilities.browser.mjs tests/academicServices.test.mjs tests/academicServices.browser.mjs tests/direction.test.mjs tests/institutionalServices.test.mjs` — 69 pruebas aprobadas, incluidas 19 de Ambiente escolar.
- Las pruebas interactivas usan poblaciones de prueba calculadas por el motor estable. Comprueban vista global, filtros, tres criterios, incidencia independiente, detalle con conteos y comentarios, ausencia de datos, fallos históricos y descarte de solicitudes anteriores. Las regresiones de los módulos anteriores pasan.
- La primera ejecución de frontend, simultánea con la revisión de TypeScript, agotó el tiempo de carga de la prueba de navegador preexistente de Servicios académicos. Se repitió la suite completa sin esa carga simultánea: 69/69 aprobadas, sin modificar la prueba anterior.
- Vite: `node node_modules/vite/bin/vite.js build` — paquete de producción generado; advertencia de tamaño de bundle superior a 500 kB.
- TypeScript: mismos 10 diagnósticos preexistentes comparando códigos, archivos y mensajes completos; 0 nuevos. La compilación completa sigue bloqueada por esos errores previos fuera de este alcance.
- `git diff --check`: sin errores de espacios.
- Integridad: 101 archivos previos preservados byte por byte. Solo cambian los dos archivos de integración enumerados abajo. Los bloques JSX de Dirección, Servicios institucionales, Servicios académicos e Instalaciones son idénticos a la instantánea anterior a esta tarea. No se modifican normalización, metodología, equivalencias ni exportaciones PDF/CSV.

## Archivos de esta implementación

Modificados:

- `backend/GATEWAY/main.py`: registro del router independiente de Ambiente escolar.
- `frontend/src/pages/EncuestaCalidad.tsx`: integración del componente y exclusión exclusiva de Ambiente de los resúmenes, criterios y comentarios genéricos redundantes.

Agregados:

- `backend/GATEWAY/school_environment_api.py`
- `backend/SERVICES/calidad/school_environment.py`
- `backend/TESTS/test_school_environment.py`
- `backend/TESTS/school_environment_browser_fixture.py`
- `backend/TESTS/audit_school_environment.py`
- `frontend/src/components/calidad/SchoolEnvironment.tsx`
- `frontend/src/services/schoolEnvironment.ts`
- `frontend/tests/schoolEnvironment.test.mjs`
- `frontend/tests/schoolEnvironment.browser.mjs`
- `docs/auditorias/ambiente_escolar.md`
