# Auditoría de Centro de Idiomas

Lectura de fuentes: 2026-11-11. Captura: `udl-idiomas-audit.json`. Ámbito: Escolarizadas, 2026, todas las carreras, modalidades, idiomas y tipos de curso.

- Encuestas analizadas: **637**.
- Respuestas a criterios: **282**.
- Encuestas analizadas es la unión de formularios con al menos una respuesta válida en los criterios, uso, variables descriptivas o preguntas abiertas aprobadas de la sección. Respeta RESPUESTA_VALIDA y no convierte respuestas categóricas o abiertas en puntajes.
- Respuestas a criterios suma exclusivamente las respuestas válidas a los tres criterios especializados. Cada índice y distribución procede del motor estable. No se publica un resultado general agregado ni un ranking docente.
- Se retiraron los registros de uso, idioma, curso y profesor: los tres índices, denominadores y distribuciones cuantitativas permanecieron idénticos.

## Tres criterios especializados

| Criterio | ID canónico | Pregunta | Columna | Respuestas válidas | Encuestas con respuesta válida | Resultado 2026 | Histórico disponible | Estado visual esperado |
|---|---|---|---|---:|---:|---:|:---:|---|
| ¿En general te gusta tu clase? | IDI_GUSTO_CLASE | ESC_IDI_GUSTO_CLASE | CS | 94 | 94 | 92.02 | No | Barra actual + distribución + Sin comparable histórico |
| ¿En general estás satisfecho con tu docente? | IDI_SAT_DOCENTE | ESC_IDI_SAT_DOCENTE | CT | 94 | 94 | 91.49 | No | Barra actual + distribución + Sin comparable histórico |
| ¿En general es satisfactorio para ti el servicio que te ofrecen en la oficina del Centro de Idiomas? | IDI_SAT_OFICINA | ESC_IDI_SAT_OFICINA | CU | 94 | 94 | 88.03 | No | Barra actual + distribución + Sin comparable histórico |

**Los tres criterios especializados 2026 no tienen comparación histórica válida con la pregunta general 2025.**

La implementación no solicita ni une datos históricos. No usa nombres similares, coincidencias aproximadas o equivalencias nuevas. No muestra barras 2025 contra 2026 ni calcula deltas.

- Referencia 2025 existente: `ESC25_SER_CENTRO_IDIOMAS` / `SER_CENTRO_IDIOMAS_2025` en `ESC25_03_SERVICIOS`, COMPARABLE_HISTORICO=NO. Solo trazabilidad metodológica; su resultado no se utiliza ni se muestra como comparable.

## Uso del servicio · indicador descriptivo

| Respuestas Sí | Respuestas No | Total válido | Uso Sí (%) | Uso No (%) |
|---:|---:|---:|---:|---:|
| 94 | 543 | 637 | 14.76 | 85.24 |

## Idiomas

Denominador propio: **94 respuestas válidas** a esta pregunta. Los faltantes no se convierten en Otro ni en cero.

| Valor informado | Respuestas | Porcentaje |
|---|---:|---:|
| Inglés | 94 | 100.00 |

## Tipos de curso

Denominador propio: **94 respuestas válidas** a esta pregunta. Los faltantes no se convierten en Otro ni en cero.

| Valor informado | Respuestas | Porcentaje |
|---|---:|---:|
| Cuatrimestral entre semana | 54 | 57.45 |
| Cuatrimestral sabatino | 16 | 17.02 |
| Intensivo | 10 | 10.64 |
| Semestral entre semana | 14 | 14.89 |

## Profesor actual

- ID canónico: IDI_PROFESOR. Respuestas descriptivas válidas: 94; valores distintos informados: 14.
- Solo se presentan los nombres informados en orden alfabético dentro del detalle contextual. No se asignan puntajes ni posiciones por cantidad de respuestas. Los nombres no se reproducen en este informe.

## Comprobación de filtros

- carrera=Derecho: encuestas 28; respuestas a criterios 24; uso 28.57%.
- modalidad_academica=Presencial: encuestas 509; respuestas a criterios 279; uso 18.27%.
- idioma=Inglés: encuestas 94; respuestas a criterios 282; uso 100.00%.
- tipo_curso=Cuatrimestral entre semana: encuestas 54; respuestas a criterios 162; uso 100.00%.

Idioma y curso se filtran por identidad de encuesta antes de proyectar todos sus criterios, uso, distribuciones, profesores y comentarios. Quitar filtros restaura el global.

Las dos preguntas abiertas se mantienen separadas en el detalle contextual, sin clasificación semántica. Esta auditoría no incluye comentarios personales ni se muestra en producción. No se avanza a Educación Continua.

## Validación de implementación · 11 de noviembre de 2026

- Backend: `python -m unittest discover -s backend/TESTS -p 'test_*.py'` — 91 pruebas aprobadas, incluidas 12 de Centro de Idiomas.
- Frontend: `node --test --test-concurrency=1 tests/languageCenter.test.mjs tests/languageCenter.browser.mjs tests/satisfaction.test.mjs tests/satisfaction.browser.mjs tests/schoolEnvironment.test.mjs tests/schoolEnvironment.browser.mjs tests/facilities.test.mjs tests/facilities.browser.mjs tests/academicServices.test.mjs tests/academicServices.browser.mjs tests/direction.test.mjs tests/institutionalServices.test.mjs` — 97 pruebas aprobadas, incluidas 13 de Centro de Idiomas.
- Navegador: vista global útil sin filtros, tres criterios, uso separado, distribuciones, filtros por carrera/modalidad/idioma/curso, detalle con las dos preguntas abiertas y profesor descriptivo, regreso al global, estados vacíos y recuperación tras error de carga. Usa poblaciones de prueba calculadas por el servicio y el motor estable.
- Histórico: la prueba incorpora la pregunta general 2025 con un texto idéntico al criterio especializado y confirma que no modifica ningún resultado ni crea un cuarto criterio. El componente solicita únicamente datos actuales; la API no dispone de endpoint histórico para esta sección.
- Vite: `node node_modules/vite/bin/vite.js build` — paquete de producción generado; advertencia de tamaño de bundle superior a 500 kB.
- TypeScript: mismos 10 diagnósticos preexistentes, comparando archivos, códigos y mensajes completos; 0 nuevos. La compilación completa sigue pendiente de corregir esos errores previos fuera del alcance de esta sección.
- `git diff --check`: sin errores de espacios.
- Integridad: 119 archivos previos preservados byte por byte. Solo cambian los dos archivos de integración enumerados abajo. Los bloques JSX de los seis módulos aprobados y de Educación Continua son idénticos a la instantánea inicial. La comparación del AST confirma que las funciones de PDF/CSV permanecen idénticas; las únicas funciones auxiliares retiradas son las cinco de la antigua interfaz de Centro de Idiomas. No se modifican metodología, equivalencias ni normalización.

## Archivos de esta implementación

Modificados:

- `backend/GATEWAY/main.py`: registro del router independiente de Centro de Idiomas.
- `frontend/src/pages/EncuestaCalidad.tsx`: integración de la sección, reemplazo exclusivo de su interfaz anterior y eliminación de sus auxiliares de comparación/clasificación y resúmenes redundantes.

Agregados:

- `backend/GATEWAY/language_center_api.py`
- `backend/SERVICES/calidad/language_center.py`
- `backend/TESTS/test_language_center.py`
- `backend/TESTS/language_center_browser_fixture.py`
- `backend/TESTS/audit_language_center.py`
- `frontend/src/components/calidad/LanguageCenter.tsx`
- `frontend/src/services/languageCenter.ts`
- `frontend/tests/languageCenter.test.mjs`
- `frontend/tests/languageCenter.browser.mjs`
- `docs/auditorias/centro_idiomas.md`
