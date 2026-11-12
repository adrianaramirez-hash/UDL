# Auditoría de Educación Continua

Lectura de fuentes: 2026-11-11. Captura: `udl-continua-audit.json`. Ámbito: Escolarizadas, 2026, todas las carreras y modalidades; sin filtro de participación.

- Encuestas analizadas: **637**.
- Respuestas a criterios: **1,398**.
- Encuestas analizadas es la unión de formularios con al menos una respuesta válida en cualquiera de los seis indicadores, participación, medio o comentario aprobado. Los datos descriptivos no se convierten en puntajes.
- Respuestas a criterios suma los seis indicadores cuantitativos: tres escalas de experiencia y tres preguntas binarias. Sus denominadores son independientes: conocer la oferta no requiere haber participado.
- No se publica un resultado general agregado. Se retiraron participación y medios y se comprobó que los seis resultados, denominadores y distribuciones permanecen idénticos.

## Participación · indicador descriptivo independiente

| Respuestas Sí | Respuestas No | Total válido | Participación Sí (%) | Participación No (%) |
|---:|---:|---:|---:|---:|
| 31 | 606 | 637 | 4.87 | 95.13 |

## Seis indicadores especializados

| Indicador | ID canónico | Pregunta aprobada | Columna | Escala | Respuestas válidas | Encuestas válidas | Resultado 2026 | Unidad | Histórico | Estado visual |
|---|---|---|---|---|---:|---:|---:|---|:---:|---|
| Calidad de cursos y diplomados | EC_CALIDAD | ESC_EC_CALIDAD | CY | CALIDAD_5 | 31 | 31 | 75.00 | Índice 0–100 | No | Resultado actual + distribución + Sin comparable histórico |
| Relevancia de los temas para la formación profesional | EC_TEMAS | ESC_EC_TEMAS | CZ | CALIDAD_5 | 31 | 31 | 76.61 | Índice 0–100 | No | Resultado actual + distribución + Sin comparable histórico |
| Horarios y modalidades | EC_HORARIOS | ESC_EC_HORARIOS | DA | CALIDAD_5 | 31 | 31 | 73.39 | Índice 0–100 | No | Resultado actual + distribución + Sin comparable histórico |
| Recomendación de programas de Educación Continua | EC_RECOMIENDA | ESC_EC_RECOMIENDA | DC | BINARIA_SI_NO | 31 | 31 | 83.87 | % Sí | No | Resultado actual + distribución + Sin comparable histórico |
| Conocimiento de cursos, talleres y diplomados | EC_CONOCE | ESC_EC_CONOCE | DE | BINARIA_SI_NO | 637 | 637 | 35.79 | % Sí | No | Resultado actual + distribución + Sin comparable histórico |
| Conocimiento de dónde consultar la oferta académica | EC_SABE_DONDE | ESC_EC_SABE_DONDE | DF | BINARIA_SI_NO | 637 | 637 | 43.49 | % Sí | No | Resultado actual + distribución + Sin comparable histórico |

**Los criterios especializados 2026 no tienen comparación histórica válida con la pregunta general 2025.**

La implementación no solicita ni une datos históricos; no utiliza posiciones, palabras similares, coincidencias aproximadas ni equivalencias nuevas. No muestra barras 2025/2026 ni calcula deltas.

- Referencia metodológica existente: `ESC25_SER_EDUC_CONT` / `SER_EDUC_CONT_2025` en `ESC25_03_SERVICIOS`, COMPARABLE_HISTORICO=NO. Su resultado no se usa como comparable de ningún indicador especializado.

## Recomendación de programas

**Recomendación NO se calcula como NPS.** Se conserva la escala BINARIA_SI_NO: porcentaje de respuestas Sí sobre el total válido, sin restar No ni asignar categorías de promotores, pasivos o detractores.

| Sí | No | Total válido | Sí (%) | No (%) |
|---:|---:|---:|---:|---:|
| 26 | 5 | 31 | 83.87 | 16.13 |

## Medios por los que se enteran

Denominador propio: **31 respuestas válidas**. Se conservan las categorías informadas, sin puntajes ni faltantes convertidos en Otro.

| Medio | Respuestas | Porcentaje |
|---|---:|---:|
| Correo institucional | 12 | 38.71 |
| Profesores | 8 | 25.81 |
| Página web | 4 | 12.90 |
| Redes sociales | 7 | 22.58 |

## Comprobación de filtros

- carrera=Derecho: encuestas 28; respuestas a criterios 64; participación 7.14%; respuestas de medios 2.
- modalidad_academica=Presencial: encuestas 509; respuestas a criterios 1114; participación 4.72%; respuestas de medios 24.
- participacion=si: encuestas 31; respuestas a criterios 186; participación 100.00%; respuestas de medios 31.
- participacion=no: encuestas 606; respuestas a criterios 1212; participación 0.00%; respuestas de medios 0.

La participación se filtra por identidad de encuesta antes de recalcular todos los indicadores, medios y comentarios. Quitar filtros restaura el global. Los estados vacíos se conservan como null; un cero válido se conserva como resultado.

La pregunta EC_MEJORA se muestra solo en el detalle contextual, sin clasificación semántica. Esta auditoría no contiene comentarios personales ni se muestra en producción. No se avanza al PDF.

## Validación de implementación · 11 de noviembre de 2026

- Backend: `python -m unittest discover -s backend/TESTS -p 'test_*.py'` — 104 pruebas aprobadas, incluidas 13 de Educación Continua.
- Frontend: `node --test --test-concurrency=1 tests/continuingEducation.test.mjs tests/continuingEducation.browser.mjs tests/languageCenter.test.mjs tests/languageCenter.browser.mjs tests/satisfaction.test.mjs tests/satisfaction.browser.mjs tests/schoolEnvironment.test.mjs tests/schoolEnvironment.browser.mjs tests/facilities.test.mjs tests/facilities.browser.mjs tests/academicServices.test.mjs tests/academicServices.browser.mjs tests/direction.test.mjs tests/institutionalServices.test.mjs` — 111 pruebas aprobadas, incluidas 14 de Educación Continua.
- Navegador: global sin filtros, seis indicadores separados, recomendación Sí/No sin NPS, participación y medios descriptivos, filtros de carrera/modalidad/participación, comentarios únicamente en detalle, regreso a global y estados vacíos. Verifica también que quienes no participaron conservan sus respuestas válidas de conocimiento sin recibir calificaciones de experiencia ficticias. Usa poblaciones de prueba calculadas por el servicio y el motor estable.
- Histórico: la prueba incorpora la pregunta general 2025 con texto idéntico al criterio de calidad y confirma que no cambia ningún resultado ni crea indicadores adicionales. El componente solicita únicamente datos actuales y la API no dispone de endpoint histórico para esta sección.
- Vite: `node node_modules/vite/bin/vite.js build` — paquete de producción generado; advertencia de tamaño de bundle superior a 500 kB.
- TypeScript: mismos 10 diagnósticos preexistentes, comparando archivos, códigos y mensajes completos; 0 nuevos. La compilación completa sigue pendiente de corregir esos errores previos fuera del alcance de esta sección.
- `git diff --check`: sin errores de espacios.
- Integridad: 128 archivos previos preservados byte por byte. Solo cambian los dos archivos de integración enumerados abajo. Los bloques JSX de los siete módulos aprobados son idénticos a la instantánea inicial. La comparación del AST confirma que las funciones PDF/CSV permanecen idénticas y que solo se retiraron cinco auxiliares de la antigua interfaz de Educación Continua. No se modifican metodología, equivalencias ni normalización.

## Archivos de esta implementación

Modificados:

- `backend/GATEWAY/main.py`: registro del router independiente.
- `frontend/src/pages/EncuestaCalidad.tsx`: integración del componente; reemplazo exclusivo de la interfaz anterior de Educación Continua, sus auxiliares de comparación/clasificación y sus resúmenes redundantes.

Agregados:

- `backend/GATEWAY/continuing_education_api.py`
- `backend/SERVICES/calidad/continuing_education.py`
- `backend/TESTS/test_continuing_education.py`
- `backend/TESTS/continuing_education_browser_fixture.py`
- `backend/TESTS/audit_continuing_education.py`
- `frontend/src/components/calidad/ContinuingEducation.tsx`
- `frontend/src/services/continuingEducation.ts`
- `frontend/tests/continuingEducation.test.mjs`
- `frontend/tests/continuingEducation.browser.mjs`
- `docs/auditorias/educacion_continua.md`
