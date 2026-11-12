# Auditoría global de Encuesta de Calidad

**Fecha:** 11 de noviembre de 2026. **Dictamen: NO LISTO PARA ETAPA PDF.**

Se revisaron los nueve módulos solicitados sin corregir código, datos, cálculos, asociaciones ni metodología. Solo se agrega este reporte. Se conservan los cambios locales que ya existían al comenzar la auditoría.

Alcance de datos: instrumento **Escolarizadas**, 2026 global y 2025 comparable, leídos de las fuentes configuradas. Se contrastaron los catálogos activos, las proyecciones de cada sección y la población Derecho / Presencial. No se extrapolan estos resultados a los instrumentos Virtuales o Prepa. Los filtros de idioma, curso y participación se comprobaron además mediante las pruebas existentes.

Se revisó la página completa en un navegador aislado, con sus componentes y CSS de producción y una captura de las respuestas reales. Para esa inspección se omitió únicamente el cuerpo `respuestas_export` de los mocks: no se descargó ni trabajó el PDF/CSV. El fallo histórico se simuló con respuestas HTTP 503. Las reproducciones de concurrencia y fallback ejecutan las funciones reales extraídas del archivo, con respuestas controladas, sin editar la aplicación. Las capturas originales permanecieron intactas.

## A. Correcto · elementos congelables en el alcance verificado

### Navegación inicial

Los **nueve módulos muestran información global sin exigir carrera**. No se encontró una excepción a esta regla inicial. En Asistentes el detalle por carrera/persona es opcional; en Dirección se puede consultar el resumen y las cuatro preguntas antes de escoger una población. Los módulos nuevos envían inicialmente filtros vacíos.

### Comprobación por módulo

| Módulo | Confirmación con fuentes actuales | Límites del congelamiento |
|---|---|---|
| Dirección de carrera | Global 67.03; cuatro criterios. La presentación tiene 28 poblaciones, 26 carreras/programas distintos y 21 poblaciones sin respuestas. No hay nombres vacíos, duplicados visibles ni conflictos de identidad. El histórico no crea carreras: la lista parte del endpoint actual. | La integridad del universo de carreras **no** está resuelta: ver C4. La deduplicación de los 30 registros de origen a 28 poblaciones respeta los alias ya existentes. |
| Asistentes / coordinadores | Global disponible: 398 encuestas analizadas, cuatro criterios y 11 responsables con datos. El campo interno `evaluaciones` del resumen cuenta formularios únicos; la etiqueta visible principal es «Encuestas analizadas». | No congelar todavía su presentación, comentarios y exposición de asignaciones: B y C5–C6. |
| Servicios institucionales | **11 servicios**, todos con nombre e ID distintos; el servicio es **Paramédico**. 5,087 respuestas a criterios y 633 encuestas, correctamente diferenciadas. Global 74.54; comparable 70.33 → 74.54, con 11 servicios en la intersección. | El panel específico usa la proyección aprobada, no el fallback general de C1. |
| Servicios académicos | **12 criterios**. 6,812 respuestas a criterios y 635 encuestas. Prácticas académicas usa **365 respuestas válidas**, resultado 78.22, sin dividir entre las 635 encuestas. Global 72.88; comparable 70.10 → 72.88. | No se detectó mezcla entre los seis conceptos SEAC/Aulas Virtuales. |
| Instalaciones y equipo tecnológico | **10 criterios**. Biblioteca: `INS_BIBLIOTECA`, 442 respuestas, 72.57, histórico 69.68. Cafetería: `INS_CAFETERIA`, 463 respuestas, 61.07, histórico 57.64. Son independientes. Global 67.86; comparable 65.24 → 67.86. | No se detectó cruce entre Biblioteca y Cafetería. |
| Ambiente escolar | **Tres criterios cuantitativos**; global 79.83 y 1,851 respuestas a criterios. La incidencia está separada y excluida del resultado y distribución cuantitativos. Incidencia 12.43% → 10.19%; delta **−2.24 puntos porcentuales**. | Las tasas no se presentan como puntos de índice. |
| Satisfacción y recomendación | Indicadores independientes: satisfacción 66.52 → 70.53, **+4.01 puntos**; recomendación 89.45% → 89.18%, **−0.27 puntos porcentuales**. No se calcula NPS ni un resultado combinado en el panel específico. «¿Por qué?» aparece solo en el detalle de recomendación. | El tab general «Comparativos» sigue siendo una interfaz heredada distinta: C1–C3 y B. |
| Centro de Idiomas | **Tres criterios especializados**; 637 encuestas y 282 respuestas a criterios. Uso 14.76% separado. Idioma y curso son descriptivos, con denominador propio de 94 respuestas. Profesor se presenta como dato descriptivo, sin ranking. No hay histórico especializado ni delta. | La pregunta general 2025 no sustituye los tres criterios 2026. |
| Educación Continua | Participación 4.87% separada; **seis indicadores**, 637 encuestas y 1,398 respuestas a criterios. Recomendación es porcentaje Sí, **no NPS**. Medios son descriptivos sobre 31 respuestas. No existe histórico especializado ni delta. | Se preservan 31 respuestas para experiencia/recomendación y 637 para las dos preguntas de conocimiento. No se inventan calificaciones de experiencia para quienes no participaron. |

### Comparativos, unidades y estados correctos

- En los paneles específicos de servicios institucionales, académicos, instalaciones, ambiente y satisfacción/recomendación se verificó que actual e histórico reciban **Derecho / Presencial en ambos años**. Las proyecciones no hacen fallback a la población global cuando falta histórico de ese filtro.
- Dirección usa claves estables de carrera/modalidad y los cuatro IDs aprobados. Los módulos recientes usan IDs canónicos y banderas de comparabilidad; no semejanza textual.
- Índices: deltas en `pts`. Incidencia y recomendación porcentual: deltas en **puntos porcentuales**. Participaciones no tienen un histórico inventado ni un delta de índice.
- Los estados nulos de los paneles específicos se mantienen distintos de un cero válido. **Karla tiene 0.00 con una encuesta y cuatro respuestas reales**: en esa captura no es un vacío convertido en cero.
- Centro de Idiomas y Educación Continua especializados conservan **«Sin comparable histórico»**, sin barras 2025/2026 ni equivalencias nuevas.
- Los componentes nuevos mantienen los resultados actuales ante error exclusivamente histórico; las pruebas también comprueban filtros y recuperación del global.

### Asociaciones de Asistentes / coordinadores

Se distinguen tres evidencias: catálogo activo del MASTER, reglas confirmadas del motor y poblaciones con respuestas en la captura actual. Ausencia de respuestas no prueba una reasignación.

| Persona | Asociación verificada | Observación |
|---|---|---|
| **Claudia Diaz** | Las asignaciones activas de esta persona son Maestrías/Posgrado; la población con respuestas es Odontología Legal y Forense, Posgrado, seis encuestas. La restricción del motor la limita a Posgrado/Maestrías. | Su etiqueta de rol actual menciona «Equipo de Licenciaturas Ejecutivas / Coordinadora», aunque la población es Posgrado; inconsistencia de texto, ver B. No confundirla con Claudia Vázquez Barajas, que corresponde a Dirección. |
| **Grecia Ceballos Ibarra** | Conserva Administración de Empresas, Contaduría, Finanzas y Mercadotecnia en Presencial, tanto en catálogo como en reglas. | Solo Administración y Finanzas tienen respuestas actuales. La interfaz de «Carreras asignadas» no expone las cuatro: C5. |
| **Karla Sánchez Cruz** | Psicología / Presencial en catálogo y datos actuales. | El 0.00 es un resultado válido, no ausencia. |
| **Oscar Corona Amezcua** | La regla confirmada conserva Desarrollo del Potencial Humano y Organizacional / Posgrado. | No aparece en el catálogo activo de personas leído ni en los resultados nominales actuales; no puede confirmarse su exposición en pantalla. No se cambió la regla. |
| **Lourdes Quiroga** | La regla confirmada conserva Psicoterapia / Posgrado. | No aparece en el catálogo activo leído ni en los resultados nominales actuales; misma limitación de verificación. |
| **Militza Fermoso** | La regla confirmada conserva Odontología Legal y Forense / Posgrado. | No aparece en el catálogo activo leído ni en los resultados nominales actuales; misma limitación de verificación. |
| **Alejandra Romero Peña** | Conserva la función ejecutiva de Responsable de titulación; catálogo con seis carreras ejecutivas. | Los datos actuales solo muestran Derecho y Pedagogía; el detalle no lista todas las asignaciones sin respuestas: C5. |
| **Brenda Ponce Pérez** | Conserva la función ejecutiva de Responsable de evaluación; catálogo con seis carreras ejecutivas. | Misma limitación de exposición de asignaciones que Alejandra. |
| **Balbina García Arroyo** | Actuación / Presencial en catálogo, regla y datos; 16 encuestas actuales. | La fuente usa **Balbina**, no Albina. No se creó un alias ni se modificó el nombre. |

Las reglas están en `backend/SERVICES/calidad/kpi_service.py:674` y `:924`. La evidencia no permite afirmar que todas las asociaciones estén completamente expuestas en la interfaz; ver C5.

## B. Ajustes visuales menores · sin aplicar

1. **Asistentes mantiene otra presentación.** Usa tarjetas superiores independientes, niveles de color, pestañas por carrera/persona, textos «Promedio», «Criterios evaluados» y `n =`. Los módulos posteriores agrupan contadores en `GlobalSectionSummary`, usan `MetricComparison` y etiquetas «Resultado actual» / «Respuestas a criterios». La diferencia se confirmó en pantalla a 1440×1100; no se rediseñó.
2. **Estados y etiquetas heredados.** En Asistentes aparecen «Sin respuestas», «Sin antecedente comparable en 2025» y «Sin antecedente comparable por persona», frente a los dos textos estándar. En detalles tradicionales de Dirección quedan «Promedio», «Promedio global» y `n =`. Son variaciones de presentación; no demuestran por sí solas un error de cálculo.
3. **Rol de Claudia.** El título «Equipo de Licenciaturas Ejecutivas / Coordinadora» no coincide con su población Posgrado. La ubicación observada es correcta; la etiqueta merece conciliación, sin reasignar automáticamente a la persona.
4. **Botones SIA redundantes.** Hay un botón flotante activo «Asistente SIA · En construcción» que abre un panel conceptual. Su textarea y botón de consulta están deshabilitados: no se observó una consulta IA activa. En Asistentes existe además «Preguntar a SIA sobre esta sección», también activo para abrir ese panel. La regla solicitada permite únicamente el estado discreto «Análisis cualitativo con SIA · En construcción».
5. **El flotante tapa contenido.** En la inspección de Dirección cubre parcialmente la tarjeta derecha de criterios. Es un ajuste de posición/presencia pendiente, no una corrección aplicada.
6. **Incidencia y recomendación porcentual.** Sus unidades y deltas son correctos y cercanos al título, pero muestran filas de porcentajes 2025/2026, sin barras históricas. El comparativo general, por su parte, repite números además de las barras y presenta «Variación» sin flechas junto al indicador. Es una diferencia respecto del patrón visual solicitado; no hay evidencia de una unidad errónea en esos dos paneles específicos.
7. **Filtros con alcance visual distinto.** En Dirección los selectores locales filtran el listado; el resumen superior permanece global y el resultado de la población se consulta al abrir el detalle. Los selectores de los módulos recientes recalculan su resumen. Conviene explicitar esta diferencia para no interpretar el resumen global como un resultado filtrado.
8. **Residuos conceptuales.** La pestaña general «Preguntas abiertas» conserva una descripción de futuro sobre búsquedas/temas. La estructura de títulos, espaciado y placeholders no coincide totalmente con los detalles nuevos.
9. **Fragilidad del arnés de navegador en Windows.** La corrida completa tuvo un `EPERM` al eliminar el perfil temporal de Chrome de Centro de Idiomas, después de que sus seis casos interactivos pasaran. La reejecución aislada pasó 7/7. No se alteró el arnés.

Referencias: `frontend/src/pages/EncuestaCalidad.tsx:2834`, `:3079`, `:3155`, `:3283`, `:4242`, `:4632`, `:4652`, `:4772`; `frontend/src/components/calidad/Satisfaction.tsx:25` y `SchoolEnvironment.tsx:12`.

## C. Problemas funcionales y bloqueos · sin corregir

### C1 · El comparativo general puede perder la población seleccionada

**Prioridad alta.** `cargarHistorico()` pide primero el histórico con modalidad/servicio. Si no hay encuestas o preguntas, ejecuta `request(false)` y elimina ambos filtros.

Se reprodujo ejecutando la función real extraída de [EncuestaCalidad.tsx](../../frontend/src/pages/EncuestaCalidad.tsx), líneas 1451–1493, con respuestas controladas:

1. Solicitud: `periodo=2025&historico=true&modalidad=Escolarizadas&servicio=Programa sin comparable` → respuesta vacía.
2. Segunda solicitud: `periodo=2025&historico=true` → devuelve el año global.

El resultado queda disponible para compararse contra una selección distinta. La regla exige la misma población en ambos años. **No afecta el diseño de los endpoints específicos nuevos, que sí conservan el filtro; persiste en el tab general «Comparativos» y en consumidores heredados.**

### C2 · Una respuesta antigua puede sobrescribir el filtro actual

**Prioridad alta.** El efecto principal `cargarDatos()` no tiene cancelación, comprobación de vigencia ni identidad de solicitud antes de `setSummary(data)`.

Reproducción con la función real, sin editarla: se inició filtro A y después B; B respondió primero y A después. El orden de escritura fue **B → A** y la muestra final fue **A**, aunque el filtro vigente era B. Referencia: `EncuestaCalidad.tsx:1424–1449`.

El efecto histórico general tampoco descarta respuestas de un efecto anterior. Los componentes nuevos sí tienen protección por identidad de filtro y limpieza de efecto, pero reciben sus parámetros y universo de sección desde esta página compartida. No se puede congelar el comportamiento global de cambios rápidos de filtros.

### C3 · El fallo histórico general queda presentado como carga permanente

**Prioridad alta.** En el catch se ponen ambos históricos a `null`; la vista interpreta esos mismos nulos como «Cargando comparativo histórico...», sin estado de error diferenciado.

Se reprodujo en el navegador con la página completa: las dos peticiones históricas devolvieron HTTP 503; después de finalizar ambas, el tab «Comparativos» continuaba mostrando «Cargando». Los datos actuales sí habían cargado. Referencias: `EncuestaCalidad.tsx:1496–1512` y `:4369–4373`.

### C4 · Dirección no expone todas las poblaciones del catálogo activo

**Prioridad alta.** Aunque no hay duplicados ni nombres vacíos visibles y 21 poblaciones sin respuestas se conservan, faltan estas **seis combinaciones carrera/modalidad** presentes en el catálogo activo leído, al comparar con las claves de presentación ya existentes:

- Maestría en Desarrollo del Potencial Humano y Organizacional (Coaching) / Posgrado.
- Maestría en Psicoterapia Psicoanalítica / Posgrado.
- Administración de Empresas / Ejecutiva.
- Contaduría / Ejecutiva.
- Mercadotecnia / Ejecutiva.
- Tecnologías de la Información / Ejecutiva.

No se añadieron al resultado ni se inventaron equivalencias. El motor combina un catálogo de Dirección codificado con modalidades observadas y poblaciones presentes en respuestas; la preservación de carreras sin respuesta no cubre todo el catálogo activo. Referencias: `backend/SERVICES/calidad/kpi_service.py:2706–2770`; `frontend/src/components/calidad/directionModel.ts:45`.

La auditoría existente `direction-audit.mjs` pasó para las 28 poblaciones recibidas; comprueba una lista mínima de nombres, pero no esta diferencia contra el catálogo completo. Por eso pasar la prueba no confirma cobertura total de carreras.

### C5 · «Carreras asignadas» muestra carreras con respuestas, no todas las asignadas

**Prioridad media.** `staffOverviewAssignments` se construye desde `personas_por_carrera_modalidad`, que contiene grupos con respuestas. Se usa después bajo el título «Carreras asignadas».

Ejemplos actuales: Grecia tiene cuatro carreras de EDN en catálogo/reglas, pero el detalle obtiene únicamente Administración y Finanzas; Alejandra y Brenda tienen seis carreras ejecutivas en catálogo, pero el detalle obtiene Derecho y Pedagogía. Las demás asignaciones no se han demostrado eliminadas del motor: **se omiten en esa presentación**. Referencias: `EncuestaCalidad.tsx:1761–1783` y `:3252–3273`.

Además, Oscar, Lourdes y Militza conservan reglas confirmadas en el código, pero no aparecen en el catálogo activo leído ni en la salida nominal actual. La auditoría no puede certificar desde pantalla sus asociaciones. Requiere conciliar catálogo, reglas y exposición de personas sin datos; no reasignarlas por inferencia ni modificar cálculos en esta etapa.

### C6 · Siguen visibles comentarios globales y conclusiones heurísticas no solicitadas

**Prioridad media.** Se confirmó en el navegador:

- Asistentes muestra «Respuestas abiertas por sección» sin abrir detalle, con «Positivos explícitos» y «Oportunidades / incidencias».
- Su detalle por persona muestra «Interpretación general», producida por `getStaffInterpretation()`, además del placeholder de resumen.
- El detalle tradicional por director mantiene «Lectura del resultado» y «Resumen preliminar de comentarios» producido con reglas y palabras clave.

No se presenta esto como una llamada real a IA: son reglas/heurísticas existentes. Sin embargo, incumple la instrucción de mantener únicamente el estado discreto SIA y comentarios contextuales sin clasificación/conclusiones todavía. Referencias: `EncuestaCalidad.tsx:998`, `:1092`, `:3361`, `:4023–4046`, `:4209–4294`.

Los detalles de los módulos nuevos conservan el patrón contextual y no necesitan que se cambien sus cálculos para resolver este problema compartido.

### C7 · La compilación completa sigue bloqueada por TypeScript

**Bloqueo de construcción heredado; sin corregir.** `node node_modules/typescript/bin/tsc -b --pretty false` termina con código 1 y **10 errores**:

- `EncuestaCalidad.tsx:74`: tipo `QualitativeSummaryItem` sin usar.
- `EncuestaCalidad.tsx:837`, `:1827`, `:2016`, `:2138`: incompatibilidad de `polaridad_base: string | null` con tipos que esperan `string | undefined`.
- `EncuestaCalidad.tsx:1347`, `:1352`, `:1358`, `:1617`: setters/variables sin usar.
- `frontend/src/services/calidadService.ts:269`: `CalidadSection` no definido.

El script oficial `build` ejecuta `tsc -b && vite build`; por ello **Vite aprobado por separado no equivale a build completo aprobado**.

### Dependencia heredada que debe considerarse antes del PDF

`findHistoricalComparableQuestion()` todavía admite coincidencia por etiqueta y similitud de palabras con umbral 0.3 (`EncuestaCalidad.tsx:1141–1204`). Se conserva en cálculos auxiliares heredados de comparabilidad y en funciones de exportación.

Los paneles específicos nuevos no usan ese helper para sus comparativos, y esta auditoría no demuestra un cruce actual entre sus indicadores. No obstante, no puede certificarse «sin equivalencias por semejanza» para todo el archivo. **Se reporta como dependencia pendiente, sin intervenir en PDF/CSV ni crear equivalencias nuevas.**

### Regresiones ejecutadas y límites de evidencia

| Verificación | Resultado |
|---|---|
| Backend: `python -m unittest discover -s backend/TESTS -p 'test_*.py'` | **104/104 aprobadas**. |
| Frontend: todos los archivos `*.test.mjs` y `*.browser.mjs`, con `--test-concurrency=1` | Primera corrida: **110/111 aprobadas**. El único fallo fue limpieza del perfil temporal de Chrome (`EPERM`) en Centro de Idiomas, no una aserción de datos; sus seis interacciones habían pasado. |
| Reejecución de `node --test tests/languageCenter.browser.mjs` | **7/7 aprobadas**, sin cambiar código ni pruebas. |
| Auditor existente de Dirección con la captura actual | Aprobado: 28 poblaciones recibidas. No detecta las seis omisiones frente al catálogo activo. |
| Vite: `node node_modules/vite/bin/vite.js build` | Aprobado. Advertencia de bundle superior a 500 kB. |
| TypeScript | **10 errores heredados; compilación fallida**. No corregidos. |
| Revisión aislada de página completa | Confirmó global de Asistentes y Dirección, placeholders, botones SIA, comentarios/clasificaciones de Asistentes y carga indefinida tras fallo histórico simulado. |
| Reproducciones de funciones reales | Confirmaron fallback a histórico global y sobrescritura de un filtro nuevo por una respuesta antigua. |
| Integridad y espacios | **139 archivos de código/pruebas idénticos** a la instantánea inicial; `git diff --check` sin errores. Solo se agrega este reporte. |

Los tests existentes cubren ampliamente las proyecciones y componentes recientes; no cubren todas las condiciones de la página compartida ni certifican por sí solos el catálogo completo de Dirección o las nueve asociaciones nominales solicitadas. No se ejecutaron descargas ni se cambió el exportador.

**NO LISTO PARA ETAPA PDF.** Primero deben resolverse, en una etapa expresamente autorizada, los problemas de población/concurrencia, el estado histórico de error, las omisiones y exposición de catálogos y las salidas cualitativas heredadas; también debe acordarse el tratamiento de los errores TypeScript y de la dependencia de comparabilidad por texto. Tras eso procede repetir la validación. No se implementó ninguna de estas correcciones y no se avanzó al PDF.
