# Pasada visual final de Encuesta de Calidad

Fecha: 12 de noviembre de 2026. Alcance: exclusivamente presentación frontend. No se avanzó al PDF.

## Presentación final

- Asistentes/coordinadores utiliza `GlobalSectionSummary`, tarjetas de criterio, barras y distribuciones compartidas. Muestra Resultado general, Respuestas a criterios, Encuestas analizadas y Responsables evaluados. Conserva los cuatro criterios, los mismos responsables, las asignaciones sin respuestas y las vistas opcionales por carrera/persona.
- Su comparativo global conserva los valores históricos existentes y los presenta con barras y delta junto al título. Los detalles nominales conservan `Sin comparable histórico`; no se construyeron equivalencias ni históricos nuevos.
- Dirección mantiene su catálogo, resultados y filtros. Sus detalles tradicionales usan las tarjetas compartidas. Se aclara visualmente que los filtros locales acotan el listado y que el resumen superior conserva su alcance global.
- Recomendación e Incidencia muestran barras 2025/2026 en escala fija 0–100%, valores porcentuales y deltas en `pp`, con tooltip «puntos porcentuales». Los índices mantienen `pts`. Incidencia conserva un delta de color neutro; una subida no se presenta como una mejora automática.
- Los comparativos generales también usan barras compartidas y deltas cercanos al indicador, sin una tarjeta independiente dedicada solo a la variación.
- Se ocultó el flotante de SIA únicamente en esta página y se retiraron sus paneles inactivos, botones redundantes y placeholders grandes. En los detalles con comentarios queda una sola nota discreta: `Análisis cualitativo con SIA · En construcción`.
- Los comentarios permanecen en sus detalles contextuales. Se retiró el bloque global redundante de Asistentes. La pestaña general de comentarios contiene una indicación breve para encontrarlos en cada detalle.
- Se unificaron espaciado, tamaños de controles, foco accesible, opción `Todas / Vista global`, alineación numérica, estados vacíos y skeletons. Los filtros opcionales tienen menor jerarquía que el panorama general.
- Idiomas y Educación Continua conservan únicamente sus resultados actuales. Se retiraron los avisos históricos repetidos en cada indicador; no se añadieron históricos ni índices globales. Idiomas aprovecha dos columnas en ancho intermedio sin estirar una tarjeta con un único idioma.

## Revisión de los nueve módulos

| Módulo | Global y filtros | Desktop 1920 | Laptop 1366 | Intermedio 900 | Presentación |
|---|---|---|---|---|---|
| Dirección | Verificados | Correcto | Correcto | Correcto | Resumen, criterios, listado y detalle compartidos |
| Asistentes/coordinadores | Verificados | Correcto | Correcto | Correcto | Global → responsables → detalle opcional |
| Servicios institucionales | Verificados | Correcto | Correcto | Correcto | Controles y estados unificados |
| Servicios académicos | Verificados | Correcto | Correcto | Correcto | Hereda los ajustes compartidos |
| Instalaciones | Verificados | Correcto | Correcto | Correcto | Hereda los ajustes compartidos |
| Ambiente escolar | Verificados | Correcto | Correcto | Correcto | Comparativo porcentual separado del índice |
| Satisfacción y recomendación | Verificados | Correcto | Correcto | Correcto | `pts` y `pp` diferenciados |
| Centro de Idiomas | Verificados | Correcto | Correcto | Correcto | Contexto compacto, sin histórico añadido |
| Educación Continua | Verificados | Correcto | Correcto | Correcto | Seis indicadores independientes, sin histórico añadido |

La inspección de Chrome utilizó la captura de datos validada en la auditoría anterior con transporte simulado. Se comprobaron 27 combinaciones de módulo/ancho y se generaron 35 capturas temporales, incluidos detalles. No se detectaron desbordamientos horizontales, filtros fuera del viewport ni flotantes cubriendo contenido. Se verificó el retorno a global tras seleccionar carrera y la presencia de una sola nota de SIA por detalle con comentarios. Las regresiones existentes cubren además modalidad, filtros especializados, errores y respuestas tardías.

## Conservación funcional

- **Cero cambios funcionales.** Se conservaron APIs, servicios frontend/backend, fuentes, cálculos, filtros, equivalencias, normalización, agrupaciones, asociaciones, catálogos e históricos.
- Se compararon los hashes de todos los archivos backend y servicios frontend contra la instantánea tomada antes de esta pasada: permanecen iguales.
- La comparación por AST confirmó que las consultas actuales/históricas, los manejadores de filtros superiores y las expresiones existentes de selección de población e histórico permanecen iguales.
- Las funciones de PDF y CSV permanecen idénticas por comparación de AST. No se generaron ni modificaron exportaciones.
- La limpieza de SIA afecta solamente a la interfaz inactiva de esta página. No se modificaron componentes ni lógica global de SIA en otras pantallas.

## Archivos frontend modificados

Presentación:

- `frontend/src/pages/EncuestaCalidad.tsx`
- `frontend/src/components/common/CalidadMetrics.tsx`
- `frontend/src/components/calidad/qualityPresentation.css` (nuevo, estilos limitados a esta página)
- `frontend/src/components/calidad/DirectionResults.tsx`
- `frontend/src/components/calidad/InstitutionalServices.tsx`
- `frontend/src/components/calidad/SchoolEnvironment.tsx`
- `frontend/src/components/calidad/Satisfaction.tsx`
- `frontend/src/components/calidad/LanguageCenter.tsx`
- `frontend/src/components/calidad/ContinuingEducation.tsx`

Pruebas: se actualizaron expectativas de presentación; se conservaron las comprobaciones de resultados, filtros y transporte.

- `frontend/tests/institutionalServices.test.mjs`
- `frontend/tests/schoolEnvironment.test.mjs`
- `frontend/tests/schoolEnvironment.browser.mjs`
- `frontend/tests/satisfaction.test.mjs`
- `frontend/tests/satisfaction.browser.mjs`
- `frontend/tests/languageCenter.test.mjs`
- `frontend/tests/languageCenter.browser.mjs`
- `frontend/tests/continuingEducation.test.mjs`
- `frontend/tests/continuingEducation.browser.mjs`

## Observaciones conservadas

No se identificaron bloqueos visuales en los anchos revisados. La etiqueta heredada de rol de Claudia sigue mencionando Licenciaturas Ejecutivas aunque sus asignaciones son de Posgrado. Proviene del catálogo y se conservó, conforme a la restricción de no cambiar datos, asociaciones ni catálogos.

Los filtros de Dirección mantienen su alcance original sobre el listado; no se igualó su comportamiento al de otros módulos. La interfaz ahora lo explica. Tampoco se añadió un índice global a Idiomas, Educación Continua o Satisfacción/recomendación.

## Validación final

- Frontend completo: **133/133 pruebas aprobadas**, incluidos navegador, filtros, detalles, históricos, errores y respuestas tardías.
- Regresión backend: **110/110 pruebas aprobadas**, sin modificar archivos backend.
- Vite: **build aprobado**. Se conserva la advertencia de bundle superior a 500 kB.
- TypeScript: **6 diagnósticos heredados y 0 nuevos**. Cuatro de los diez originales desaparecieron como consecuencia de retirar la interfaz inactiva de SIA y sus referencias. No se repararon errores ajenos al alcance.
- Los tests visuales se actualizaron para comprobar las barras porcentuales y sus valores, la opción global unificada y la ausencia de avisos históricos repetidos en los módulos que solo tienen medición actual. Se conservaron las comprobaciones numéricas y de transporte.
- Integridad: 17 archivos frontend preexistentes modificados y un CSS nuevo. Ningún cambio backend o en servicios frontend respecto de la instantánea de inicio. `git diff --check` no presenta errores de espacios; Git únicamente informa la conversión habitual LF/CRLF de la página en Windows.

**VISUALMENTE LISTO PARA PDF.** No se avanzó a la implementación ni generación del PDF.
