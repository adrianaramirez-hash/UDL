# Auditoría de Servicios académicos

Fuente: captura normalizada `udl-servicios-academicos-audit.json` (fecha de captura: no registrada en el archivo). Ámbito: todas las carreras y modalidades académicas de la captura 2026. Histórico 2025 del mismo ámbito. No representa una nueva lectura de las fuentes.

- Resultado general: **72.88**.
- Respuestas a criterios: **6,812**; suma de respuestas válidas de los doce reactivos.
- Encuestas analizadas: **635**; filas únicas con al menos una respuesta válida en esta sección.
- Criterios mostrados: 12; con histórico válido: 12; sin histórico: 0.
- Comparativo global: 70.10 → 72.88; variación 2.78 puntos.
- La unión utiliza únicamente PREGUNTA_CANONICA_ID, aprobación COMPARABLE_HISTORICO de ambos mapas y los indicadores de los registros normalizados. No usa texto, palabras clave ni aproximaciones.
- Ambos índices históricos se calculan sobre la misma intersección con respuestas válidas en ambos años, mediante el motor estable de KPIs.

| Criterio | ID canónico | Respuestas actuales | Encuestas con respuesta válida | Resultado actual | Histórico | 2025 | 2026 comparable | Delta (pts) | Estado esperado |
|---|---|---:|---:|---:|:---:|---:|---:|---:|---|
| Profesores / Asesores | ACA_PROFESORES | 628 | 628 | 82.64 | Sí | 79.87 | 82.64 | 2.77 | Resultado + barras + delta |
| Plan de estudios | ACA_PLAN_ESTUDIOS | 631 | 631 | 77.97 | Sí | 79.77 | 77.97 | -1.80 | Resultado + barras + delta |
| Clases presenciales | ACA_CLASES_PRESENCIALES | 606 | 606 | 82.47 | Sí | 81.51 | 82.47 | 0.96 | Resultado + barras + delta |
| Clases en Zoom | ACA_CLASES_ZOOM | 521 | 521 | 70.49 | Sí | 65.22 | 70.49 | 5.27 | Resultado + barras + delta |
| Prácticas académicas | ACA_PRACTICAS | 365 | 365 | 78.22 | Sí | 75.17 | 78.22 | 3.05 | Resultado + barras + delta |
| Certificaciones | ACA_CERTIFICACIONES | 423 | 423 | 72.58 | Sí | 71.25 | 72.58 | 1.33 | Resultado + barras + delta |
| Plataforma SEAC en general | ACA_SEAC_GENERAL | 631 | 631 | 66.09 | Sí | 62.59 | 66.09 | 3.50 | Resultado + barras + delta |
| SEAC · Apps móviles | ACA_SEAC_APPS | 564 | 564 | 60.64 | Sí | 56.08 | 60.64 | 4.56 | Resultado + barras + delta |
| SEAC · Facilidad para ingresar | ACA_SEAC_INGRESO | 632 | 632 | 69.54 | Sí | 64.54 | 69.54 | 5.00 | Resultado + barras + delta |
| SEAC · Trámites administrativos | ACA_SEAC_TRAMITES | 601 | 601 | 68.05 | Sí | 65.62 | 68.05 | 2.43 | Resultado + barras + delta |
| SEAC · Tareas y trabajos | ACA_SEAC_TAREAS | 624 | 624 | 71.23 | Sí | 67.87 | 71.23 | 3.36 | Resultado + barras + delta |
| Contenidos de las Aulas Virtuales | ACA_AULAS_VIRTUALES | 586 | 586 | 74.66 | Sí | 71.72 | 74.66 | 2.94 | Resultado + barras + delta |

## Denominadores y nombres

- Prácticas académicas tiene 365 respuestas válidas; no se divide entre las 635 encuestas analizadas de la sección. Vacíos y no aplicables mantienen la exclusión del normalizador estable.
- Clases presenciales y Zoom usan sus propios denominadores. Ausencia de respuestas se conserva como null, nunca como cero.
- Los seis conceptos SEAC/Aulas Virtuales permanecen separados por su ID canónico; no se crea un promedio sustituto de SEAC.
- Los nombres de presentación son exactamente los doce solicitados. No se modifica el mapa de origen, sus IDs ni las equivalencias.

## Filtro real: Derecho / Presencial

- Resultado general: 65.56; respuestas a criterios: 291; encuestas analizadas: 26; criterios conservados: 12.
- Histórico de la misma población: 69.48 → 65.56.
- Al quitar carrera/modalidad se recupera exactamente el resumen global. No hay fallback a datos globales cuando la población no tiene histórico.

La auditoría es documentación de desarrollo y no se muestra en producción. La interacción se verifica por separado con `node --test tests/academicServices.browser.mjs` desde frontend, usando poblaciones de prueba calculadas por el motor estable. No se continúa a Instalaciones.
