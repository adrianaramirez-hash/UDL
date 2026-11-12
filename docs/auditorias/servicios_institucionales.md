# Auditoría de Servicios institucionales

Lectura real: 5 de noviembre de 2026. Ámbito: Escolarizadas, 2026, todas las carreras y modalidades académicas. Histórico 2025 del mismo ámbito.

- Servicios actuales: 11; con histórico válido: 11; sin histórico: 0.
- Índice actual: 74.54; respuestas válidas: 5087.
- Intersección global aprobada con datos en ambos años: 11 servicios. 2025: 70.33; 2026: 74.54; delta: +4.21 puntos.
- Los índices delegan en el motor estable: igual peso por pregunta. La distribución global agrupa respuestas válidas y se muestra separada del índice.
- La unión usa exclusivamente PREGUNTA_CANONICA_ID y COMPARABLE_HISTORICO de ambos mapas y registros; sin coincidencias por texto.
- Centro de Idiomas y Educación Continua, presentes en la sección 2025, no pertenecen a la intersección actual y no se incorporan al índice ni al listado 2026.
- El mapa actual denomina al ID SER_PARAMEDICO «Servicio institucional 09». Se conserva esa denominación; no se modifica catálogo ni normalización.

| Servicio | Actual | Respuestas actuales | Histórico | 2025 | 2026 comparable | Delta (pts) | Estado visual esperado |
|---|---:|---:|:---:|---:|---:|---:|---|
| Admisiones | 77.12 | 424 | Sí | 72.03 | 77.12 | 5.09 | Resultado actual + barras + delta |
| Desarrollo Estudiantil | 73.74 | 358 | Sí | 70.00 | 73.74 | 3.74 | Resultado actual + barras + delta |
| Servicios escolares | 74.38 | 565 | Sí | 69.86 | 74.38 | 4.52 | Resultado actual + barras + delta |
| Becas | 75.04 | 584 | Sí | 71.84 | 75.04 | 3.20 | Resultado actual + barras + delta |
| Cobranzas | 72.35 | 575 | Sí | 67.87 | 72.35 | 4.48 | Resultado actual + barras + delta |
| Vigilancia | 82.59 | 583 | Sí | 79.73 | 82.59 | 2.86 | Resultado actual + barras + delta |
| Intendencia (limpieza) | 85.52 | 587 | Sí | 81.49 | 85.52 | 4.03 | Resultado actual + barras + delta |
| Mantenimiento de instalaciones | 73.58 | 583 | Sí | 69.78 | 73.58 | 3.80 | Resultado actual + barras + delta |
| Servicio institucional 09 | 66.70 | 289 | Sí | 57.83 | 66.70 | 8.87 | Resultado actual + barras + delta |
| Bolsa de Trabajo | 67.78 | 239 | Sí | 63.98 | 67.78 | 3.80 | Resultado actual + barras + delta |
| Titulaciones | 71.17 | 300 | Sí | 69.24 | 71.17 | 1.93 | Resultado actual + barras + delta |

## Filtro real: Derecho / Presencial

- Resultado actual: 67.45; respuestas: 205; servicios conservados: 11.
- Histórico de la misma población: 68.01 → 67.45.
- Ninguna lectura amplía el filtro a la población global cuando faltan respuestas.

## Verificación

- Pruebas unitarias del cálculo, catálogo, intersección canónica y filtros por carrera/modalidad.
- Pruebas de renderizado: global sin carrera, todos los servicios, estados ausentes, barras y deltas, carga y error histórico independiente.
- Pruebas del cliente: ambos endpoints reciben exactamente la misma carrera, modalidad y programa.
- Dirección, componentes compartidos, normalización, equivalencias, motor KPI, catálogo y exportaciones sin cambios de comportamiento.
- Esta auditoría es documentación de desarrollo; no aparece en producción.
