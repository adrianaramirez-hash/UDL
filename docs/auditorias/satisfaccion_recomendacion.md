# Auditoría de Satisfacción y recomendación

Lectura de fuentes: 2026-11-11. Captura: `udl-satisfaccion-audit.json`. Ámbito: Escolarizadas, todas las carreras y modalidades académicas; 2026 frente a 2025 de la misma población.

- Encuestas analizadas: **638**. Unión de formularios con al menos una respuesta válida en cualquiera de los dos indicadores; los textos sin respuesta cuantitativa no inflan el contador.
- Los indicadores permanecen separados. No existe un resultado general combinado en esta proyección.
- **Recomendación NO se calcula como NPS.** Es el porcentaje de respuestas Sí sobre el total binario válido. No se resta el porcentaje No ni se aplican categorías de promotores, pasivos o detractores.

## Satisfacción general

| ID canónico | Respuestas válidas 2026 | Encuestas con respuesta válida | Resultado 2025 | Resultado 2026 | Delta | Unidad | Escala | Histórico |
|---|---:|---:|---:|---:|---:|---|---|:---:|
| SAT_GENERAL | 638 | 638 | 66.52 | 70.53 | 4.01 | puntos de índice | SATISFACCION_5 | Sí |

Escala existente de cinco respuestas: Totalmente insatisfecho, Insatisfecho, Regularmente satisfecho, Satisfecho y Totalmente satisfecho. Se usa el índice normalizado 0–100 y la distribución del motor estable; no se modifica la escala.
Distribución 2026: favorable 70.22%; neutral 23.20%; desfavorable 6.58%.

## Recomendación

| Año | ID canónico | Respuestas Sí | Respuestas No | Total válido | Porcentaje Sí | Porcentaje No |
|---|---|---:|---:|---:|---:|---:|
| 2026 | REC_UDL | 569 | 69 | 638 | 89.18 | 10.82 |
| 2025 | REC_UDL | 1569 | 185 | 1754 | 89.45 | 10.55 |

- Resultado comparable 2025: 89.45%; resultado actual 2026: 89.18%; delta: **-0.27 puntos porcentuales**.
- Histórico aprobado de recomendación: Sí. Escala: BINARIA_SI_NO.
- Ausencia de respuestas o histórico se conserva como null; una tasa válida de 0% permanece visible. Vacíos y no aplicables no entran al denominador.

## Trazabilidad de las equivalencias

| Año | ID canónico | Pregunta aprobada | Columna de origen | Escala |
|---|---|---|---|---|
| 2026 | SAT_GENERAL | ESC_SAT_GENERAL | DX | SATISFACCION_5 |
| 2026 | REC_UDL | ESC_REC_UDL | DY | BINARIA_SI_NO |
| 2025 | SAT_GENERAL | ESC25_SAT_GENERAL | BC | SATISFACCION_5 |
| 2025 | REC_UDL | ESC25_REC_UDL | BD | BINARIA_SI_NO |

Satisfacción 2025 ↔ Satisfacción 2026 usa exclusivamente SAT_GENERAL. Recomendación 2025 ↔ Recomendación 2026 usa exclusivamente REC_UDL. Ambos mapas y registros deben aprobar COMPARABLE_HISTORICO. No se usan nombres, posiciones, palabras similares ni coincidencias aproximadas.

## Filtros

- Derecho / Todas las modalidades: satisfacción 65.18; recomendación 85.71%; encuestas 28.
- Todas las carreras / Presencial: satisfacción 68.76; recomendación 88.21%; encuestas 509.
- Derecho / Presencial: satisfacción 63.89; recomendación 85.19%; encuestas 27.

Quitar filtros restaura el global. Los motivos REC_PORQUE se muestran solo en el detalle de Recomendación; esta auditoría no incluye comentarios personales y no se muestra en producción. No se avanza a Centro de Idiomas ni Educación Continua.

## Validación de implementación · 11 de noviembre de 2026

- Backend: `python -m unittest discover -s backend/TESTS -p 'test_*.py'` — 79 pruebas aprobadas, incluidas 14 de esta sección.
- Frontend: `node --test --test-concurrency=1 tests/satisfaction.test.mjs tests/satisfaction.browser.mjs tests/schoolEnvironment.test.mjs tests/schoolEnvironment.browser.mjs tests/facilities.test.mjs tests/facilities.browser.mjs tests/academicServices.test.mjs tests/academicServices.browser.mjs tests/direction.test.mjs tests/institutionalServices.test.mjs` — 84 pruebas aprobadas, incluidas 15 de esta sección. Después del ajuste final de las etiquetas de conteo, las 15 pruebas de esta sección se repitieron y aprobaron.
- Navegador: global sin carrera, dos indicadores independientes, distribución Sí/No, filtros carrera/modalidad, recuperación del global, motivos únicamente en el detalle de Recomendación, estados vacíos, errores históricos independientes y descarte de respuestas de solicitudes anteriores. Usa poblaciones de prueba calculadas por el servicio y el motor estable.
- Históricos: pruebas con orden distinto, nombres engañosos, ausencia del histórico de un indicador y aprobación de ambos mapas y registros. Se confirma que satisfacción nunca recibe datos de recomendación ni a la inversa.
- Recomendación: prueba explícita de tres Sí y un No; resultado 75%, sin restar el porcentaje No ni calcular NPS. El delta se expresa en puntos porcentuales; satisfacción usa puntos de índice.
- Vite: `node node_modules/vite/bin/vite.js build` — paquete de producción generado; advertencia de tamaño de bundle superior a 500 kB.
- TypeScript: mismos 10 diagnósticos preexistentes, comparando archivos, códigos y mensajes completos; 0 nuevos. La compilación completa sigue pendiente de corregir esos errores previos fuera del alcance de esta sección.
- `git diff --check`: sin errores de espacios.
- Integridad: 110 archivos previos preservados byte por byte. Los únicos archivos anteriores modificados son los dos de integración listados abajo. Los bloques JSX de Dirección, Servicios institucionales, Servicios académicos, Instalaciones y Ambiente escolar son idénticos a la instantánea anterior a esta tarea. Normalización, metodología, equivalencias y exportaciones PDF/CSV no cambiaron.

## Archivos de esta implementación

Modificados:

- `backend/GATEWAY/main.py`: registro del router independiente.
- `frontend/src/pages/EncuestaCalidad.tsx`: integración de la sección y exclusión exclusiva de sus resúmenes y comentarios genéricos redundantes.

Agregados:

- `backend/GATEWAY/satisfaction_api.py`
- `backend/SERVICES/calidad/satisfaction.py`
- `backend/TESTS/test_satisfaction.py`
- `backend/TESTS/satisfaction_browser_fixture.py`
- `backend/TESTS/audit_satisfaction.py`
- `frontend/src/components/calidad/Satisfaction.tsx`
- `frontend/src/services/satisfaction.ts`
- `frontend/tests/satisfaction.test.mjs`
- `frontend/tests/satisfaction.browser.mjs`
- `docs/auditorias/satisfaccion_recomendacion.md`
