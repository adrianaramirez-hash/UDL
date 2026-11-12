# Etapa M · PDF Informe de resultados

Fecha de validación: 12 de noviembre de 2026. Implementación exclusiva del PDF. La interfaz queda congelada.

## Documentos entregados

| Caso real · instrumento 2026 | Páginas | Encuestas analizadas | Resultado general |
|---|---:|---:|---:|
| Escolarizadas · Todas las carreras · Todas las modalidades | 14 | 638 | 72.11 |
| Escolarizadas · Derecho · Presencial | 11 | 27 | 65.45 |

- `artifacts/calidad/Informe_de_resultados_Escolarizadas_Global_2026.pdf`
- `artifacts/calidad/Informe_de_resultados_Escolarizadas_Derecho_Presencial_2026.pdf`
- `artifacts/calidad/validacion_informes.json`: filtros, cifras, hashes SHA-256 y comprobaciones individuales de todas las páginas.

Ambos archivos se abrieron y renderizaron correctamente con un lector PDF independiente, sin reparación ni contraseña. Se revisaron visualmente todas las páginas y las páginas afectadas por los ajustes finales. La validación automática comprueba todas las páginas de la versión final, sus márgenes, encabezados, numeración, filtros y presencia de cifras.

## Estructura final

| Contenido | Global | Derecho · Presencial |
|---|---|---|
| Título, filtros activos, panorama y gráfica de secciones | 1 | 1 |
| Dirección | 2 | 2 |
| Asistentes/coordinadores | 3–5 | 2 |
| Servicios institucionales | 6–7 | 3–4 |
| Servicios académicos | 8–9 | 5–6 |
| Instalaciones y equipo tecnológico | 10–11 | 7–8 |
| Ambiente escolar e Incidencia independiente | 12 | 9 |
| Satisfacción y Recomendación independientes | 13 | 10 |
| Centro de Idiomas, solo medición actual | 13 | 10 |
| Educación Continua, solo medición actual | 14 | 11 |

El título y los metadatos del documento dicen **Informe de resultados**. Los encabezados repiten fuente, carrera, modalidad académica y periodo. Los pies indican página actual y total. Se utiliza A4 con gráficos vectoriales, escala fija 0–100, tipografía legible y bloques que no se cortan entre páginas.

Los criterios se ordenan de menor a mayor, dejando los que no tienen respuestas al final. Asistentes conserva las preguntas y los resultados existentes de cada responsable; presenta responsables alfabéticamente y ordena los cuatro criterios dentro de cada uno. No calcula un nuevo agregado de criterios ni ordena personas por desempeño.

Los comparativos proceden de los servicios aprobados y sus IDs canónicos. Se muestran valores 2025/2026 y deltas en `pts` o `pp`. Biblioteca y Cafetería permanecen separadas, incluidos sus antecedentes. La ausencia de datos utiliza `Sin respuestas en este periodo`; la ausencia de comparación válida utiliza `Sin comparable histórico`. Un cero válido se mantiene como 0.00.

Incidencia no forma parte del resultado de Ambiente escolar. Satisfacción y Recomendación no se mezclan en un índice. Idiomas y Educación Continua no reciben históricos ni índices agregados forzados. Su lugar en la gráfica general remite al detalle por criterio.

En cada módulo se reserva únicamente el espacio discreto `Resumen cualitativo` / `Análisis cualitativo con SIA · En construcción`. No se exportan comentarios crudos, clasificaciones heurísticas, conclusiones automáticas, tablas gigantes, rankings de personas ni NPS.

## Origen y comprobación de cifras

Se obtuvo una captura real completa de las fuentes: **37,597 registros normalizados de 2026** y **94,437 de 2025**, con sus mapas aprobados y catálogos. No se escribió en las fuentes.

La primera descarga global desde la API con acceso directo a Google Sheets respondió 200. Una consulta posterior alcanzó el límite temporal de lecturas de Sheets (429). Para reproducir ambos casos de forma consistente sin consumir más cuota, se ejecutó la ruta HTTP real con `TestClient`, sustituyendo únicamente el transporte por la captura real completa. No se simularon puntuaciones ni resultados: las mismas funciones de producción procesaron ambas poblaciones.

La auditoría reproducible ejecuta el motor y los servicios originales sobre esa captura, verifica los resultados y denominadores de todos los criterios proyectados y compara los históricos contra los servicios originales. Derecho/Presencial se aplica a los registros de ambos años; no existe fallback global. La identidad resuelta desde un programa seleccionado también limita ambos años a los mismos campos de carrera/modalidad.

El botón existente lee los filtros superiores y los controles opcionales de carrera/modalidad que ya están en pantalla, sin cambiar sus valores ni el estado del dashboard. Si están activos, los filtros especializados de idioma, curso y participación se aplican únicamente a su módulo y quedan impresos allí. La ruta corresponde al instrumento de nueve módulos de Escolarizadas 2026; 2025 se utiliza como antecedente comparable, no se fuerza al instrumento anterior dentro de esta estructura.

La ruta PDF responde con un error específico y sin documento parcial cuando la fuente no está disponible. No elimina filtros ni consulta una población global de reemplazo.

## Problemas encontrados y corregidos

1. El exportador anterior utilizaba agregaciones y comparaciones por texto heredadas. Se retiró ese exportador y sus auxiliares exclusivos; el PDF nuevo consume los servicios y equivalencias ya aprobados.
2. Dos etiquetas de rol de Claudia podían generar una entrada adicional sin respuestas. El informe conserva sus resultados existentes y evita duplicar una persona ya observada al añadir responsables sin datos. No se modificaron roles, asociaciones ni catálogos.
3. Las barras de valor cero producían un remate gráfico al dibujar un rectángulo redondeado de ancho cero. El PDF mantiene el valor 0.00 y dibuja únicamente el fondo cuando el relleno es cero.
4. El caso Derecho dejaba una página casi vacía para un único responsable. Ese bloque comparte página con Dirección, sin cortar criterios, encabezados ni el espacio cualitativo. El caso quedó en 11 páginas.
5. Los textos largos se envuelven según el ancho disponible. Se mantienen juntos el título cualitativo y su nota; no hay páginas vacías ni pies o gráficas fuera de los márgenes.

## Archivos modificados

Integración existente, limitada al PDF:

- `backend/GATEWAY/main.py`: registro de la ruta exclusiva de PDF.
- `frontend/src/pages/EncuestaCalidad.tsx`: sustitución del exportador PDF y retiro de auxiliares utilizados exclusivamente por el exportador anterior. El componente React del dashboard y el CSV permanecen idénticos.

Archivos nuevos:

- `backend/GATEWAY/results_report_api.py`
- `backend/SERVICES/calidad/results_report.py`
- `backend/SERVICES/calidad/results_report_pdf.py`
- `backend/TESTS/test_results_report_pdf.py`
- `backend/TESTS/audit_results_report.py`
- `frontend/src/services/resultsReportPdf.ts`
- `frontend/tests/resultsReportPdf.test.mjs`
- Los dos PDFs, el manifiesto de validación y este informe.

## Reproducción e integridad

Auditoría sobre una captura real completa:

```text
python -m backend.TESTS.audit_results_report captura.json artifacts/calidad --renderer-path directorio_temporal_pymupdf
```

ReportLab ya era una dependencia del proyecto. El lector adicional de validación se instaló únicamente en la carpeta temporal, sin cambiar dependencias del proyecto.

La comparación de la instantánea de inicio confirma que los únicos archivos existentes modificados son la integración de rutas y la página que contenía el exportador. El componente `EncuestaCalidad`, sus estados, consultas, filtros, JSX y cálculos permanecen idénticos por comparación de AST/texto normalizado. También permanecen idénticas las funciones de CSV y sus auxiliares. No se modificaron módulos visuales, estilos, motores de cálculo, normalización, metodología, equivalencias, agrupaciones ni comparativos del dashboard.

## Pruebas finales

- **Backend: 121/121 aprobadas**, incluidas once nuevas pruebas del PDF: proyección exacta, nueve módulos, orden de criterios, misma población en ambos años, identidad de programa, falta de histórico, cero válido, separación de indicadores, privacidad del modelo, adjunto HTTP y errores de fuente/instrumento.
- **Frontend: 137/137 aprobadas** en la corrida completa final, incluidas cuatro nuevas pruebas de descarga, filtros activos, MIME, nombre del archivo y ausencia de fallback ante un fallo.
- **Vite build aprobado**; conserva la advertencia heredada de tamaño de bundle.
- **TypeScript: cero diagnósticos nuevos.** Permanecen cinco de los seis heredados. El sexto pertenecía al código del exportador PDF anterior que se retiró; no se corrigieron los otros diagnósticos.
- **25 páginas finales validadas**, sin páginas vacías, texto fuera de márgenes, encabezados o pies incorrectos. Se verificaron valores actuales e históricos, denominadores, títulos y filtros de todas las páginas; se renderizaron e inspeccionaron ambos documentos.
- Se detuvo el servidor temporal de pruebas de PDF. No se alteraron otros procesos del sistema ni se escribieron datos en las fuentes.

**PDF LISTO**
