# Auditoría de Dirección — Etapa B

Base: `5450b4c7aa54c6987cda4018f6373ca65beb14f9`. Lectura de las fuentes reales el 5 de noviembre de 2026.

Ámbito: Escolarizadas, periodo actual 2026, sin filtro de servicio; histórico 2025 con el mismo ámbito. Lecturas del endpoint estable `preview/resumen`; histórico con `historico=true`. No se recuperó código de la rama de respaldo.

## Resultado

- 26 carreras/programas distintos, presentados en 28 combinaciones carrera/modalidad.
- 7 poblaciones con respuestas actuales y 21 sin respuestas de Dirección.
- 16 poblaciones con histórico disponible y 12 sin histórico.
- 0 nombres vacíos, 0 claves duplicadas y 0 poblaciones con métricas ambiguas.
- 335 encuestas con respuestas válidas de Dirección; 1340 respuestas entre los cuatro criterios.
- Índice global: 69.13 en 2025 y 67.03 en 2026; variación −2.10 puntos.

Los conteos con/sin respuestas y con/sin histórico se hacen por combinación carrera/modalidad. Derecho y Pedagogía tienen modalidades separadas; una misma carrera puede tener resultados en una modalidad y carecer de ellos en otra.

## Presentación de identidades

El endpoint estable entrega 30 filas. En la presentación se unifican Informática / Tecnologías de la Información (alias ya presente en el normalizador estable) y Maestría:Educación Especial / Maestría en Educación Especial (formato del prefijo del grado). En estos dos pares no hay respuestas actuales: no se suman ni recalculan índices. No se cambia el catálogo ni la normalización del backend.

Las claves toleran acentos, mayúsculas y espacios y usan las variantes de Cine/Tecnologías ya reconocidas por la versión estable. Modalidades distintas conservan claves distintas. Si dos filas con resultados tienen la misma identidad, se marca la ambigüedad en lugar de promediar métricas. El histórico nunca incorpora carreras al listado actual.

## Auditoría por carrera y modalidad

«Encuestas actuales» cuenta encuestas únicas con respuestas válidas de Dirección, no la suma de los cuatro reactivos. Histórico «Sí» significa un índice 2025 disponible para esa misma población.

| Carrera / programa | Modalidad | Encuestas actuales | Histórico | Estado visual esperado |
|---|---|---:|:---:|---|
| Actuación | Presencial | 0 | Sí | Sin respuestas en este periodo |
| Administración de Empresas | Presencial | 0 | Sí | Sin respuestas en este periodo |
| Cine y TV Digital | Presencial | 67 | Sí | Con resultados |
| Comunicación Multimedia | Presencial | 36 | Sí | Con resultados |
| Contaduría | Presencial | 0 | Sí | Sin respuestas en este periodo |
| Creación y Gestión de Empresas Turísticas | Presencial | 0 | Sí | Sin respuestas en este periodo |
| Derecho | Ejecutiva | 0 | No | Sin respuestas en este periodo |
| Derecho | Presencial | 27 | Sí | Con resultados |
| Diseño de Modas | Presencial | 0 | Sí | Sin respuestas en este periodo |
| Diseño Gráfico | Presencial | 85 | Sí | Con resultados |
| Entrenamiento Deportivo | Presencial | 0 | No | Sin respuestas en este periodo |
| Finanzas | Presencial | 0 | Sí | Sin respuestas en este periodo |
| Gastronomía | Presencial | 0 | Sí | Sin respuestas en este periodo |
| Maestría en Administración de Negocios (MBA) | Posgrado | 0 | No | Sin respuestas en este periodo |
| Maestría en Administración de Recursos Humanos | Posgrado | 0 | No | Sin respuestas en este periodo |
| Maestría en Derecho Corporativo | Posgrado | 0 | No | Sin respuestas en este periodo |
| Maestría en Educación Especial | Posgrado | 0 | No | Sin respuestas en este periodo |
| Maestría en Finanzas | Posgrado | 0 | No | Sin respuestas en este periodo |
| Maestría en Odontología Legal y Forense | Posgrado | 0 | No | Sin respuestas en este periodo |
| Maestría en Psicoterapia Familiar | Posgrado | 0 | No | Sin respuestas en este periodo |
| Maestría: Dirección de Recursos Humanos | Posgrado | 0 | No | Sin respuestas en este periodo |
| Maestría:Gestión de Tecnologías de la Información | Posgrado | 0 | No | Sin respuestas en este periodo |
| Mercadotecnia | Presencial | 0 | Sí | Sin respuestas en este periodo |
| Nutrición | Presencial | 60 | Sí | Con resultados |
| Pedagogía | Ejecutiva | 1 | No | Con resultados · Sin comparable histórico |
| Pedagogía | Presencial | 59 | Sí | Con resultados |
| Psicología | Presencial | 0 | Sí | Sin respuestas en este periodo |
| Tecnologías de la Información | Presencial | 0 | Sí | Sin respuestas en este periodo |

## Cuatro criterios globales

| Criterio | Respuestas 2026 | 2025 | 2026 | Variación |
|---|---:|---:|---:|---:|
| Disponibilidad | 335 | 69.24 | 66.64 | -2.60 pts |
| Atención a necesidades | 335 | 69.61 | 67.76 | -1.85 pts |
| Solución de problemas | 335 | 68.73 | 66.87 | -1.86 pts |
| Seguimiento al desempeño docente | 335 | 68.96 | 66.87 | -2.09 pts |

## Verificación

- `node --test tests/direction.test.mjs`: 12 pruebas de regresión sobre datos sintéticos y renderizado HTML de los componentes.
- Auditoría adicional con datos reales: 28 claves únicas, sin nombres vacíos ni conflictos; las 10 carreras solicitadas presentes.
- TypeScript: sin errores nuevos; permanecen 11 errores heredados. El error de estado `sin_encuestas` del bloque sustituido deja de existir.
- Compilación Vite correcta. Esta comprobación de empaquetado no sustituye la comprobación de tipos, que conserva los errores heredados.
- Backend, normalización, catálogo, asignaciones, clasificación de comentarios, fórmulas, equivalencias, PDF y CSV sin modificaciones.
- Pendiente: validación visual interactiva del usuario antes de aplicar los componentes a cualquier otra sección.

## Revisión visual sugerida

1. Abrir Por sección → Dirección: ver índice global, 1340 respuestas y cuatro criterios sin elegir carrera.
2. Abrir Diseño Gráfico: ver sus cuatro criterios, barras 2025/2026 y distribuciones.
3. Abrir Gastronomía o Actuación: conservar la tarjeta y mostrar «Sin respuestas en este periodo», incluso cuando haya histórico.
4. Abrir Pedagogía Ejecutiva: mostrar resultado 50.00 y «Sin comparable histórico».
5. Usar Todas / Vista global en ambos selectores: recuperar todo el listado sin alterar el resumen global.
6. Confirmar que los detalles por director y los comentarios siguen disponibles.

La etapa B termina aquí. No se aplicaron los componentes a otras secciones.
