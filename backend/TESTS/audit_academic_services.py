"""Generate the requested audit from captured existing map/normalized responses.

Usage: python -m backend.TESTS.audit_academic_services <capture.json> <report.md>
"""
import json
from pathlib import Path
import sys
from backend.SERVICES.calidad.academic_services import CRITERIOS, resumen_actual, resumen_historico


def audit(source, output):
    capture = json.loads(Path(source).read_text(encoding="utf-8"))
    now, before = capture["2026"], capture["2025"]
    current = resumen_actual(now["registros"], now["mapa"])
    historical = resumen_historico(now["registros"], now["mapa"], before["registros"], before["mapa"])
    old = {row["id"]: row for row in historical["criterios"]}
    global_delta = (historical["indice_2026"] - historical["indice_2025"]
                    if historical["indice_2025"] is not None and historical["indice_2026"] is not None else None)
    assert len(current["criterios"]) == len(CRITERIOS) == 12
    assert len({row["id"] for row in current["criterios"]}) == 12
    assert sum(row["n"] for row in current["criterios"]) == current["respuestas"]
    fmt = lambda x: "—" if x is None else f"{x:.2f}"
    rows = []
    for criterion in current["criterios"]:
        previous, comparable = old[criterion["id"]]["indice_2025"], old[criterion["id"]]["indice_2026"]
        delta = round(comparable - previous, 2) if previous is not None and comparable is not None else None
        state = "Sin respuestas en este periodo" if criterion["indice"] is None else "Sin comparable histórico" if previous is None else "Resultado + barras + delta"
        rows.append(f"| {criterion['nombre']} | {criterion['id']} | {criterion['n']} | {criterion['encuestas_validas']} | {fmt(criterion['indice'])} | {'Sí' if previous is not None else 'No'} | {fmt(previous)} | {fmt(comparable)} | {fmt(delta)} | {state} |")
    selected = resumen_actual(now["registros"], now["mapa"], "Derecho", "Presencial")
    selected_history = resumen_historico(now["registros"], now["mapa"], before["registros"], before["mapa"], "Derecho", "Presencial")
    assert selected["poblacion"] == selected_history["poblacion"]
    assert selected["respuestas"] <= current["respuestas"]
    assert len(selected["criterios"]) == 12
    assert resumen_actual(now["registros"], now["mapa"]) == current
    lines = [
        "# Auditoría de Servicios académicos", "",
        f"Fuente: captura normalizada `{Path(source).name}` (fecha de captura: {capture.get('fecha_captura', 'no registrada en el archivo')}). Ámbito: todas las carreras y modalidades académicas de la captura 2026. Histórico 2025 del mismo ámbito. No representa una nueva lectura de las fuentes.", "",
        f"- Resultado general: **{fmt(current['indice'])}**.",
        f"- Respuestas a criterios: **{current['respuestas']:,}**; suma de respuestas válidas de los doce reactivos.",
        f"- Encuestas analizadas: **{current['encuestas_analizadas']}**; filas únicas con al menos una respuesta válida en esta sección.",
        f"- Criterios mostrados: 12; con histórico válido: {len(historical['ids_comparables'])}; sin histórico: {12 - len(historical['ids_comparables'])}.",
        f"- Comparativo global: {fmt(historical['indice_2025'])} → {fmt(historical['indice_2026'])}; variación {fmt(global_delta)} puntos.",
        "- La unión utiliza únicamente PREGUNTA_CANONICA_ID, aprobación COMPARABLE_HISTORICO de ambos mapas y los indicadores de los registros normalizados. No usa texto, palabras clave ni aproximaciones.",
        "- Ambos índices históricos se calculan sobre la misma intersección con respuestas válidas en ambos años, mediante el motor estable de KPIs.", "",
        "| Criterio | ID canónico | Respuestas actuales | Encuestas con respuesta válida | Resultado actual | Histórico | 2025 | 2026 comparable | Delta (pts) | Estado esperado |",
        "|---|---|---:|---:|---:|:---:|---:|---:|---:|---|", *rows, "",
        "## Denominadores y nombres", "",
        f"- Prácticas académicas tiene {next(row['n'] for row in current['criterios'] if row['id'] == 'ACA_PRACTICAS')} respuestas válidas; no se divide entre las {current['encuestas_analizadas']} encuestas analizadas de la sección. Vacíos y no aplicables mantienen la exclusión del normalizador estable.",
        "- Clases presenciales y Zoom usan sus propios denominadores. Ausencia de respuestas se conserva como null, nunca como cero.",
        "- Los seis conceptos SEAC/Aulas Virtuales permanecen separados por su ID canónico; no se crea un promedio sustituto de SEAC.",
        "- Los nombres de presentación son exactamente los doce solicitados. No se modifica el mapa de origen, sus IDs ni las equivalencias.", "",
        "## Filtro real: Derecho / Presencial", "",
        f"- Resultado general: {fmt(selected['indice'])}; respuestas a criterios: {selected['respuestas']}; encuestas analizadas: {selected['encuestas_analizadas']}; criterios conservados: 12.",
        f"- Histórico de la misma población: {fmt(selected_history['indice_2025'])} → {fmt(selected_history['indice_2026'])}.",
        "- Al quitar carrera/modalidad se recupera exactamente el resumen global. No hay fallback a datos globales cuando la población no tiene histórico.", "",
        "La auditoría es documentación de desarrollo y no se muestra en producción. La interacción se verifica por separado con `node --test tests/academicServices.browser.mjs` desde frontend, usando poblaciones de prueba calculadas por el motor estable. No se continúa a Instalaciones.", "",
    ]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(lines), encoding="utf-8")
    print(f"Auditoría: 12 criterios, {fmt(current['indice'])} / {current['respuestas']} respuestas / {current['encuestas_analizadas']} encuestas.")


if __name__ == "__main__":
    audit(*sys.argv[1:])
