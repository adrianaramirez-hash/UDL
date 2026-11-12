"""Read-only audit of an existing normalized Ambiente escolar capture.

python -m backend.TESTS.audit_school_environment <capture.json> <report.md>
"""
import json
from pathlib import Path
import sys

from backend.SERVICES.calidad.school_environment import (
    CRITERIOS, resumen_actual, resumen_historico, filtrar,
    _incidence, _incidence_rows, _incidence_definition,
)


def audit(source, output):
    capture = json.loads(Path(source).read_text(encoding="utf-8"))
    now, before = capture["2026"], capture["2025"]
    current = resumen_actual(now["registros"], now["mapa"])
    previous = {"incidencia": _incidence(_incidence_rows(filtrar(before["registros"]), _incidence_definition(before["mapa"])))}
    history = resumen_historico(now["registros"], now["mapa"], before["registros"], before["mapa"])
    historical = {row["id"]: row for row in history["criterios"]}
    assert len(current["criterios"]) == len(CRITERIOS) == 3
    assert len({row["id"] for row in current["criterios"]}) == 3
    assert all(row["nombre"].strip() for row in current["criterios"])
    assert sum(row["n"] for row in current["criterios"]) == current["respuestas"]
    # Independently remove incidence and prove that indices, criterion counts
    # and distributions are unchanged. Survey union may legitimately change.
    incidence_ids = {r["PREGUNTA_ID"] for r in now["mapa"] if r.get("PREGUNTA_CANONICA_ID") == "AMB_PROBLEMA"}
    without_incidence = resumen_actual([r for r in now["registros"] if r.get("PREGUNTA_ID") not in incidence_ids], now["mapa"])
    for key in ("indice", "respuestas", "criterios", "distribucion"):
        assert without_incidence[key] == current[key], "La incidencia alteró los criterios cuantitativos"
    fmt = lambda value: "—" if value is None else f"{value:.2f}"
    delta = lambda old, new: None if old is None or new is None else round(new - old, 2)
    lines = ["# Auditoría de Ambiente escolar", "",
             f"Lectura de fuentes: {capture.get('fecha_captura', 'fecha no registrada')}. Captura: `{Path(source).name}`. Ámbito: Escolarizadas, todas las carreras y modalidades académicas, 2026 frente a 2025 de la misma población.", "",
             f"- Resultado general: **{fmt(current['indice'])}**.",
             f"- Respuestas a criterios: **{current['respuestas']:,}**.",
             f"- Encuestas analizadas: **{current['encuestas_analizadas']:,}**.",
             f"- Criterios con histórico: {len(history['ids_comparables'])}; sin histórico: {3 - len(history['ids_comparables'])}.",
             f"- Comparativo global: {fmt(history['indice_2025'])} → {fmt(history['indice_2026'])}; delta {fmt(delta(history['indice_2025'], history['indice_2026']))} puntos de índice.",
             "- La incidencia NO entra al Resultado general, a Respuestas a criterios ni a la distribución cuantitativa. Se verificó retirando sus registros: el resultado, respuestas, tres criterios y distribución permanecen idénticos.",
             "- Encuestas analizadas es la unión de filas de formulario con al menos una respuesta cuantitativa válida o una respuesta binaria válida de incidencia. No cuenta formularios con solo texto abierto ni suma los denominadores de las preguntas.",
             "- Cada criterio utiliza exclusivamente sus respuestas válidas; vacíos y no aplicables no son cero. Se preserva el motor estable de índices y sus pesos.", "",
             "| Nombre | ID canónico | Respuestas actuales | Encuestas con respuesta válida | Resultado 2026 | Resultado 2025 | Delta | Histórico | Estado visual |",
             "|---|---|---:|---:|---:|---:|---:|:---:|---|"]
    for row in current["criterios"]:
        old, new = historical[row["id"]]["indice_2025"], historical[row["id"]]["indice_2026"]
        state = "Sin respuestas en este periodo" if row["indice"] is None else "Sin comparable histórico" if old is None else "Barras 2025/2026 y delta"
        lines.append(f"| {row['nombre']} | {row['id']} | {row['n']} | {row['encuestas_validas']} | {fmt(row['indice'])} | {fmt(old)} | {fmt(delta(old, new))} | {'Sí' if old is not None else 'No'} | {state} |")
    incidence_history = history["incidencia"]
    lines += ["", "## Incidencia reportada · indicador independiente", "",
              "| Año | Respuestas Sí | Respuestas No | Total válido | Porcentaje Sí | Porcentaje No |",
              "|---|---:|---:|---:|---:|---:|"]
    for year, result in [("2026", current), ("2025", previous)]:
        incidence = result["incidencia"]
        lines.append(f"| {year} | {incidence['si']} | {incidence['no']} | {incidence['total_valido']} | {fmt(incidence['si_pct'])} | {fmt(incidence['no_pct'])} |")
    lines += ["", f"- Histórico aprobado disponible: {'Sí' if incidence_history['comparable'] else 'No'}.",
              f"- Porcentaje comparable 2025: {fmt(incidence_history['porcentaje_2025'])}%; 2026: {fmt(incidence_history['porcentaje_2026'])}%; delta: **{fmt(incidence_history['delta_pp'])} puntos porcentuales**.",
              "- Los porcentajes son tasas de respuestas Sí/No sobre su total válido. No constituyen puntajes de índice. Ausencia de respuestas se conserva como null, mientras que una tasa válida de 0% permanece visible.", "",
              "## Equivalencias y población", "",
              "| Año | ID canónico | Pregunta aprobada | Columna de origen |",
              "|---|---|---|---|"]
    for year, data in [("2026", now), ("2025", before)]:
        for row in data["mapa"]:
            if row.get("PREGUNTA_CANONICA_ID") in {*CRITERIOS, "AMB_PROBLEMA"}:
                lines.append(f"| {year} | {row['PREGUNTA_CANONICA_ID']} | {row['PREGUNTA_ID']} | {row.get('COLUMNA_ORIGEN', 'No registrada')} |")
    lines += ["", "Solo se usan IDs canónicos aprobados y COMPARABLE_HISTORICO en ambos mapas y registros. El global comparable usa la misma intersección de criterios válidos en ambos años. No se usan nombres similares, posiciones ni aproximaciones.", ""]
    for career, modality in [("Derecho", ""), ("", "Presencial"), ("Derecho", "Presencial")]:
        selected = resumen_actual(now["registros"], now["mapa"], career, modality)
        selected_history = resumen_historico(now["registros"], now["mapa"], before["registros"], before["mapa"], career, modality)
        assert selected["poblacion"] == selected_history["poblacion"]
        assert len(selected["criterios"]) == 3
        lines.append(f"- {career or 'Todas las carreras'} / {modality or 'Todas las modalidades'}: resultado {fmt(selected['indice'])}; respuestas {selected['respuestas']}; encuestas {selected['encuestas_analizadas']}; incidencia {fmt(selected['incidencia']['si_pct'])}%; delta de incidencia {fmt(selected_history['incidencia']['delta_pp'])} puntos porcentuales.")
    assert resumen_actual(now["registros"], now["mapa"]) == current
    lines += ["", "Quitar filtros restaura el global. Los comentarios y descripciones permanecen en el detalle contextual de la aplicación; esta auditoría no incluye textos personales y no se muestra en producción. No se avanza a Satisfacción y recomendación.", ""]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(lines), encoding="utf-8")
    print(f"Ambiente: {fmt(current['indice'])}; {current['respuestas']} respuestas; {current['encuestas_analizadas']} encuestas; incidencia {fmt(current['incidencia']['si_pct'])}%.")


if __name__ == "__main__":
    audit(*sys.argv[1:])
