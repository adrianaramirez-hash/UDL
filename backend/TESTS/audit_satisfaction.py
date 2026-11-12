"""Audit captured approved maps and normalized responses without source writes.

python -m backend.TESTS.audit_satisfaction <capture.json> <report.md>
"""
import json
from pathlib import Path
import sys

from backend.SERVICES.calidad.satisfaction import (
    resumen_actual, resumen_historico, catalogo, filtrar, _rows, _recommendation,
)


def audit(source, output):
    capture = json.loads(Path(source).read_text(encoding="utf-8"))
    now, old = capture["2026"], capture["2025"]
    current = resumen_actual(now["registros"], now["mapa"])
    history = resumen_historico(now["registros"], now["mapa"], old["registros"], old["mapa"])
    definitions, old_definitions = catalogo(now["mapa"]), catalogo(old["mapa"], completo=False)
    previous_recommendation = _recommendation(_rows(filtrar(old["registros"]), old_definitions.get("REC_UDL")), old_definitions.get("REC_UDL"))
    assert definitions["SAT_GENERAL"]["PREGUNTA_ID"] != definitions["REC_UDL"]["PREGUNTA_ID"]
    assert current["satisfaccion"]["id"] == history["satisfaccion"]["id"] == "SAT_GENERAL"
    assert current["recomendacion"]["id"] == history["recomendacion"]["id"] == "REC_UDL"
    assert not {"indice", "resultado_general", "nps"} & current.keys()
    without_recommendation = resumen_actual([r for r in now["registros"] if r.get("PREGUNTA_ID") != definitions["REC_UDL"]["PREGUNTA_ID"]], now["mapa"])
    assert without_recommendation["satisfaccion"] == current["satisfaccion"]
    recommendation, satisfaction = current["recomendacion"], current["satisfaccion"]
    assert recommendation["si"] + recommendation["no"] == recommendation["total_valido"]
    assert recommendation["si_pct"] == (round(recommendation["si"] * 100 / recommendation["total_valido"], 2) if recommendation["total_valido"] else None)
    fmt = lambda value: "—" if value is None else f"{value:.2f}"
    sat_history, rec_history = history["satisfaccion"], history["recomendacion"]
    lines = ["# Auditoría de Satisfacción y recomendación", "",
             f"Lectura de fuentes: {capture.get('fecha_captura', 'fecha no registrada')}. Captura: `{Path(source).name}`. Ámbito: Escolarizadas, todas las carreras y modalidades académicas; 2026 frente a 2025 de la misma población.", "",
             f"- Encuestas analizadas: **{current['encuestas_analizadas']:,}**. Unión de formularios con al menos una respuesta válida en cualquiera de los dos indicadores; los textos sin respuesta cuantitativa no inflan el contador.",
             "- Los indicadores permanecen separados. No existe un resultado general combinado en esta proyección.",
             "- **Recomendación NO se calcula como NPS.** Es el porcentaje de respuestas Sí sobre el total binario válido. No se resta el porcentaje No ni se aplican categorías de promotores, pasivos o detractores.", "",
             "## Satisfacción general", "",
             "| ID canónico | Respuestas válidas 2026 | Encuestas con respuesta válida | Resultado 2025 | Resultado 2026 | Delta | Unidad | Escala | Histórico |",
             "|---|---:|---:|---:|---:|---:|---|---|:---:|",
             f"| SAT_GENERAL | {satisfaction['n']} | {satisfaction['encuestas_validas']} | {fmt(sat_history['indice_2025'])} | {fmt(satisfaction['indice'])} | {fmt(sat_history['delta'])} | puntos de índice | {satisfaction['escala_id']} | {'Sí' if sat_history['comparable'] else 'No'} |", "",
             "Escala existente de cinco respuestas: Totalmente insatisfecho, Insatisfecho, Regularmente satisfecho, Satisfecho y Totalmente satisfecho. Se usa el índice normalizado 0–100 y la distribución del motor estable; no se modifica la escala.",
             f"Distribución 2026: favorable {fmt(satisfaction['favorable_pct'])}%; neutral {fmt(satisfaction['neutral_pct'])}%; desfavorable {fmt(satisfaction['desfavorable_pct'])}%.", "",
             "## Recomendación", "",
             "| Año | ID canónico | Respuestas Sí | Respuestas No | Total válido | Porcentaje Sí | Porcentaje No |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for year, result in [("2026", recommendation), ("2025", previous_recommendation)]:
        lines.append(f"| {year} | REC_UDL | {result['si']} | {result['no']} | {result['total_valido']} | {fmt(result['si_pct'])} | {fmt(result['no_pct'])} |")
    lines += ["", f"- Resultado comparable 2025: {fmt(rec_history['porcentaje_2025'])}%; resultado actual 2026: {fmt(recommendation['si_pct'])}%; delta: **{fmt(rec_history['delta_pp'])} puntos porcentuales**.",
              f"- Histórico aprobado de recomendación: {'Sí' if rec_history['comparable'] else 'No'}. Escala: {recommendation['escala_id']}.",
              "- Ausencia de respuestas o histórico se conserva como null; una tasa válida de 0% permanece visible. Vacíos y no aplicables no entran al denominador.", "",
              "## Trazabilidad de las equivalencias", "",
              "| Año | ID canónico | Pregunta aprobada | Columna de origen | Escala |",
              "|---|---|---|---|---|"]
    for year, data in [("2026", now), ("2025", old)]:
        for definition in catalogo(data["mapa"], completo=year == "2026").values():
            lines.append(f"| {year} | {definition['PREGUNTA_CANONICA_ID']} | {definition['PREGUNTA_ID']} | {definition.get('COLUMNA_ORIGEN', 'No registrada')} | {definition.get('ESCALA_ID', 'No registrada')} |")
    lines += ["", "Satisfacción 2025 ↔ Satisfacción 2026 usa exclusivamente SAT_GENERAL. Recomendación 2025 ↔ Recomendación 2026 usa exclusivamente REC_UDL. Ambos mapas y registros deben aprobar COMPARABLE_HISTORICO. No se usan nombres, posiciones, palabras similares ni coincidencias aproximadas.", "",
              "## Filtros", ""]
    for career, modality in [("Derecho", ""), ("", "Presencial"), ("Derecho", "Presencial")]:
        selected = resumen_actual(now["registros"], now["mapa"], career, modality)
        selected_history = resumen_historico(now["registros"], now["mapa"], old["registros"], old["mapa"], career, modality)
        assert selected["poblacion"] == selected_history["poblacion"]
        lines.append(f"- {career or 'Todas las carreras'} / {modality or 'Todas las modalidades'}: satisfacción {fmt(selected['satisfaccion']['indice'])}; recomendación {fmt(selected['recomendacion']['si_pct'])}%; encuestas {selected['encuestas_analizadas']}.")
    assert resumen_actual(now["registros"], now["mapa"]) == current
    lines += ["", "Quitar filtros restaura el global. Los motivos REC_PORQUE se muestran solo en el detalle de Recomendación; esta auditoría no incluye comentarios personales y no se muestra en producción. No se avanza a Centro de Idiomas ni Educación Continua.", ""]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(lines), encoding="utf-8")
    print(f"Satisfaccion: {fmt(satisfaction['indice'])}; recomendacion: {fmt(recommendation['si_pct'])}%; encuestas: {current['encuestas_analizadas']}.")


if __name__ == "__main__":
    audit(*sys.argv[1:])
