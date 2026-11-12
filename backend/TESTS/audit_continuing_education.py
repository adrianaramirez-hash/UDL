"""Read-only audit of specialized 2026 indicators; no historical scoring or NPS.

python -m backend.TESTS.audit_continuing_education <capture.json> <report.md>
"""
import json
from pathlib import Path
import sys
from backend.SERVICES.calidad.continuing_education import CRITERIOS, catalogo, resumen_actual


def audit(source, output):
    capture = json.loads(Path(source).read_text(encoding="utf-8"))
    now = capture["2026"]
    current = resumen_actual(now["registros"], now["mapa"])
    definitions = catalogo(now["mapa"])
    assert len(current["criterios"]) == len(CRITERIOS) == 6
    assert len({r["id"] for r in current["criterios"]}) == 6
    assert all(r["nombre"].strip() and r["historico_disponible"] is False for r in current["criterios"])
    assert all("indice_2025" not in r and "delta" not in r for r in current["criterios"])
    assert sum(r["n"] for r in current["criterios"]) == current["respuestas"]
    assert "indice" not in current and "nps" not in current
    context_ids = {r["PREGUNTA_ID"] for key, r in definitions.items() if key in {"EC_USO", "EC_MEDIO"}}
    without_context = resumen_actual([r for r in now["registros"] if r.get("PREGUNTA_ID") not in context_ids], now["mapa"])
    assert without_context["criterios"] == current["criterios"]
    assert without_context["respuestas"] == current["respuestas"]
    fmt = lambda value: "—" if value is None else f"{value:.2f}"
    participation = current["participacion"]
    recommendation = next(r["binaria"] for r in current["criterios"] if r["id"] == "EC_RECOMIENDA")
    assert recommendation["si"] + recommendation["no"] == recommendation["total_valido"]
    assert recommendation["si_pct"] == (round(recommendation["si"] * 100 / recommendation["total_valido"], 2) if recommendation["total_valido"] else None)
    lines = ["# Auditoría de Educación Continua", "",
             f"Lectura de fuentes: {capture.get('fecha_captura', 'fecha no registrada')}. Captura: `{Path(source).name}`. Ámbito: Escolarizadas, 2026, todas las carreras y modalidades; sin filtro de participación.", "",
             f"- Encuestas analizadas: **{current['encuestas_analizadas']:,}**.",
             f"- Respuestas a criterios: **{current['respuestas']:,}**.",
             "- Encuestas analizadas es la unión de formularios con al menos una respuesta válida en cualquiera de los seis indicadores, participación, medio o comentario aprobado. Los datos descriptivos no se convierten en puntajes.",
             "- Respuestas a criterios suma los seis indicadores cuantitativos: tres escalas de experiencia y tres preguntas binarias. Sus denominadores son independientes: conocer la oferta no requiere haber participado.",
             "- No se publica un resultado general agregado. Se retiraron participación y medios y se comprobó que los seis resultados, denominadores y distribuciones permanecen idénticos.", "",
             "## Participación · indicador descriptivo independiente", "",
             "| Respuestas Sí | Respuestas No | Total válido | Participación Sí (%) | Participación No (%) |",
             "|---:|---:|---:|---:|---:|",
             f"| {participation['si']} | {participation['no']} | {participation['total_valido']} | {fmt(participation['si_pct'])} | {fmt(participation['no_pct'])} |", "",
             "## Seis indicadores especializados", "",
             "| Indicador | ID canónico | Pregunta aprobada | Columna | Escala | Respuestas válidas | Encuestas válidas | Resultado 2026 | Unidad | Histórico | Estado visual |",
             "|---|---|---|---|---|---:|---:|---:|---|:---:|---|"]
    for row in current["criterios"]:
        state = "Sin respuestas en este periodo" if row["resultado"] is None else "Resultado actual + distribución + Sin comparable histórico"
        unit = "% Sí" if row["unidad"] == "porcentaje" else "Índice 0–100"
        lines.append(f"| {row['nombre']} | {row['id']} | {row['pregunta_id']} | {definitions[row['id']].get('COLUMNA_ORIGEN', 'No registrada')} | {row['escala_id']} | {row['n']} | {row['encuestas_validas']} | {fmt(row['resultado'])} | {unit} | No | {state} |")
    lines += ["", "**Los criterios especializados 2026 no tienen comparación histórica válida con la pregunta general 2025.**", "",
              "La implementación no solicita ni une datos históricos; no utiliza posiciones, palabras similares, coincidencias aproximadas ni equivalencias nuevas. No muestra barras 2025/2026 ni calcula deltas.", ""]
    for reference in capture.get("2025", {}).get("referencias_generales", []):
        lines.append(f"- Referencia metodológica existente: `{reference['PREGUNTA_ID']}` / `{reference['PREGUNTA_CANONICA_ID']}` en `{reference['SECCION_ID']}`, COMPARABLE_HISTORICO={reference.get('COMPARABLE_HISTORICO')}. Su resultado no se usa como comparable de ningún indicador especializado.")
    lines += ["", "## Recomendación de programas", "",
              "**Recomendación NO se calcula como NPS.** Se conserva la escala BINARIA_SI_NO: porcentaje de respuestas Sí sobre el total válido, sin restar No ni asignar categorías de promotores, pasivos o detractores.", "",
              "| Sí | No | Total válido | Sí (%) | No (%) |", "|---:|---:|---:|---:|---:|",
              f"| {recommendation['si']} | {recommendation['no']} | {recommendation['total_valido']} | {fmt(recommendation['si_pct'])} | {fmt(recommendation['no_pct'])} |", "",
              "## Medios por los que se enteran", "",
              f"Denominador propio: **{current['medios']['n']} respuestas válidas**. Se conservan las categorías informadas, sin puntajes ni faltantes convertidos en Otro.", "",
              "| Medio | Respuestas | Porcentaje |", "|---|---:|---:|"]
    assert sum(r["n"] for r in current["medios"]["valores"]) == current["medios"]["n"]
    lines += [f"| {r['nombre']} | {r['n']} | {fmt(r['porcentaje'])} |" for r in current["medios"]["valores"]]
    if not current["medios"]["valores"]:
        lines.append("Sin respuestas en este periodo.")
    lines += ["", "## Comprobación de filtros", ""]
    for filters in [{"carrera": "Derecho"}, {"modalidad_academica": "Presencial"}, {"participacion": "si"}, {"participacion": "no"}]:
        selected = resumen_actual(now["registros"], now["mapa"], **filters)
        assert len(selected["criterios"]) == 6
        assert selected["encuestas_analizadas"] <= current["encuestas_analizadas"]
        lines.append(f"- {', '.join(f'{k}={v}' for k, v in filters.items())}: encuestas {selected['encuestas_analizadas']}; respuestas a criterios {selected['respuestas']}; participación {fmt(selected['participacion']['si_pct'])}%; respuestas de medios {selected['medios']['n']}.")
    assert resumen_actual(now["registros"], now["mapa"]) == current
    lines += ["", "La participación se filtra por identidad de encuesta antes de recalcular todos los indicadores, medios y comentarios. Quitar filtros restaura el global. Los estados vacíos se conservan como null; un cero válido se conserva como resultado.", "",
              "La pregunta EC_MEJORA se muestra solo en el detalle contextual, sin clasificación semántica. Esta auditoría no contiene comentarios personales ni se muestra en producción. No se avanza al PDF.", ""]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(lines), encoding="utf-8")
    print(f"Educacion Continua: {current['encuestas_analizadas']} encuestas; {current['respuestas']} respuestas a criterios; participacion {fmt(participation['si_pct'])}%; sin historico comparable.")


if __name__ == "__main__":
    audit(*sys.argv[1:])
