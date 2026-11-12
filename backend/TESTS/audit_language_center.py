"""Read-only audit of the specialized 2026 section and descriptive context.

python -m backend.TESTS.audit_language_center <capture.json> <report.md>
"""
import json
from pathlib import Path
import sys

from backend.SERVICES.calidad.language_center import CRITERIOS, CONTEXTOS, resumen_actual, catalogo


def audit(source, output):
    capture = json.loads(Path(source).read_text(encoding="utf-8"))
    now = capture["2026"]
    current = resumen_actual(now["registros"], now["mapa"])
    definitions = catalogo(now["mapa"])
    assert len(current["criterios"]) == len(CRITERIOS) == 3
    assert len({r["id"] for r in current["criterios"]}) == 3
    assert all(r["nombre"].strip() and r["historico_disponible"] is False for r in current["criterios"])
    assert all("indice_2025" not in r and "delta" not in r for r in current["criterios"])
    assert sum(r["n"] for r in current["criterios"]) == current["respuestas"]
    assert "indice" not in current
    context_ids = {r["PREGUNTA_ID"] for key, r in definitions.items() if key in CONTEXTOS}
    without_context = resumen_actual([r for r in now["registros"] if r.get("PREGUNTA_ID") not in context_ids], now["mapa"])
    assert without_context["criterios"] == current["criterios"]
    assert without_context["respuestas"] == current["respuestas"]
    fmt = lambda value: "—" if value is None else f"{value:.2f}"
    usage = current["uso"]
    lines = ["# Auditoría de Centro de Idiomas", "",
             f"Lectura de fuentes: {capture.get('fecha_captura', 'fecha no registrada')}. Captura: `{Path(source).name}`. Ámbito: Escolarizadas, 2026, todas las carreras, modalidades, idiomas y tipos de curso.", "",
             f"- Encuestas analizadas: **{current['encuestas_analizadas']:,}**.",
             f"- Respuestas a criterios: **{current['respuestas']:,}**.",
             "- Encuestas analizadas es la unión de formularios con al menos una respuesta válida en los criterios, uso, variables descriptivas o preguntas abiertas aprobadas de la sección. Respeta RESPUESTA_VALIDA y no convierte respuestas categóricas o abiertas en puntajes.",
             "- Respuestas a criterios suma exclusivamente las respuestas válidas a los tres criterios especializados. Cada índice y distribución procede del motor estable. No se publica un resultado general agregado ni un ranking docente.",
             "- Se retiraron los registros de uso, idioma, curso y profesor: los tres índices, denominadores y distribuciones cuantitativas permanecieron idénticos.", "",
             "## Tres criterios especializados", "",
             "| Criterio | ID canónico | Pregunta | Columna | Respuestas válidas | Encuestas con respuesta válida | Resultado 2026 | Histórico disponible | Estado visual esperado |",
             "|---|---|---|---|---:|---:|---:|:---:|---|"]
    for row in current["criterios"]:
        state = "Sin respuestas en este periodo" if row["indice"] is None else "Barra actual + distribución + Sin comparable histórico"
        lines.append(f"| {row['nombre']} | {row['id']} | {row['pregunta_id']} | {definitions[row['id']].get('COLUMNA_ORIGEN', 'No registrada')} | {row['n']} | {row['encuestas_validas']} | {fmt(row['indice'])} | No | {state} |")
    lines += ["", "**Los tres criterios especializados 2026 no tienen comparación histórica válida con la pregunta general 2025.**", "",
              "La implementación no solicita ni une datos históricos. No usa nombres similares, coincidencias aproximadas o equivalencias nuevas. No muestra barras 2025 contra 2026 ni calcula deltas.", ""]
    for reference in capture.get("2025", {}).get("referencias_generales", []):
        lines.append(f"- Referencia 2025 existente: `{reference['PREGUNTA_ID']}` / `{reference['PREGUNTA_CANONICA_ID']}` en `{reference['SECCION_ID']}`, COMPARABLE_HISTORICO={reference.get('COMPARABLE_HISTORICO')}. Solo trazabilidad metodológica; su resultado no se utiliza ni se muestra como comparable.")
    lines += ["", "## Uso del servicio · indicador descriptivo", "",
              "| Respuestas Sí | Respuestas No | Total válido | Uso Sí (%) | Uso No (%) |",
              "|---:|---:|---:|---:|---:|",
              f"| {usage['si']} | {usage['no']} | {usage['total_valido']} | {fmt(usage['si_pct'])} | {fmt(usage['no_pct'])} |", ""]
    for field, title in [("idiomas", "Idiomas"), ("tipos_curso", "Tipos de curso")]:
        distribution = current[field]
        assert sum(r["n"] for r in distribution["valores"]) == distribution["n"]
        lines += [f"## {title}", "", f"Denominador propio: **{distribution['n']} respuestas válidas** a esta pregunta. Los faltantes no se convierten en Otro ni en cero.", "",
                  "| Valor informado | Respuestas | Porcentaje |", "|---|---:|---:|"]
        lines += [f"| {r['nombre']} | {r['n']} | {fmt(r['porcentaje'])} |" for r in distribution["valores"]]
        if not distribution["valores"]:
            lines.append("Sin respuestas en este periodo.")
        lines.append("")
    lines += ["## Profesor actual", "",
              f"- ID canónico: IDI_PROFESOR. Respuestas descriptivas válidas: {current['profesores']['n']}; valores distintos informados: {len(current['profesores']['nombres'])}.",
              "- Solo se presentan los nombres informados en orden alfabético dentro del detalle contextual. No se asignan puntajes ni posiciones por cantidad de respuestas. Los nombres no se reproducen en este informe.", "",
              "## Comprobación de filtros", ""]
    examples = [{"carrera": "Derecho"}, {"modalidad_academica": "Presencial"}]
    if current["opciones"]["idiomas"]:
        examples.append({"idioma": current["opciones"]["idiomas"][0]})
    if current["opciones"]["tipos_curso"]:
        examples.append({"tipo_curso": current["opciones"]["tipos_curso"][0]})
    for filters in examples:
        selected = resumen_actual(now["registros"], now["mapa"], **filters)
        assert len(selected["criterios"]) == 3
        assert selected["encuestas_analizadas"] <= current["encuestas_analizadas"]
        lines.append(f"- {', '.join(f'{k}={v}' for k, v in filters.items())}: encuestas {selected['encuestas_analizadas']}; respuestas a criterios {selected['respuestas']}; uso {fmt(selected['uso']['si_pct'])}%.")
    assert resumen_actual(now["registros"], now["mapa"]) == current
    lines += ["", "Idioma y curso se filtran por identidad de encuesta antes de proyectar todos sus criterios, uso, distribuciones, profesores y comentarios. Quitar filtros restaura el global.", "",
              "Las dos preguntas abiertas se mantienen separadas en el detalle contextual, sin clasificación semántica. Esta auditoría no incluye comentarios personales ni se muestra en producción. No se avanza a Educación Continua.", ""]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(lines), encoding="utf-8")
    print(f"Idiomas: {current['encuestas_analizadas']} encuestas; {current['respuestas']} respuestas a criterios; uso {fmt(usage['si_pct'])}%; sin historico comparable.")


if __name__ == "__main__":
    audit(*sys.argv[1:])
