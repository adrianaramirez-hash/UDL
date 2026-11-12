"""Audit an existing map/normalized capture without changing sources or methodology.

python -m backend.TESTS.audit_facilities <capture.json> <report.md>
"""
import json
from pathlib import Path
import sys

from backend.SERVICES.calidad.facilities import CRITERIOS, catalogo, resumen_actual, resumen_historico


def audit(source, output):
    capture = json.loads(Path(source).read_text(encoding="utf-8"))
    now, before = capture["2026"], capture["2025"]
    current = resumen_actual(now["registros"], now["mapa"])
    historical = resumen_historico(now["registros"], now["mapa"], before["registros"], before["mapa"])
    old = {row["id"]: row for row in historical["criterios"]}
    assert len(current["criterios"]) == len(CRITERIOS) == 10
    assert len({row["id"] for row in current["criterios"]}) == 10
    assert all(row["nombre"].strip() for row in current["criterios"])
    assert sum(row["n"] for row in current["criterios"]) == current["respuestas"]
    fmt = lambda value: "—" if value is None else f"{value:.2f}"
    delta = lambda previous, actual: None if previous is None or actual is None else round(actual - previous, 2)
    lines = [
        "# Auditoría de Instalaciones y equipo tecnológico", "",
        f"Fuente: captura normalizada `{Path(source).name}`. Fecha de lectura: {capture.get('fecha_captura', 'no registrada')}. Ámbito: Escolarizadas; todas las carreras y modalidades académicas; 2026 frente a 2025 del mismo ámbito.", "",
        f"- Resultado general: **{fmt(current['indice'])}**.",
        f"- Respuestas a criterios: **{current['respuestas']:,}**.",
        f"- Encuestas analizadas: **{current['encuestas_analizadas']:,}**.",
        f"- Criterios encontrados: 10; con histórico: {len(historical['ids_comparables'])}; sin histórico: {10 - len(historical['ids_comparables'])}.",
        f"- Comparativo global: {fmt(historical['indice_2025'])} → {fmt(historical['indice_2026'])}; delta: {fmt(delta(historical['indice_2025'], historical['indice_2026']))} puntos.",
        "- El motor estable calcula índices y denominadores. Encuestas analizadas es la unión de formularios con al menos una respuesta válida. Vacíos y no aplicables no se convierten en cero.",
        "- La comparación usa exclusivamente IDs canónicos y aprobación COMPARABLE_HISTORICO de ambos mapas y registros; los globales usan la misma intersección con respuestas válidas en ambos años.", "",
        "| Nombre | ID canónico | Respuestas actuales | Encuestas con respuesta válida | Resultado 2026 | Resultado 2025 | Delta | Histórico disponible | Estado visual |",
        "|---|---|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for criterion in current["criterios"]:
        history = old[criterion["id"]]
        previous, comparable = history["indice_2025"], history["indice_2026"]
        state = ("Sin respuestas en este periodo" if criterion["indice"] is None else
                 "Sin comparable histórico" if previous is None else "Resultado actual + barras 2025/2026 + delta")
        lines.append(f"| {criterion['nombre']} | {criterion['id']} | {criterion['n']} | {criterion['encuestas_validas']} | {fmt(criterion['indice'])} | {fmt(previous)} | {fmt(delta(previous, comparable))} | {'Sí' if previous is not None else 'No'} | {state} |")
    lines += ["", "## Biblioteca y Cafetería: trazabilidad independiente", "",
              "| Año | Criterio | ID canónico | Pregunta | Columna aprobada | Encabezado RAW verificado |",
              "|---|---|---|---|---|---|"]
    for year, source_data in [("2026", now), ("2025", before)]:
        definitions = catalogo(source_data["mapa"], completo=year == "2026")
        library, cafeteria = definitions.get("INS_BIBLIOTECA"), definitions.get("INS_CAFETERIA")
        if library and cafeteria:
            assert library["PREGUNTA_ID"] != cafeteria["PREGUNTA_ID"]
            if library.get("COLUMNA_ORIGEN") and cafeteria.get("COLUMNA_ORIGEN"):
                assert library["COLUMNA_ORIGEN"] != cafeteria["COLUMNA_ORIGEN"]
        for canonical in ("INS_BIBLIOTECA", "INS_CAFETERIA"):
            definition = definitions.get(canonical)
            if definition is None:
                continue
            header = capture.get("encabezados_verificados", {}).get(year, {}).get(canonical)
            if header:
                assert header["columna"] == definition["COLUMNA_ORIGEN"]
                assert header["encabezado"].strip() == f"[{CRITERIOS[canonical]}]", "Encabezados de Biblioteca/Cafetería cruzados"
            lines.append(f"| {year} | {CRITERIOS[canonical]} | {canonical} | {definition['PREGUNTA_ID']} | {definition.get('COLUMNA_ORIGEN', 'No registrada')} | {header['encabezado'].strip() if header else 'No capturado'} |")
    lines += ["", "Biblioteca solo se compara con INS_BIBLIOTECA; Cafetería solo con INS_CAFETERIA. No se utilizan posiciones, nombres similares ni equivalencias inferidas.", "",
              "## Verificación de población", ""]
    for career, modality in [("Derecho", ""), ("", "Presencial"), ("Derecho", "Presencial")]:
        selected = resumen_actual(now["registros"], now["mapa"], career, modality)
        history = resumen_historico(now["registros"], now["mapa"], before["registros"], before["mapa"], career, modality)
        assert selected["poblacion"] == history["poblacion"]
        assert selected["respuestas"] <= current["respuestas"]
        assert len(selected["criterios"]) == 10
        lines.append(f"- {career or 'Todas las carreras'} / {modality or 'Todas las modalidades'}: resultado {fmt(selected['indice'])}; respuestas {selected['respuestas']}; encuestas {selected['encuestas_analizadas']}; histórico {fmt(history['indice_2025'])} → {fmt(history['indice_2026'])}.")
    assert resumen_actual(now["registros"], now["mapa"]) == current
    lines += ["", "Quitar filtros recupera el global. La auditoría es temporal, no se muestra en producción y no incluye comentarios personales. No se avanza a Ambiente escolar.", ""]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(lines), encoding="utf-8")
    print(f"Instalaciones: {fmt(current['indice'])}; {current['respuestas']} respuestas; {current['encuestas_analizadas']} encuestas; {len(historical['ids_comparables'])}/10 con histórico.")


if __name__ == "__main__":
    audit(*sys.argv[1:])
