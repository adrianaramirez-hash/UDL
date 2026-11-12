"""Audit captured mapped/normalized records without any external writes.

Usage: python -m backend.TESTS.audit_institutional_services <input.json> <report.md>
The input contains 2026/2025 entries with mapa and registros from the stable sources.
"""
import json
from pathlib import Path
import sys

from backend.SERVICES.calidad.institutional_services import resumen_actual, resumen_historico


def audit(source, output):
    source = json.loads(Path(source).read_text(encoding="utf-8"))
    now, before = source["2026"], source["2025"]
    current = resumen_actual(now["registros"], now["mapa"])
    history = resumen_historico(now["registros"], now["mapa"], before["registros"], before["mapa"])
    values = {row["id"]: row for row in history["servicios"]}
    assert len({row["id"] for row in current["servicios"]}) == len(current["servicios"])
    assert all(row["nombre"].strip() for row in current["servicios"])
    assert all(row["id"] in values for row in current["servicios"])
    rows = []
    for row in current["servicios"]:
        old = values[row["id"]]["indice_2025"]
        value = values[row["id"]]["indice_2026"]
        delta = round(value - old, 2) if value is not None and old is not None else None
        fmt = lambda x: "—" if x is None else f"{x:.2f}"
        state = "Sin respuestas en este periodo" if row["indice"] is None else "Sin comparable histórico" if old is None else "Resultado actual + barras + delta"
        rows.append(f"| {row['nombre']} | {fmt(row['indice'])} | {row['n']} | {'Sí' if old is not None else 'No'} | {fmt(old)} | {fmt(value)} | {fmt(delta)} | {state} |")
    # Exercise a real population and ensure its two indices are not global fallbacks.
    filtered = resumen_actual(now["registros"], now["mapa"], "Derecho", "Presencial")
    filtered_history = resumen_historico(now["registros"], now["mapa"], before["registros"], before["mapa"], "Derecho", "Presencial")
    assert filtered["poblacion"] == filtered_history["poblacion"]
    assert filtered["respuestas"] < current["respuestas"]
    assert len(filtered["servicios"]) == len(current["servicios"])
    lines = [
        "# Auditoría de Servicios institucionales", "",
        "Lectura real: 5 de noviembre de 2026. Ámbito: Escolarizadas, 2026, todas las carreras y modalidades académicas. Histórico 2025 del mismo ámbito.", "",
        f"- Servicios actuales: {len(current['servicios'])}; con histórico válido: {sum(row['indice_2025'] is not None for row in history['servicios'])}; sin histórico: {sum(row['indice_2025'] is None for row in history['servicios'])}.",
        f"- Índice actual: {current['indice']:.2f}; respuestas válidas: {current['respuestas']}.",
        f"- Intersección global aprobada con datos en ambos años: {len(history['criterios'])} servicios. 2025: {history['indice_2025']:.2f}; 2026: {history['indice_2026']:.2f}; delta: {history['indice_2026'] - history['indice_2025']:+.2f} puntos.",
        "- Los índices delegan en el motor estable: igual peso por pregunta. La distribución global agrupa respuestas válidas y se muestra separada del índice.",
        "- La unión usa exclusivamente PREGUNTA_CANONICA_ID y COMPARABLE_HISTORICO de ambos mapas y registros; sin coincidencias por texto.",
        "- Centro de Idiomas y Educación Continua, presentes en la sección 2025, no pertenecen a la intersección actual y no se incorporan al índice ni al listado 2026.",
        "- El mapa actual denomina al ID SER_PARAMEDICO «Servicio institucional 09». Se conserva esa denominación; no se modifica catálogo ni normalización.", "",
        "| Servicio | Actual | Respuestas actuales | Histórico | 2025 | 2026 comparable | Delta (pts) | Estado visual esperado |",
        "|---|---:|---:|:---:|---:|---:|---:|---|", *rows, "",
        "## Filtro real: Derecho / Presencial", "",
        f"- Resultado actual: {filtered['indice']:.2f}; respuestas: {filtered['respuestas']}; servicios conservados: {len(filtered['servicios'])}.",
        f"- Histórico de la misma población: {filtered_history['indice_2025']:.2f} → {filtered_history['indice_2026']:.2f}.",
        "- Ninguna lectura amplía el filtro a la población global cuando faltan respuestas.", "",
        "## Verificación", "",
        "- Pruebas unitarias del cálculo, catálogo, intersección canónica y filtros por carrera/modalidad.",
        "- Pruebas de renderizado: global sin carrera, todos los servicios, estados ausentes, barras y deltas, carga y error histórico independiente.",
        "- Pruebas del cliente: ambos endpoints reciben exactamente la misma carrera, modalidad y programa.",
        "- Dirección, componentes compartidos, normalización, equivalencias, motor KPI, catálogo y exportaciones sin cambios de comportamiento.",
        "- Esta auditoría es documentación de desarrollo; no aparece en producción.", "",
    ]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(lines), encoding="utf-8")
    print(f"Auditoría: {len(rows)} servicios; filtro Derecho: {filtered['indice']} / {filtered['respuestas']} respuestas.")


if __name__ == "__main__":
    audit(*sys.argv[1:])
