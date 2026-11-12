"""Compare a captured normalized source with the pre-fix engine, without writes upstream.

python -m backend.TESTS.audit_quality_functional base.json capture.json before.py output.json
The JSON is intended for the frontend catalog audit (presentation identity aliases).
"""
import json
from pathlib import Path
import sys
from unittest.mock import patch

from backend.SERVICES.calidad import kpi_service


def audit(base_path, capture_path, before_path, output_path):
    base = json.loads(Path(base_path).read_text(encoding="utf-8"))
    capture = json.loads(Path(capture_path).read_text(encoding="utf-8"))
    people = capture["2026"]["personas"]
    baseline = {}
    exec(compile(Path(before_path).read_text(encoding="utf-8-sig"), before_path, "exec"), baseline)
    baseline["cargar_personas_evaluadas"] = lambda *_: people
    before = baseline["calcular_kpis_preview"](base["registros"])
    with patch.object(kpi_service, "cargar_personas_evaluadas", return_value=people):
        after = kpi_service.calcular_kpis_preview(base["registros"])
    for key in before:
        if key not in {"comparativos_personal", "comparativos_direccion"}:
            assert before[key] == after[key], key
    for group in ["comparativos_personal", "comparativos_direccion"]:
        for key in before[group]:
            if group == "comparativos_direccion" and key == "por_carrera_modalidad":
                # All previously computed metrics must remain byte-for-byte equal.
                for row in before[group][key]:
                    assert row in after[group][key], row
            else:
                assert before[group][key] == after[group][key], (group, key)
    def summary(result):
        return {key: result[key] for key in ["indice_general", "secciones", "preguntas", "personas", "comparativos_personal", "comparativos_direccion"]}
    output = {"before": summary(before), "after": summary(after),
              "history": capture["2025"]["summary"], "catalog": people,
              "unchanged_existing_metrics": True}
    output["history"]["respuestas_export"] = []
    Path(output_path).write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
    print("Existing metrics unchanged; assignment rows:", len(after["comparativos_personal"]["asignaciones"]))


if __name__ == "__main__":
    audit(*sys.argv[1:])
