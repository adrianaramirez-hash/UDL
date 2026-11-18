"""Reproduce both HTTP PDF cases from a real normalized capture, without upstream writes.

python -m backend.TESTS.audit_results_report capture.json output_dir --renderer-path temporary_pymupdf_dir
The capture has 2026/2025 entries with registros, mapa and personas.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend.SERVICES.calidad import kpi_service, institutional_services, academic_services, facilities, school_environment, satisfaction, language_center, continuing_education
from backend.SERVICES.calidad.results_report import scoped


def audit(source, output, renderer_path):
    data = json.loads(Path(source).read_text(encoding="utf-8"))
    # Replace only transport with the captured real response. No score is mocked.
    with patch("backend.REPOSITORIES.calidad.calidad_repository.CalidadRepository"):
        from backend.GATEWAY import results_report_api as api
        from backend.SERVICES.calidad import results_report_source as sources
    sources.clear_snapshots()
    app = FastAPI()
    from backend.TESTS.auth_fixture import authorize_dg
    authorize_dg(app)
    app.include_router(api.router)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    models = []
    original = api.prepare_report

    def captured_source(year, modality, service):
        assert modality == "Escolarizadas" and not service
        return data[year]["registros"], data[year]["mapa"]

    def collect(*args, **kwargs):
        result = original(*args, **kwargs)
        models.append(result)
        return result

    cases = [("Global", {}), ("Derecho_Presencial", {"carrera": "Derecho", "modalidad_academica": "Presencial"})]
    with patch.object(sources, "_fuente", side_effect=captured_source), patch.object(sources, "_current_labels", side_effect=lambda rows, mapping, *_: (rows, mapping)), patch.object(sources, "cargar_personas_evaluadas", side_effect=lambda source, year: data[str(year)]["personas"]), patch.object(api, "prepare_report", side_effect=collect), patch.object(kpi_service, "cargar_personas_evaluadas", side_effect=lambda source, year: data[str(year)]["personas"]):
        for name, scope in cases:
            response = TestClient(app).get("/api/encuesta-calidad/informe-resultados.pdf", params={"periodo": "2026", "modalidad": "Escolarizadas", **scope})
            assert response.status_code == 200, response.text
            assert response.headers["content-type"] == "application/pdf"
            (out / f"Informe_de_resultados_Escolarizadas_{name}_2026.pdf").write_bytes(response.content)
            model = models[-1]
            career, modality = scope.get("carrera", ""), scope.get("modalidad_academica", "")
            engine = kpi_service.calcular_kpis_preview(scoped(data["2026"]["registros"], career, modality))
            assert model["index"] == engine["indice_general"] and model["surveys"] == engine["encuestas"]
            # Validate every projected criterion against its original service.
            for identifier, service, field in [("institutional", institutional_services, "servicios"), ("academic", academic_services, "criterios"), ("facilities", facilities, "criterios"), ("environment", school_environment, "criterios"), ("languages", language_center, "criterios"), ("continuing", continuing_education, "criterios")]:
                actual = service.resumen_actual(data["2026"]["registros"], data["2026"]["mapa"], career, modality)
                module = next(m for m in model["modules"] if m["id"] == identifier)
                for criterion in actual[field]:
                    row = next(r for r in module["criteria"] if r["id"] == criterion["id"])
                    assert (row["value"], row["n"]) == (criterion.get("resultado") if identifier == "continuing" else criterion["indice"], criterion["n"])
                if identifier not in {"languages", "continuing"}:
                    history = service.resumen_historico(data["2026"]["registros"], data["2026"]["mapa"], data["2025"]["registros"], data["2025"]["mapa"], career, modality)
                    assert module["history"] == (history["indice_2025"], history["indice_2026"])
                    for previous in history[field]:
                        row = next(r for r in module["criteria"] if r["id"] == previous["id"])
                        assert row["previous"] == previous["indice_2025"]
            actual = satisfaction.resumen_actual(data["2026"]["registros"], data["2026"]["mapa"], career, modality)
            sat = next(m for m in model["modules"] if m["id"] == "satisfaction")
            assert next(r["value"] for r in sat["criteria"] if r["id"] == "SAT_GENERAL") == actual["satisfaccion"]["indice"]
            assert next(r["value"] for r in sat["criteria"] if r["id"] == "REC_UDL") == actual["recomendacion"]["si_pct"]

    sys.path.insert(0, renderer_path)
    import pymupdf
    review = Path(tempfile.gettempdir()) / "udl-pdf-final-review"
    review.mkdir(exist_ok=True)
    manifest = {"source_rows": {year: len(entry["registros"]) for year, entry in data.items()}, "cases": []}
    for (name, _), model in zip(cases, models):
        path = out / f"Informe_de_resultados_Escolarizadas_{name}_2026.pdf"
        doc = pymupdf.open(path)
        assert not doc.is_repaired and not doc.needs_pass
        assert doc.metadata["title"] == "Informe de resultados"
        pages, all_text = [], ""
        for number, page in enumerate(doc, 1):
            text = page.get_text()
            all_text += text
            assert model["source"] in text and model["career"] in text and model["modality"] in text
            assert f"Página {number} de {len(doc)}" in text
            assert len(text.splitlines()) >= 12, (name, number, "empty page")
            spans = [s for block in page.get_text("dict")["blocks"] if "lines" in block for line in block["lines"] for s in line["spans"]]
            assert all(s["bbox"][0] >= 43 and s["bbox"][2] <= page.rect.width - 42 and s["bbox"][1] >= 16 and s["bbox"][3] <= page.rect.height - 16 for s in spans), (name, number, "margin")
            image = review / f"{name}-{number:02d}.png"
            page.get_pixmap(matrix=pymupdf.Matrix(1.2, 1.2)).save(image)
            pages.append({"page": number, "text_length": len(text), "image": str(image), "margins_ok": True})
        assert "NPS" not in all_text and "RAW_PRIVATE" not in all_text
        assert all(m["name"] in all_text for m in model["modules"])
        assert all(f"{r['value']:.2f}" in all_text for m in model["modules"] for r in m["criteria"] if r["value"] is not None)
        assert all(f"{r['previous']:.2f}" in all_text for m in model["modules"] for r in m["criteria"] if r["previous"] is not None)
        assert all(r["previous"] is None and r["delta"] is None for m in model["modules"][-2:] for r in m["criteria"])
        manifest["cases"].append({"name": name, "file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "pages": len(doc),
                                  "filters": {key: model[key] for key in ["source", "period", "career", "modality"]},
                                  "index": model["index"], "surveys": model["surveys"], "responses": model["responses"], "page_checks": pages,
                                  "modules": [{k: v for k, v in m.items() if k != "people"} for m in model["modules"]]})
        print(name, "pages", len(doc), "surveys", model["surveys"], "index", model["index"], flush=True)
    (out / "validacion_informes.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (review / "models.json").write_text(json.dumps(models, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--renderer-path", required=True)
    args = parser.parse_args()
    audit(args.source, args.output, args.renderer_path)
