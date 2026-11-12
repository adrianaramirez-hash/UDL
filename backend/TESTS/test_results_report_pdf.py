import json
import unittest
from unittest.mock import patch

from backend.SERVICES.calidad.results_report import prepare_report
from backend.SERVICES.calidad.results_report_pdf import render_report
from backend.SERVICES.calidad import facilities
from backend.TESTS import (
    test_academic_services as academic, test_facilities as equipment,
    test_school_environment as environment, test_satisfaction as satisfaction,
    test_language_center as languages, test_continuing_education as continuing,
    test_institutional_services as institutional,
)


def instrument(year):
    definitions = [institutional.definition(year=year)]
    definitions += academic.mapping(year) + equipment.mapping(year) + environment.mapping(year) + satisfaction.mapping(year)
    if year == 2026:
        definitions += languages.mapping() + continuing.mapping()
    return definitions


def population(definitions, score, number=1, career="Derecho", modality="Presencial"):
    result = []
    for definition in definitions:
        definition = {"TIPO_DATO": "LIKERT", **definition}
        value = "RAW_PRIVATE_COMMENT_NEVER_EXPORTED" if definition["TIPO_DATO"] == "TEXTO" else (100 if score else 0) if definition["TIPO_DATO"] == "BINARIA" else score
        result.append(languages.response(definition, value, number, career, modality))
    return result


class ResultsReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.now_map, cls.old_map = instrument(2026), instrument(2025)
        cls.now = population(cls.now_map, 0) + population(cls.now_map, 100, 2, "Nutrición") + population(cls.now_map, 100, 3, modality="Ejecutiva")
        cls.old = population(cls.old_map, 100) + population(cls.old_map, 0, 2, "Nutrición") + population(cls.old_map, 0, 3, modality="Ejecutiva")

    def model(self, **scope):
        return prepare_report(self.now, self.now_map, self.old, self.old_map, **scope)

    def test_nine_modules_and_exact_separate_criteria(self):
        model = self.model()
        self.assertEqual(len(model["modules"]), 9)
        sizes = {m["id"]: len(m["criteria"]) for m in model["modules"]}
        self.assertEqual((sizes["direction"], sizes["academic"], sizes["facilities"], sizes["environment"], sizes["satisfaction"], sizes["languages"], sizes["continuing"]), (4, 12, 10, 3, 2, 3, 6))
        for module in model["modules"]:
            values = [r["value"] for r in module["criteria"] if r["value"] is not None]
            self.assertEqual(values, sorted(values))

    def test_same_career_and_modality_in_current_and_history_without_global_fallback(self):
        model = self.model(career="Derecho", modality="Presencial")
        module = next(m for m in model["modules"] if m["id"] == "facilities")
        self.assertEqual((module["index"], module["surveys"], module["responses"]), (0, 1, 10))
        self.assertTrue(all(r["previous"] == 100 and r["compared_current"] == 0 and r["delta"] == -100 for r in module["criteria"]))
        self.assertNotEqual(module["index"], next(m for m in self.model()["modules"] if m["id"] == "facilities")["index"])

    def test_missing_history_never_uses_other_career_or_modality(self):
        old = [r for r in self.old if r["MODALIDAD_ACADEMICA"] == "Ejecutiva" or r["CARRERA_BASE"] == "Nutrición"]
        model = prepare_report(self.now, self.now_map, old, self.old_map, career="Derecho", modality="Presencial")
        module = next(m for m in model["modules"] if m["id"] == "facilities")
        self.assertTrue(all(r["previous"] is None and r["delta"] is None and r["value"] == 0 for r in module["criteria"]))

    def test_identity_resolved_from_program_also_constrains_historical_population(self):
        now = [r for r in self.now if r["CARRERA_BASE"] == "Derecho" and r["MODALIDAD_ACADEMICA"] == "Presencial"]
        old = [r for r in self.old if r["CARRERA_BASE"] == "Derecho"]
        model = prepare_report(now, self.now_map, old, self.old_map, service="Programa seleccionado")
        self.assertEqual((model["career"], model["modality"]), ("Derecho", "Presencial"))
        module = next(m for m in model["modules"] if m["id"] == "facilities")
        self.assertTrue(all(r["previous"] == 100 and r["compared_current"] == 0 for r in module["criteria"]))

    def test_no_responses_is_none_and_valid_zero_survives(self):
        empty = self.model(career="Sin muestra")
        self.assertEqual(empty["surveys"], 0)
        self.assertTrue(all(r["value"] is None for m in empty["modules"] for r in m["criteria"]))
        present = self.model(career="Derecho", modality="Presencial")
        self.assertTrue(any(r["value"] == 0 for m in present["modules"] for r in m["criteria"]))

    def test_percentages_and_indices_are_separate_and_specialized_have_no_history(self):
        model = self.model(career="Derecho", modality="Presencial")
        environment = next(m for m in model["modules"] if m["id"] == "environment")
        self.assertEqual(environment["incidence"]["unit"], "pp")
        self.assertNotIn("AMB_PROBLEMA", {r["id"] for r in environment["criteria"]})
        satisfaction = next(m for m in model["modules"] if m["id"] == "satisfaction")
        self.assertIsNone(satisfaction["index"])
        self.assertEqual({r["id"]: r["unit"] for r in satisfaction["criteria"]}, {"SAT_GENERAL": "pts", "REC_UDL": "pp"})
        for module in model["modules"][-2:]:
            self.assertTrue(module["current_only"])
            self.assertTrue(all(r["previous"] is None and r["delta"] is None for r in module["criteria"]))

    def test_facilities_values_and_counts_are_exact_existing_projection(self):
        model = self.model(career="Derecho", modality="Presencial")
        original = facilities.resumen_actual(self.now, self.now_map, "Derecho", "Presencial")
        module = next(m for m in model["modules"] if m["id"] == "facilities")
        for criterion in original["criterios"]:
            row = next(r for r in module["criteria"] if r["id"] == criterion["id"])
            self.assertEqual((row["value"], row["n"]), (criterion["indice"], criterion["n"]))
        library, cafeteria = [r for r in module["criteria"] if r["name"] in {"Biblioteca", "Cafetería"}]
        self.assertNotEqual(library["id"], cafeteria["id"])

    def test_comments_never_enter_report_model_or_rendered_content(self):
        model = self.model()
        self.assertNotIn("RAW_PRIVATE_COMMENT", json.dumps(model))
        pdf = render_report(model)
        self.assertTrue(pdf.startswith(b"%PDF-"))
        self.assertTrue(pdf.rstrip().endswith(b"%%EOF"))
        self.assertNotIn(b"RAW_PRIVATE_COMMENT", pdf)


    def test_pdf_endpoint_keeps_filters_and_returns_attachment(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        with patch("backend.REPOSITORIES.calidad.calidad_repository.CalidadRepository"):
            from backend.GATEWAY import results_report_api as api
            from backend.SERVICES.calidad import results_report_source as sources
        sources.clear_snapshots()
        app = FastAPI()
        app.include_router(api.router)
        def source(year, source, service):
            rows, mapping = (self.now, self.now_map) if year == "2026" else (self.old, self.old_map)
            return [{**r, "SERVICIO_PROGRAMA": "Programa activo"} for r in rows], mapping
        with patch.object(sources, "_fuente", side_effect=source) as fetch:
            result = TestClient(app).get("/api/encuesta-calidad/informe-resultados.pdf", params={"carrera": "Derecho", "modalidad_academica": "Presencial", "servicio": "Programa activo"})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.headers["content-type"], "application/pdf")
        self.assertIn("Derecho_Presencial", result.headers["content-disposition"])
        self.assertEqual([call.args for call in fetch.call_args_list], [("2026", "Escolarizadas", ""), ("2025", "Escolarizadas", "")])
        self.assertEqual(result.headers["x-pdf-snapshot-cache"], "MISS")

    def test_source_quota_is_reported_without_global_retry_or_partial_pdf(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from gspread.exceptions import APIError
        with patch("backend.REPOSITORIES.calidad.calidad_repository.CalidadRepository"):
            from backend.GATEWAY import results_report_api as api
            from backend.SERVICES.calidad import results_report_source as sources
        sources.clear_snapshots()
        app = FastAPI()
        app.include_router(api.router)
        error = APIError(type("Quota", (), {"json": lambda self: {"error": {"code": 429, "message": "quota"}}})())
        with patch.object(sources, "_fuente", side_effect=error) as source, self.assertLogs(api.logger, level="ERROR") as logs:
            result = TestClient(app).get("/api/encuesta-calidad/informe-resultados.pdf", params={"carrera": "Derecho", "modalidad_academica": "Presencial"})
        self.assertEqual(result.status_code, 503)
        self.assertEqual(source.call_count, 1)
        self.assertIn("temporalmente", result.json()["detail"])
        self.assertNotEqual(result.headers["content-type"], "application/pdf")
        self.assertIn("APIError", " ".join(logs.output))

    def test_old_instrument_is_not_forced_into_nine_current_modules(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        with patch("backend.REPOSITORIES.calidad.calidad_repository.CalidadRepository"):
            from backend.GATEWAY import results_report_api as api
            from backend.SERVICES.calidad import results_report_source as sources
        app = FastAPI()
        app.include_router(api.router)
        with patch.object(sources, "_fuente") as source:
            result = TestClient(app).get("/api/encuesta-calidad/informe-resultados.pdf?periodo=2025")
        self.assertEqual(result.status_code, 400)
        source.assert_not_called()
