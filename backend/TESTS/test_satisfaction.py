import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from backend.SERVICES.calidad.satisfaction import resumen_actual, resumen_historico
from backend.SERVICES.calidad.kpi_service import calcular_kpis_preview


def mapping(year=2026):
    prefix = "ESC" if year == 2026 else "ESC25"
    return [{"SECCION_ID": f"{prefix}_09_SATISFACCION", "PREGUNTA_ID": f"{prefix}_{key}",
             "PREGUNTA_CANONICA_ID": key, "TEXTO_CORTO": name, "TIPO_DATO": kind,
             "ESCALA_ID": scale, "ENTRA_PROMEDIO_SECCION": "SI" if key == "SAT_GENERAL" else "NO",
             "CALCULA_KPI_PREGUNTA": "NO" if kind == "TEXTO" else "SI", "COMPARABLE_HISTORICO": "SI"}
            for key, name, kind, scale in [("SAT_GENERAL", "Satisfacción general", "LIKERT", "SATISFACCION_5"),
                                            ("REC_UDL", "Recomendación", "BINARIA", "BINARIA_SI_NO"),
                                            ("REC_PORQUE", "Motivo de recomendación", "TEXTO", "")]]


def response(definition, value=75, number=1, career="Derecho", modality="Presencial"):
    binary, text = definition["TIPO_DATO"] == "BINARIA", definition["TIPO_DATO"] == "TEXTO"
    return {**definition, "FILA_ORIGEN": number, "CARRERA_BASE": career, "MODALIDAD_ACADEMICA": modality,
            "SECCION_NOMBRE": "Satisfacción y recomendación", "INDICE_100": None if text else value,
            "VALOR_BASE": None if text or value is None else value / 100 if binary else value / 25 + 1,
            "INCLUIR_EN_KPI": "SI" if value is not None and not text else "NO",
            "RESPUESTA_ORIGINAL": value if text else str(value)}


class SatisfactionTests(unittest.TestCase):
    def setUp(self):
        self.now, self.old = mapping(), mapping(2025)

    def test_global_has_two_independent_indicators_and_no_aggregate(self):
        result = resumen_actual([response(self.now[0], 75), response(self.now[1], 100)], self.now)
        self.assertEqual(result["satisfaccion"]["indice"], 75)
        self.assertEqual(result["recomendacion"]["si_pct"], 100)
        self.assertEqual(result["encuestas_analizadas"], 1)
        self.assertIsNone(result["poblacion"]["carrera"])
        for key in ("indice", "resultado_general", "nps"):
            self.assertNotIn(key, result)

    def test_recommendation_is_yes_percentage_not_net_of_no_percentage(self):
        rows = [response(self.now[1], value, index) for index, value in enumerate((100, 100, 100, 0), 1)]
        result = resumen_actual(rows, self.now)["recomendacion"]
        self.assertEqual((result["si"], result["no"], result["si_pct"], result["no_pct"]), (3, 1, 75, 25))
        self.assertNotIn("nps", result)
        self.assertNotEqual(result["si_pct"], result["si_pct"] - result["no_pct"])

    def test_satisfaction_matches_stable_engine_and_is_unchanged_by_recommendation(self):
        rows = [response(self.now[0], value, index) for index, value in enumerate((0, 25, 50, 75, 100), 1)]
        original = copy.deepcopy(rows)
        baseline = calcular_kpis_preview(rows)["preguntas"][0]
        for value in (0, 100):
            current = resumen_actual(rows + [response(self.now[1], value)], self.now)["satisfaccion"]
            for key in ("indice", "n", "favorable_pct", "neutral_pct", "desfavorable_pct"):
                self.assertEqual(current[key], baseline[key])
        self.assertEqual(rows, original)

    def test_unique_survey_union_and_independent_denominators_exclude_empty_and_text(self):
        rows = [response(self.now[0], 75, 1), response(self.now[1], 100, 1),
                response(self.now[1], 0, 2), response(self.now[0], None, 3),
                response(self.now[2], "Motivo contextual", 4), response(self.now[1], 75, 5)]
        result = resumen_actual(rows, self.now)
        self.assertEqual(result["encuestas_analizadas"], 2)
        self.assertEqual((result["satisfaccion"]["n"], result["recomendacion"]["total_valido"]), (1, 2))
        self.assertEqual(result["recomendacion"]["si_pct"], 50)

    def test_missing_scores_are_null_and_valid_zero_is_preserved(self):
        empty = resumen_actual([], self.now)
        self.assertIsNone(empty["satisfaccion"]["indice"])
        self.assertIsNone(empty["recomendacion"]["si_pct"])
        zero = resumen_actual([response(self.now[0], 0), response(self.now[1], 0)], self.now)
        self.assertEqual(zero["satisfaccion"]["indice"], 0)
        self.assertEqual(zero["recomendacion"]["si_pct"], 0)

    def test_career_modality_filters_recalculate_both_indicators_and_restore_global(self):
        rows = []
        for number, career, modality, satisfaction, recommendation in [(1, "Derecho", "Presencial", 100, 100), (2, "Derecho", "Ejecutiva", 50, 0), (3, "Nutrición", "Presencial", 0, 0)]:
            rows += [response(self.now[0], satisfaction, number, career, modality), response(self.now[1], recommendation, number, career, modality)]
        original = copy.deepcopy(rows)
        baseline = resumen_actual(rows, self.now)
        for career, modality, score, percentage, count in [("Derecho", "", 75, 50, 2), ("", "Ejecutiva", 50, 0, 1), ("Derecho", "Presencial", 100, 100, 1)]:
            selected = resumen_actual(rows, self.now, career, modality)
            self.assertEqual((selected["satisfaccion"]["indice"], selected["recomendacion"]["si_pct"], selected["encuestas_analizadas"]), (score, percentage, count))
        self.assertEqual(resumen_actual(rows, self.now), baseline)
        self.assertEqual(rows, original)

    def test_history_uses_canonical_ids_not_order_or_misleading_display_names(self):
        now = [response(self.now[0], 75), response(self.now[1], 100)]
        old = [response(self.old[0], 25), response(self.old[1], 0)]
        self.old[0]["TEXTO_CORTO"], self.old[1]["TEXTO_CORTO"] = "Recomendación", "Satisfacción general"
        result = resumen_historico(now, self.now, list(reversed(old)), list(reversed(self.old)))
        self.assertEqual((result["satisfaccion"]["indice_2025"], result["satisfaccion"]["indice_2026"], result["satisfaccion"]["delta"]), (25, 75, 50))
        self.assertEqual((result["recomendacion"]["porcentaje_2025"], result["recomendacion"]["porcentaje_2026"], result["recomendacion"]["delta_pp"]), (0, 100, 100))

    def test_missing_history_of_one_indicator_does_not_borrow_from_the_other(self):
        now = [response(self.now[0], 75), response(self.now[1], 100)]
        old = [response(self.old[1], 0)]
        result = resumen_historico(now, self.now, old, self.old[1:])
        self.assertFalse(result["satisfaccion"]["comparable"])
        self.assertIsNone(result["satisfaccion"]["indice_2025"])
        self.assertTrue(result["recomendacion"]["comparable"])
        self.assertEqual(resumen_actual(now, self.now)["satisfaccion"]["indice"], 75)

    def test_history_uses_identical_population_without_global_fallback(self):
        now = [response(self.now[0], 100), response(self.now[1], 100), response(self.now[0], 0, 2, "Nutrición")]
        old = [response(self.old[0], 75), response(self.old[1], 0), response(self.old[0], 100, 2, "Nutrición")]
        selected = resumen_historico(now, self.now, old, self.old, "Derecho", "Presencial")
        self.assertEqual(selected["satisfaccion"]["indice_2025"], 75)
        self.assertEqual(selected["recomendacion"]["delta_pp"], 100)
        absent = resumen_historico(now, self.now, old, self.old, "Derecho", "Ejecutiva")
        self.assertFalse(absent["satisfaccion"]["comparable"])
        self.assertFalse(absent["recomendacion"]["comparable"])
        self.assertEqual(selected["poblacion"], resumen_actual(now, self.now, "Derecho", "Presencial")["poblacion"])

    def test_both_map_and_record_approval_are_required_for_history(self):
        now, old = [response(row, 100) for row in self.now[:2]], [response(row, 0) for row in self.old[:2]]
        for target in (self.now, self.old, now, old):
            original = copy.deepcopy(target)
            for row in target:
                row["COMPARABLE_HISTORICO"] = "NO"
            result = resumen_historico(now, self.now, old, self.old)
            self.assertFalse(result["satisfaccion"]["comparable"])
            self.assertFalse(result["recomendacion"]["comparable"])
            target[:] = original

    def test_comments_use_only_approved_reason_under_both_filters(self):
        rows = [response(self.now[2], "Motivo de la población seleccionada"), response(self.now[2], "Motivo de otra carrera", 2, "Nutrición"),
                {**response(self.now[2], "Texto no autorizado"), "PREGUNTA_ID": "OTHER"}]
        result = resumen_actual(rows, self.now, "Derecho", "Presencial")
        self.assertEqual(len(result["comentarios"]), 1)
        self.assertEqual(result["comentarios"][0]["texto"], "Motivo de la población seleccionada")
        self.assertEqual(result["encuestas_analizadas"], 0)

    def test_duplicate_missing_wrong_type_and_extra_indicators_fail_explicitly(self):
        for invalid in [self.now + [self.now[0]], self.now[1:],
                        [{**self.now[0], "TIPO_DATO": "BINARIA"}, *self.now[1:]],
                        self.now + [{**self.now[0], "PREGUNTA_CANONICA_ID": "OTHER"}]]:
            with self.assertRaises(ValueError):
                resumen_actual([], invalid)

    def test_audit_handles_no_responses_and_absent_historical_maps_without_zero_scores(self):
        from backend.TESTS.audit_satisfaction import audit
        capture = {"2026": {"registros": [], "mapa": self.now}, "2025": {"registros": [], "mapa": []}}
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "capture.json", Path(directory) / "audit.md"
            source.write_text(json.dumps(capture), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                audit(source, output)
            report = output.read_text(encoding="utf-8")
            self.assertIn("Recomendación NO se calcula como NPS", report)
            self.assertIn("resultado actual 2026: —%", report)
            self.assertIn("Histórico aprobado de recomendación: No", report)
            self.assertIn("| SAT_GENERAL | 0 | 0 | — | — | — |", report)


class SatisfactionRoutesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        with patch("backend.REPOSITORIES.calidad.calidad_repository.CalidadRepository"):
            from backend.GATEWAY import satisfaction_api as api
        cls.api = api
        app = FastAPI()
        app.include_router(api.router)
        cls.client = TestClient(app)

    def test_http_global_and_history_share_scope(self):
        def source(year, modality, service):
            definitions = mapping(int(year))
            return [response(row, 100 if year == "2026" else 0) for row in definitions[:2]], definitions
        with patch.object(self.api, "_fuente", side_effect=source):
            current = self.client.get("/api/encuesta-calidad/satisfaccion-recomendacion/actual")
            params = {"carrera": "Derecho", "modalidad_academica": "Presencial"}
            selected = self.client.get("/api/encuesta-calidad/satisfaccion-recomendacion/actual", params=params)
            history = self.client.get("/api/encuesta-calidad/satisfaccion-recomendacion/historico", params=params)
        self.assertEqual(current.status_code, 200)
        self.assertIsNone(current.json()["poblacion"]["carrera"])
        self.assertEqual(selected.json()["poblacion"], history.json()["poblacion"])
        self.assertEqual(history.json()["recomendacion"]["delta_pp"], 100)


if __name__ == "__main__":
    unittest.main()
