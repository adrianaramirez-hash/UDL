import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from backend.SERVICES.calidad.facilities import CRITERIOS, resumen_actual, resumen_historico
from backend.SERVICES.calidad.kpi_service import calcular_kpis_preview


def mapping(year=2026):
    prefix = "ESC" if year == 2026 else "ESC25"
    return [{"SECCION_ID": f"{prefix}_07_INSTALACIONES", "PREGUNTA_ID": f"{prefix}_{key}",
             "PREGUNTA_CANONICA_ID": key, "TEXTO_CORTO": label, "TIPO_DATO": "LIKERT",
             "ENTRA_PROMEDIO_SECCION": "SÍ", "CALCULA_KPI_PREGUNTA": "SÍ", "COMPARABLE_HISTORICO": "SÍ"}
            for key, label in CRITERIOS.items()]


def response(definition, value=75, number=1, career="Derecho", modality="Presencial"):
    return {**definition, "FILA_ORIGEN": number, "CARRERA_BASE": career,
            "MODALIDAD_ACADEMICA": modality, "SECCION_NOMBRE": "Instalaciones y equipo tecnológico",
            "INDICE_100": value, "VALOR_BASE": value / 25 + 1 if value is not None else None,
            "INCLUIR_EN_KPI": "SÍ" if value is not None else "NO", "RESPUESTA_ORIGINAL": str(value)}


class FacilitiesTests(unittest.TestCase):
    def setUp(self):
        self.current_map, self.previous_map = mapping(), mapping(2025)

    def test_library_and_cafeteria_keep_independent_data_with_reordered_maps(self):
        current = {row["PREGUNTA_CANONICA_ID"]: row for row in self.current_map}
        previous = {row["PREGUNTA_CANONICA_ID"]: row for row in self.previous_map}
        rows = [response(current["INS_BIBLIOTECA"], 100),
                response(current["INS_CAFETERIA"], 25, number=2)]
        old = [response(previous["INS_CAFETERIA"], 75, number=2),
               response(previous["INS_BIBLIOTECA"], 50)]
        result = resumen_historico(rows, self.current_map, old, list(reversed(self.previous_map)))
        criteria = {row["id"]: row for row in result["criterios"]}
        self.assertEqual((criteria["INS_BIBLIOTECA"]["indice_2025"], criteria["INS_BIBLIOTECA"]["indice_2026"]), (50, 100))
        self.assertEqual((criteria["INS_CAFETERIA"]["indice_2025"], criteria["INS_CAFETERIA"]["indice_2026"]), (75, 25))
        self.assertEqual(result["ids_comparables"], ["INS_BIBLIOTECA", "INS_CAFETERIA"])
        self.assertNotEqual(current["INS_BIBLIOTECA"]["PREGUNTA_ID"], current["INS_CAFETERIA"]["PREGUNTA_ID"])

    def test_library_without_history_never_uses_cafeteria_history_or_display_name(self):
        current = {row["PREGUNTA_CANONICA_ID"]: row for row in self.current_map}
        # Misleading text cannot override the approved canonical equivalence.
        previous = {**self.previous_map[7], "TEXTO_CORTO": "Biblioteca"}
        result = resumen_historico([response(current["INS_BIBLIOTECA"], 100),
                                    response(current["INS_CAFETERIA"], 25)],
                                   self.current_map, [response(previous, 75)], [previous])
        criteria = {row["id"]: row for row in result["criterios"]}
        self.assertIsNone(criteria["INS_BIBLIOTECA"]["indice_2025"])
        self.assertEqual(criteria["INS_CAFETERIA"]["indice_2025"], 75)
        self.assertEqual(result["ids_comparables"], ["INS_CAFETERIA"])

    def test_unique_surveys_are_a_union_and_each_criterion_has_its_own_denominator(self):
        rows = [response(self.current_map[6], 100, number=1),
                response(self.current_map[6], 50, number=2),
                response(self.current_map[7], 25, number=1),
                response(self.current_map[7], None, number=3)]
        result = resumen_actual(rows, self.current_map)
        self.assertEqual((result["respuestas"], result["encuestas_analizadas"]), (3, 2))
        self.assertEqual((result["criterios"][6]["n"], result["criterios"][6]["encuestas_validas"]), (2, 2))
        self.assertEqual((result["criterios"][7]["n"], result["criterios"][7]["encuestas_validas"]), (1, 1))
        self.assertEqual(result["criterios"][6]["indice"], 75)
        self.assertEqual(result["criterios"][7]["indice"], 25)
        self.assertEqual(result["indice"], 50)  # Stable equal weight of criterion indices.

    def test_audit_preserves_null_without_current_or_historical_data(self):
        from backend.TESTS.audit_facilities import audit
        capture = {"2026": {"registros": [], "mapa": self.current_map},
                   "2025": {"registros": [], "mapa": self.previous_map}}
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "capture.json", Path(directory) / "audit.md"
            source.write_text(json.dumps(capture), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                audit(source, output)
            report = output.read_text(encoding="utf-8")
            self.assertIn("con histórico: 0; sin histórico: 10", report)
            self.assertIn("Resultado general: **—**", report)
            self.assertEqual(report.count("Sin respuestas en este periodo"), 10)

    def test_audit_rejects_crossed_library_cafeteria_raw_headers(self):
        from backend.TESTS.audit_facilities import audit
        self.current_map[6]["COLUMNA_ORIGEN"] = "DM"
        self.current_map[7]["COLUMNA_ORIGEN"] = "DN"
        capture = {"2026": {"registros": [], "mapa": self.current_map},
                   "2025": {"registros": [], "mapa": self.previous_map},
                   "encabezados_verificados": {"2026": {"INS_BIBLIOTECA": {
                       "columna": "DM", "encabezado": " [Cafetería]"}}}}
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "capture.json", Path(directory) / "audit.md"
            source.write_text(json.dumps(capture), encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "cruzados"):
                audit(source, output)

    def test_global_without_career_has_exactly_ten_unique_named_criteria(self):
        result = resumen_actual([response(row) for row in self.current_map], self.current_map)
        self.assertEqual(result["indice"], 75)
        self.assertEqual(result["respuestas"], 10)
        self.assertEqual(result["encuestas_analizadas"], 1)
        self.assertEqual(len(result["criterios"]), 10)
        self.assertEqual(len({row["id"] for row in result["criterios"]}), 10)
        self.assertTrue(all(row["nombre"].strip() for row in result["criterios"]))
        self.assertIsNone(result["poblacion"]["carrera"])

    def test_career_filter_recalculates_all_counters_and_reset_restores_global(self):
        rows = [response(row, 100) for row in self.current_map] + [response(row, 50, 2, "Nutrición") for row in self.current_map]
        before = copy.deepcopy(rows)
        global_result = resumen_actual(rows, self.current_map)
        selected = resumen_actual(rows, self.current_map, "Derecho")
        self.assertEqual((selected["indice"], selected["respuestas"], selected["encuestas_analizadas"]), (100, 10, 1))
        self.assertEqual((global_result["indice"], global_result["respuestas"], global_result["encuestas_analizadas"]), (75, 20, 2))
        self.assertEqual(resumen_actual(rows, self.current_map), global_result)
        self.assertEqual(rows, before)

    def test_modality_filter_keeps_populations_separate(self):
        rows = [response(row, 100) for row in self.current_map] + [response(row, 25, 2, modality="Ejecutiva") for row in self.current_map]
        result = resumen_actual(rows, self.current_map, modalidad_academica="Ejecutiva")
        self.assertEqual((result["indice"], result["respuestas"], result["encuestas_analizadas"]), (25, 10, 1))

    def test_no_responses_is_null_and_zero_remains_a_valid_score(self):
        rows = [response(self.current_map[0], 0), response(self.current_map[3], None)]
        result = resumen_actual(rows, self.current_map)
        self.assertEqual(result["criterios"][0]["indice"], 0)
        self.assertIsNone(result["criterios"][3]["indice"])
        self.assertEqual(result["criterios"][3]["n"], 0)
        self.assertEqual(len(result["criterios"]), 10)

    def test_criterion_denominator_excludes_empty_and_not_applicable_responses(self):
        practice = self.current_map[4]
        rows = [response(practice, 75), response(practice, None, 2),
                {**response(practice, None, 3), "RESPUESTA_ORIGINAL": "No aplica"}]
        result = resumen_actual(rows, self.current_map)
        criterion = result["criterios"][4]
        self.assertEqual((criterion["indice"], criterion["n"], criterion["encuestas_validas"]), (75, 1, 1))
        self.assertEqual(criterion["favorable_pct"], 100)
        self.assertEqual(result["encuestas_analizadas"], 1)

    def test_no_history_is_null_not_zero_and_does_not_remove_current_criterion(self):
        rows = [response(row, 75) for row in self.current_map]
        previous = [response(row, 50) for row in self.previous_map[:-1]]
        result = resumen_historico(rows, self.current_map, previous, self.previous_map[:-1])
        self.assertIsNone(result["criterios"][-1]["indice_2025"])
        self.assertEqual(len(result["criterios"]), 10)
        self.assertEqual(resumen_actual(rows, self.current_map)["criterios"][-1]["indice"], 75)

    def test_historical_population_is_identical_and_never_falls_back_to_global(self):
        rows = [response(row, 100) for row in self.current_map] + [response(row, 0, 2, "Nutrición") for row in self.current_map]
        previous = [response(row, 75) for row in self.previous_map] + [response(row, 25, 2, "Nutrición") for row in self.previous_map]
        selected = resumen_historico(rows, self.current_map, previous, self.previous_map, "Derecho", "Presencial")
        self.assertEqual((selected["indice_2025"], selected["indice_2026"]), (75, 100))
        absent = resumen_historico(rows, self.current_map, previous, self.previous_map, "Derecho", "Ejecutiva")
        self.assertIsNone(absent["indice_2025"])
        self.assertIsNone(absent["indice_2026"])
        self.assertEqual(selected["poblacion"], resumen_actual(rows, self.current_map, "Derecho", "Presencial")["poblacion"])

    def test_global_comparison_uses_same_approved_intersection(self):
        self.current_map[1]["COMPARABLE_HISTORICO"] = "NO"
        current = [response(self.current_map[0], 75), response(self.current_map[1], 0), response(self.current_map[2], 100)]
        previous = [response(self.previous_map[0], 50), response(self.previous_map[1], 100)]
        result = resumen_historico(current, self.current_map, previous, self.previous_map)
        self.assertEqual(result["ids_comparables"], ["INS_LAB_COMPUTO"])
        self.assertEqual((result["indice_2025"], result["indice_2026"]), (50, 75))

    def test_matching_text_without_approved_canonical_id_is_not_comparable(self):
        previous_map = [{**self.previous_map[0], "PREGUNTA_CANONICA_ID": "NO_EQUIVALENTE"}]
        result = resumen_historico([response(self.current_map[0])], self.current_map, [response(previous_map[0])], previous_map)
        self.assertEqual(result["ids_comparables"], [])

    def test_record_approval_is_required_in_both_years(self):
        for current_flag, previous_flag in [("NO", "SÍ"), ("SÍ", "NO")]:
            with self.subTest(current=current_flag, previous=previous_flag):
                result = resumen_historico(
                    [{**response(self.current_map[0]), "COMPARABLE_HISTORICO": current_flag}], self.current_map,
                    [{**response(self.previous_map[0]), "COMPARABLE_HISTORICO": previous_flag}], self.previous_map)
                self.assertEqual(result["ids_comparables"], [])
                self.assertIsNone(result["indice_2025"])
                self.assertIsNone(result["indice_2026"])

    def test_comments_follow_both_filters_and_do_not_inflate_counters(self):
        comment = {"SECCION_ID": "ESC_07_INSTALACIONES", "TIPO_DATO": "TEXTO",
                   "PREGUNTA_ID": "INS_COMENTARIO", "TEXTO_CORTO": "Aspectos por mejorar",
                   "RESPUESTA_ORIGINAL": "Mejorar la plataforma", "FILA_ORIGEN": 1,
                   "CARRERA_BASE": "Derecho", "MODALIDAD_ACADEMICA": "Presencial"}
        rows = [response(self.current_map[0]), comment,
                {**comment, "FILA_ORIGEN": 2, "CARRERA_BASE": "Nutrición"},
                {**comment, "FILA_ORIGEN": 3, "MODALIDAD_ACADEMICA": "Ejecutiva"}]
        selected = resumen_actual(rows, self.current_map, "Derecho", "Presencial")
        self.assertEqual((selected["respuestas"], selected["encuestas_analizadas"]), (1, 1))
        self.assertEqual(len(selected["comentarios"]), 1)
        self.assertEqual(selected["comentarios"][0]["texto"], "Mejorar la plataforma")
        self.assertEqual(len(resumen_actual(rows, self.current_map)["comentarios"]), 3)

    def test_duplicate_or_additional_current_criteria_fail_explicitly(self):
        for invalid in [self.current_map + [self.current_map[0]], self.current_map[:-1],
                        self.current_map + [{**self.current_map[0], "PREGUNTA_CANONICA_ID": "OTRO_CRITERIO"}]]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                resumen_actual([], invalid)

    def test_stable_engine_scores_and_distributions_are_preserved(self):
        rows = [response(row, 25 * (i % 5), number=i + 1) for i, row in enumerate(self.current_map)]
        baseline = calcular_kpis_preview(rows)
        result = resumen_actual(rows, self.current_map)
        self.assertEqual(result["indice"], baseline["secciones"][0]["indice"])
        questions = {row["pregunta_id"]: row for row in baseline["preguntas"]}
        for row in result["criterios"]:
            for field in ["indice", "n", "favorable_pct", "neutral_pct", "desfavorable_pct"]:
                self.assertEqual(row[field], questions[row["pregunta_id"]][field])


class FacilitiesRoutesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        with patch("backend.REPOSITORIES.calidad.calidad_repository.CalidadRepository"):
            from backend.GATEWAY import facilities_api as api
        cls.api = api
        app = FastAPI()
        app.include_router(api.router)
        cls.client = TestClient(app)

    def test_http_current_and_history_receive_same_scope(self):
        now, before = mapping(), mapping(2025)
        def source(year, modality, program):
            self.assertEqual((modality, program), ("Escolarizadas", "Programa vigente"))
            definitions = now if year == "2026" else before
            return [response(row, 75 if year == "2026" else 50) for row in definitions], definitions
        params = {"servicio": "Programa vigente", "carrera": "Derecho", "modalidad_academica": "Presencial"}
        with patch.object(self.api, "_fuente", side_effect=source):
            current = self.client.get("/api/encuesta-calidad/instalaciones/actual", params=params)
            history = self.client.get("/api/encuesta-calidad/instalaciones/historico", params=params)
        self.assertEqual(current.status_code, 200)
        self.assertEqual(history.status_code, 200)
        self.assertEqual(current.json()["poblacion"], history.json()["poblacion"])
        self.assertEqual((history.json()["indice_2025"], history.json()["indice_2026"]), (50, 75))

    def test_global_http_without_career_and_map_validation(self):
        definitions = mapping()
        with patch.object(self.api, "_fuente", return_value=([response(row) for row in definitions], definitions)):
            result = self.client.get("/api/encuesta-calidad/instalaciones/actual")
        self.assertEqual(result.status_code, 200)
        self.assertIsNone(result.json()["poblacion"]["carrera"])
        self.assertEqual(len(result.json()["criterios"]), 10)
        with patch.object(self.api, "_fuente", return_value=([], definitions[:-1])):
            result = self.client.get("/api/encuesta-calidad/instalaciones/actual")
        self.assertEqual(result.status_code, 409)

    def test_historical_source_failure_does_not_break_current_endpoint(self):
        from fastapi.testclient import TestClient
        definitions = mapping()
        def source(year, modality, program):
            if year == "2025":
                raise RuntimeError("Historical source unavailable")
            return [response(row) for row in definitions], definitions
        with TestClient(self.client.app, raise_server_exceptions=False) as client, patch.object(self.api, "_fuente", side_effect=source):
            self.assertEqual(client.get("/api/encuesta-calidad/instalaciones/historico").status_code, 500)
            current = client.get("/api/encuesta-calidad/instalaciones/actual")
            self.assertEqual(current.status_code, 200)
            self.assertEqual(current.json()["indice"], 75)


if __name__ == "__main__":
    unittest.main()
