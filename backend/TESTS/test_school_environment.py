import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from backend.SERVICES.calidad.school_environment import CRITERIOS, resumen_actual, resumen_historico
from backend.SERVICES.calidad.kpi_service import calcular_kpis_preview


def mapping(year=2026):
    prefix = "ESC" if year == 2026 else "ESC25"
    definitions = [(key, name, "LIKERT", "SI", "SI") for key, name in CRITERIOS.items()]
    definitions += [("AMB_PROBLEMA", "Incidencia reportada", "BINARIA", "NO", "SI"),
                    ("AMB_DESC", "Descripción de incidencia", "TEXTO", "NO", "NO"),
                    ("AMB_COMENTARIO", "Sugerencias de ambiente", "TEXTO", "NO", "NO")]
    return [{"SECCION_ID": f"{prefix}_08_AMBIENTE", "PREGUNTA_ID": f"{prefix}_{key}",
             "PREGUNTA_CANONICA_ID": key, "TEXTO_CORTO": name, "TIPO_DATO": kind,
             "ENTRA_PROMEDIO_SECCION": enters, "CALCULA_KPI_PREGUNTA": calculates,
             "COMPARABLE_HISTORICO": "SI"} for key, name, kind, enters, calculates in definitions]


def response(definition, value=75, number=1, career="Derecho", modality="Presencial"):
    binary, text = definition["TIPO_DATO"] == "BINARIA", definition["TIPO_DATO"] == "TEXTO"
    return {**definition, "FILA_ORIGEN": number, "CARRERA_BASE": career,
            "MODALIDAD_ACADEMICA": modality, "SECCION_NOMBRE": "Ambiente escolar",
            "INDICE_100": None if text else value,
            "VALOR_BASE": None if text or value is None else value / 100 if binary else value / 25 + 1,
            "INCLUIR_EN_KPI": "SI" if value is not None and not text else "NO",
            "RESPUESTA_ORIGINAL": value if text else str(value)}


class SchoolEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.now, self.old = mapping(), mapping(2025)

    def test_global_has_exactly_three_unique_named_criteria_and_separate_incidence(self):
        rows = [response(row) for row in self.now[:3]] + [response(self.now[3], 100)]
        result = resumen_actual(rows, self.now)
        self.assertEqual((result["indice"], result["respuestas"], result["encuestas_analizadas"]), (75, 3, 1))
        self.assertEqual({r["id"] for r in result["criterios"]}, set(CRITERIOS))
        self.assertEqual(len(result["criterios"]), 3)
        self.assertTrue(all(r["nombre"].strip() for r in result["criterios"]))
        self.assertEqual(result["incidencia"]["si_pct"], 100)
        self.assertIsNone(result["poblacion"]["carrera"])

    def test_incidence_never_changes_index_counters_or_quantitative_distribution(self):
        quantitative = [response(row, 25 * i) for i, row in enumerate(self.now[:3])]
        baseline = resumen_actual(quantitative, self.now)
        for value in (0, 100):
            rows = quantitative + [{**response(self.now[3], value), "ENTRA_PROMEDIO_SECCION": "SI"}]
            result = resumen_actual(rows, self.now)
            for key in ("indice", "respuestas", "distribucion", "criterios"):
                self.assertEqual(result[key], baseline[key])
        self.assertEqual(baseline["indice"], calcular_kpis_preview(quantitative)["secciones"][0]["indice"])

    def test_incidence_denominator_uses_only_valid_binary_responses(self):
        rows = [response(self.now[3], value, i) for i, value in enumerate((100, 0, 0, None), 1)]
        rows += [{**response(self.now[3], 100, 5), "INCLUIR_EN_KPI": "NO"}, response(self.now[3], 75, 6)]
        result = resumen_actual(rows, self.now)
        self.assertEqual(result["incidencia"], {"si": 1, "no": 2, "total_valido": 3, "si_pct": 33.33, "no_pct": 66.67})
        self.assertEqual(result["encuestas_analizadas"], 3)
        self.assertEqual(result["respuestas"], 0)
        self.assertIsNone(result["indice"])

    def test_zero_incidence_is_distinct_from_absence_of_valid_responses(self):
        empty = resumen_actual([], self.now)
        self.assertIsNone(empty["incidencia"]["si_pct"])
        self.assertTrue(all(row["indice"] is None for row in empty["criterios"]))
        self.assertEqual(resumen_actual([response(self.now[3], 0)], self.now)["incidencia"]["si_pct"], 0)
        self.assertEqual(resumen_actual([response(self.now[0], 0)], self.now)["criterios"][0]["indice"], 0)

    def test_quantitative_denominators_and_survey_union_exclude_empty_and_text_only(self):
        rows = [response(self.now[0], 100, 1), response(self.now[0], 50, 2),
                response(self.now[1], None, 3), response(self.now[3], 0, 3),
                response(self.now[4], "Descripción contextual", 4)]
        result = resumen_actual(rows, self.now)
        self.assertEqual((result["respuestas"], result["encuestas_analizadas"]), (2, 3))
        self.assertEqual((result["criterios"][0]["indice"], result["criterios"][0]["encuestas_validas"]), (75, 2))

    def test_filters_recalculate_all_values_and_reset_to_global_without_mutation(self):
        rows = []
        for number, career, modality, value in [(1, "Derecho", "Presencial", 100), (2, "Nutrición", "Presencial", 0), (3, "Derecho", "Ejecutiva", 50)]:
            rows += [response(row, value, number, career, modality) for row in self.now[:3]]
            rows += [response(self.now[3], 100 if number == 1 else 0, number, career, modality)]
        original = copy.deepcopy(rows)
        global_result = resumen_actual(rows, self.now)
        for career, modality, expected in [("Derecho", "", 75), ("", "Ejecutiva", 50), ("Derecho", "Presencial", 100)]:
            current = resumen_actual(rows, self.now, career, modality)
            self.assertEqual(current["indice"], expected)
            self.assertLess(current["respuestas"], global_result["respuestas"])
        selected = resumen_actual(rows, self.now, "Derecho", "Presencial")
        self.assertEqual(selected["incidencia"]["si_pct"], 100)
        self.assertEqual(selected["distribucion"]["favorable_pct"], 100)
        self.assertEqual(resumen_actual(rows, self.now), global_result)
        self.assertEqual(rows, original)

    def test_history_uses_approved_ids_same_population_and_separate_percentage_points(self):
        now = [response(row, 100) for row in self.now[:4]]
        old = [response(row, 75 if i < 3 else 0) for i, row in enumerate(self.old[:4])]
        other = [response(row, 0, 2, "Nutrición") for row in self.now[:4]]
        history = resumen_historico(now + other, self.now, old, self.old, "Derecho", "Presencial")
        self.assertEqual((history["indice_2025"], history["indice_2026"]), (75, 100))
        self.assertEqual(len(history["ids_comparables"]), 3)
        self.assertEqual(history["incidencia"], {"comparable": True, "porcentaje_2025": 0, "porcentaje_2026": 100, "delta_pp": 100})
        absent = resumen_historico(now, self.now, old, self.old, "Derecho", "Ejecutiva")
        self.assertIsNone(absent["indice_2025"])
        self.assertIsNone(absent["incidencia"]["porcentaje_2025"])

    def test_missing_history_keeps_current_result_and_does_not_use_text_matching(self):
        self.old[0]["PREGUNTA_CANONICA_ID"] = "UNAPPROVED_ID"
        old = [response(row, 50) for row in self.old[:4]]
        now = [response(row, 75 if i < 3 else 0) for i, row in enumerate(self.now[:4])]
        result = resumen_historico(now, self.now, old, self.old)
        self.assertIsNone(result["criterios"][0]["indice_2025"])
        self.assertEqual(len(result["criterios"]), 3)
        self.assertEqual(resumen_actual(now, self.now)["criterios"][0]["indice"], 75)
        no_incidence_map = [row for row in self.old if row["PREGUNTA_CANONICA_ID"] != "AMB_PROBLEMA"]
        self.assertFalse(resumen_historico(now, self.now, old, no_incidence_map)["incidencia"]["comparable"])

    def test_record_and_map_approval_are_required_for_both_types_of_history(self):
        now = [response(row, 100) for row in self.now[:4]]
        old = [response(row, 0) for row in self.old[:4]]
        for target in (self.now, self.old, now, old):
            with self.subTest(target=target[0]["PREGUNTA_ID"]):
                original = copy.deepcopy(target)
                for row in target:
                    row["COMPARABLE_HISTORICO"] = "NO"
                result = resumen_historico(now, self.now, old, self.old)
                self.assertEqual(result["ids_comparables"], [])
                self.assertFalse(result["incidencia"]["comparable"])
                target[:] = original

    def test_global_history_uses_identical_intersection_of_quantitative_criteria(self):
        now = [response(self.now[0], 100), response(self.now[1], 0), response(self.now[3], 100)]
        old = [response(self.old[0], 50), response(self.old[3], 0)]
        result = resumen_historico(now, self.now, old, self.old)
        self.assertEqual(result["ids_comparables"], ["AMB_REL_PROF"])
        self.assertEqual((result["indice_2025"], result["indice_2026"]), (50, 100))

    def test_comments_include_description_and_suggestions_only_within_active_population(self):
        rows = [response(row, "Descripción o sugerencia contextual") for row in self.now[4:]]
        rows += [response(self.now[4], "Comentario de otra carrera", 2, "Nutrición")]
        result = resumen_actual(rows, self.now, "Derecho", "Presencial")
        self.assertEqual(len(result["comentarios"]), 2)
        self.assertEqual({r["pregunta_id"] for r in result["comentarios"]}, {r["PREGUNTA_ID"] for r in self.now[4:]})
        self.assertEqual(result["encuestas_analizadas"], 0)

    def test_catalog_rejects_duplicates_extra_criteria_and_incidence_in_average(self):
        for invalid in [self.now + [self.now[0]], self.now[1:],
                        self.now + [{**self.now[0], "PREGUNTA_CANONICA_ID": "EXTRA"}],
                        [{**r, "ENTRA_PROMEDIO_SECCION": "SI"} if r["PREGUNTA_CANONICA_ID"] == "AMB_PROBLEMA" else r for r in self.now]]:
            with self.assertRaises(ValueError):
                resumen_actual([], invalid)

    def test_audit_handles_empty_current_and_absent_historical_maps_without_zero_scores(self):
        from backend.TESTS.audit_school_environment import audit
        capture = {"2026": {"registros": [], "mapa": self.now}, "2025": {"registros": [], "mapa": []}}
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "capture.json", Path(directory) / "audit.md"
            source.write_text(json.dumps(capture), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                audit(source, output)
            text = output.read_text(encoding="utf-8")
            self.assertIn("Resultado general: **—**", text)
            self.assertIn("Criterios con histórico: 0; sin histórico: 3", text)
            self.assertIn("Histórico aprobado disponible: No", text)
            self.assertIn("La incidencia NO entra al Resultado general", text)


class SchoolEnvironmentRoutesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        with patch("backend.REPOSITORIES.calidad.calidad_repository.CalidadRepository"):
            from backend.GATEWAY import school_environment_api as api
        cls.api = api
        app = FastAPI()
        from backend.TESTS.auth_fixture import authorize_dg
        authorize_dg(app)
        app.include_router(api.router)
        cls.client = TestClient(app)

    def test_global_http_and_identical_history_filters(self):
        def source(year, modality, service):
            definitions = mapping(int(year))
            return [response(row, 100 if year == "2026" else 0) for row in definitions[:4]], definitions
        with patch.object(self.api, "_fuente", side_effect=source):
            global_result = self.client.get("/api/encuesta-calidad/ambiente-escolar/actual")
            params = {"carrera": "Derecho", "modalidad_academica": "Presencial"}
            now = self.client.get("/api/encuesta-calidad/ambiente-escolar/actual", params=params)
            history = self.client.get("/api/encuesta-calidad/ambiente-escolar/historico", params=params)
        self.assertEqual(global_result.status_code, 200)
        self.assertEqual(len(global_result.json()["criterios"]), 3)
        self.assertEqual(now.json()["poblacion"], history.json()["poblacion"])
        self.assertEqual(history.json()["incidencia"]["delta_pp"], 100)


if __name__ == "__main__":
    unittest.main()
