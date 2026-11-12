import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from backend.SERVICES.calidad.language_center import CRITERIOS, resumen_actual
from backend.SERVICES.calidad.kpi_service import calcular_kpis_preview


def mapping():
    definitions = [(key, name, "LIKERT") for key, name in CRITERIOS.items()]
    definitions += [("IDI_USO", "Uso del servicio", "BINARIA"), ("IDI_IDIOMA", "Idioma estudiado", "CATEGORICA"),
                    ("IDI_TIPO_CURSO", "Tipo de curso", "CATEGORICA"), ("IDI_PROFESOR", "Profesor actual", "CATEGORICA"),
                    ("IDI_MENOS_GUSTA", "Aspectos por mejorar", "TEXTO"), ("IDI_MAS_GUSTA", "Aspectos positivos", "TEXTO")]
    return [{"SECCION_ID": "ESC_05_IDIOMAS", "PREGUNTA_ID": f"ESC_{key}", "PREGUNTA_CANONICA_ID": key,
             "TEXTO_CORTO": name, "TIPO_DATO": kind, "COMPARABLE_HISTORICO": "NO",
             "ESCALA_ID": "ACUERDO_5" if kind == "LIKERT" else "BINARIA_SI_NO" if kind == "BINARIA" else "",
             "CALCULA_KPI_PREGUNTA": "SI" if kind in {"LIKERT", "BINARIA"} else "NO",
             "ENTRA_PROMEDIO_SECCION": "SI" if key in CRITERIOS else "NO"} for key, name, kind in definitions]


def response(definition, value=75, number=1, career="Derecho", modality="Presencial"):
    numeric = definition["TIPO_DATO"] in {"LIKERT", "BINARIA"}
    return {**definition, "FILA_ORIGEN": number, "CARRERA_BASE": career, "MODALIDAD_ACADEMICA": modality,
            "SECCION_NOMBRE": "Centro de Idiomas", "RESPUESTA_VALIDA": "SI" if value is not None else "NO",
            "INCLUIR_EN_KPI": "SI" if numeric and value is not None else "NO",
            "RESPUESTA_ORIGINAL": str(value) if value is not None else "",
            "RESPUESTA_NORMALIZADA": str(value) if value is not None else "",
            "INDICE_100": value if numeric else None,
            "VALOR_BASE": None if not numeric or value is None else value / 100 if definition["TIPO_DATO"] == "BINARIA" else value / 25 + 1}


def population(definitions, number=1, career="Derecho", modality="Presencial", language="Inglés", course="Intensivo", score=75, usage=100):
    values = {**{key: score for key in CRITERIOS}, "IDI_USO": usage, "IDI_IDIOMA": language, "IDI_TIPO_CURSO": course,
              "IDI_PROFESOR": "Docente descriptivo", "IDI_MENOS_GUSTA": f"Comentario de {career} / {language} / {course}",
              "IDI_MAS_GUSTA": f"Comentario de {career} / {language} / {course}"}
    return [response(row, values[row["PREGUNTA_CANONICA_ID"]], number, career, modality) for row in definitions]


class LanguageCenterTests(unittest.TestCase):
    def setUp(self):
        self.mapa = mapping()
        self.definitions = {r["PREGUNTA_CANONICA_ID"]: r for r in self.mapa}

    def test_global_without_filters_has_three_unique_named_criteria_and_separate_context(self):
        result = resumen_actual(population(self.mapa), self.mapa)
        self.assertEqual((result["encuestas_analizadas"], result["respuestas"]), (1, 3))
        self.assertEqual(len(result["criterios"]), 3)
        self.assertEqual({r["id"] for r in result["criterios"]}, set(CRITERIOS))
        self.assertTrue(all(r["nombre"] for r in result["criterios"]))
        self.assertEqual(result["uso"]["si_pct"], 100)
        self.assertEqual(result["idiomas"]["valores"][0]["nombre"], "Inglés")
        self.assertEqual(result["tipos_curso"]["valores"][0]["nombre"], "Intensivo")
        self.assertNotIn("indice", result)

    def test_context_and_teacher_do_not_enter_scores_or_response_count(self):
        quantitative = [response(row, 25 * index) for index, row in enumerate(self.mapa[:3])]
        baseline = resumen_actual(quantitative, self.mapa)
        result = resumen_actual(quantitative + population(self.mapa)[3:], self.mapa)
        self.assertEqual(result["criterios"], baseline["criterios"])
        self.assertEqual(result["respuestas"], 3)
        self.assertEqual(result["profesores"], {"n": 1, "nombres": ["Docente descriptivo"]})
        stable = {r["pregunta_id"]: r for r in calcular_kpis_preview(quantitative)["preguntas"]}
        for row in result["criterios"]:
            for key in ("indice", "n", "favorable_pct", "neutral_pct", "desfavorable_pct"):
                self.assertEqual(row[key], stable[row["pregunta_id"]][key])

    def test_usage_uses_only_valid_binary_denominator_without_creating_satisfaction(self):
        definition = self.definitions["IDI_USO"]
        rows = [response(definition, value, i) for i, value in enumerate((100, 0, 0, None, 75), 1)]
        result = resumen_actual(rows, self.mapa)
        self.assertEqual(result["uso"], {"si": 1, "no": 2, "total_valido": 3, "si_pct": 33.33, "no_pct": 66.67})
        self.assertEqual(result["respuestas"], 0)
        self.assertEqual(result["encuestas_analizadas"], 3)
        self.assertTrue(all(r["indice"] is None for r in result["criterios"]))

    def test_language_and_course_denominators_exclude_missing_without_creating_other(self):
        rows = [response(self.definitions["IDI_IDIOMA"], value, i) for i, value in enumerate(("Inglés", "Francés", "Inglés", "", None), 1)]
        rows += [response(self.definitions["IDI_TIPO_CURSO"], "Intensivo")]
        result = resumen_actual(rows, self.mapa)
        self.assertEqual(result["idiomas"]["n"], 3)
        self.assertEqual({r["nombre"]: r["n"] for r in result["idiomas"]["valores"]}, {"Inglés": 2, "Francés": 1})
        self.assertNotIn("Otro", str(result["idiomas"]))
        self.assertEqual(result["tipos_curso"]["n"], 1)
        self.assertEqual(result["tipos_curso"]["valores"][0]["porcentaje"], 100)

    def test_optional_filters_recalculate_all_rows_of_same_survey_and_comments(self):
        rows = population(self.mapa, score=100) + population(self.mapa, 2, "Nutrición", "Ejecutiva", "Francés", "Sabatino", 25, 0)
        original = copy.deepcopy(rows)
        global_result = resumen_actual(rows, self.mapa)
        for filters in [{"carrera": "Derecho"}, {"modalidad_academica": "Presencial"}, {"idioma": "Inglés"}, {"tipo_curso": "Intensivo"},
                        {"carrera": "Derecho", "modalidad_academica": "Presencial", "idioma": "Inglés", "tipo_curso": "Intensivo"}]:
            result = resumen_actual(rows, self.mapa, **filters)
            self.assertEqual((result["encuestas_analizadas"], result["respuestas"]), (1, 3))
            self.assertTrue(all(r["indice"] == 100 for r in result["criterios"]))
            self.assertEqual(result["uso"]["si_pct"], 100)
            self.assertEqual(len(result["comentarios"]), 2)
            self.assertTrue(all("Derecho" in r["texto"] for r in result["comentarios"]))
            self.assertEqual(result["idiomas"]["n"], 1)
            self.assertEqual(result["tipos_curso"]["n"], 1)
        self.assertEqual(resumen_actual(rows, self.mapa), global_result)
        self.assertEqual(rows, original)

    def test_valid_zero_and_empty_states_remain_distinct(self):
        result = resumen_actual([], self.mapa)
        self.assertTrue(all(r["indice"] is None for r in result["criterios"]))
        self.assertIsNone(result["uso"]["si_pct"])
        zero = resumen_actual(population(self.mapa, score=0, usage=0), self.mapa)
        self.assertTrue(all(r["indice"] == 0 for r in zero["criterios"]))
        self.assertEqual(zero["uso"]["si_pct"], 0)
        absent = resumen_actual(population(self.mapa), self.mapa, idioma="Francés")
        self.assertEqual(absent["encuestas_analizadas"], 0)
        self.assertTrue(all(r["indice"] is None for r in absent["criterios"]))

    def test_general_2025_question_cannot_supply_a_historical_value_or_extra_criterion(self):
        general = {**self.mapa[0], "SECCION_ID": "ESC25_03_SERVICIOS", "PREGUNTA_ID": "ESC25_SER_CENTRO_IDIOMAS",
                   "PREGUNTA_CANONICA_ID": "SER_CENTRO_IDIOMAS_2025", "TEXTO_CORTO": CRITERIOS["IDI_GUSTO_CLASE"]}
        before = resumen_actual(population(self.mapa), self.mapa)
        after = resumen_actual(population(self.mapa) + [response(general, 0)], self.mapa + [general])
        self.assertEqual(after, before)
        self.assertTrue(all(r["historico_disponible"] is False for r in after["criterios"]))
        self.assertTrue(all("indice_2025" not in r and "delta" not in r for r in after["criterios"]))

    def test_counts_include_valid_descriptive_and_text_only_forms_without_scoring(self):
        rows = [response(self.definitions["IDI_IDIOMA"], "Inglés", 1), response(self.definitions["IDI_MAS_GUSTA"], "Una respuesta contextual", 2)]
        result = resumen_actual(rows, self.mapa)
        self.assertEqual((result["encuestas_analizadas"], result["respuestas"]), (2, 0))

    def test_comments_keep_both_original_questions_without_semantic_selection(self):
        result = resumen_actual(population(self.mapa), self.mapa)
        self.assertEqual(len(result["comentarios"]), 2)
        self.assertEqual({r["pregunta_id"] for r in result["preguntas_comentarios"]}, {self.definitions[key]["PREGUNTA_ID"] for key in ("IDI_MAS_GUSTA", "IDI_MENOS_GUSTA")})

    def test_catalog_rejects_missing_duplicate_extra_quantitative_and_scored_context(self):
        for invalid in [self.mapa[1:], self.mapa + [self.mapa[0]],
                        self.mapa + [{**self.mapa[0], "PREGUNTA_CANONICA_ID": "EXTRA"}],
                        [{**r, "ENTRA_PROMEDIO_SECCION": "SI"} if r["PREGUNTA_CANONICA_ID"] == "IDI_PROFESOR" else r for r in self.mapa]]:
            with self.assertRaises(ValueError):
                resumen_actual([], invalid)

    def test_audit_preserves_empty_scores_and_absent_history_without_zero_or_other(self):
        from backend.TESTS.audit_language_center import audit
        capture = {"2026": {"registros": [], "mapa": self.mapa}}
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "capture.json", Path(directory) / "audit.md"
            source.write_text(json.dumps(capture), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                audit(source, output)
            report = output.read_text(encoding="utf-8")
            self.assertIn("Los tres criterios especializados 2026 no tienen comparación histórica válida", report)
            self.assertIn("Encuestas analizadas: **0**", report)
            self.assertEqual(report.count("| 0 | 0 | — | No | Sin respuestas en este periodo |"), 3)


class LanguageCenterRoutesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        with patch("backend.REPOSITORIES.calidad.calidad_repository.CalidadRepository"):
            from backend.GATEWAY import language_center_api as api
        cls.api = api
        app = FastAPI()
        app.include_router(api.router)
        cls.client = TestClient(app)

    def test_http_global_optional_filters_and_no_historical_endpoint(self):
        definitions = mapping()
        rows = population(definitions)
        with patch.object(self.api, "_fuente", return_value=(rows, definitions)) as loader:
            result = self.client.get("/api/encuesta-calidad/centro-idiomas/actual")
            self.assertEqual(result.status_code, 200)
            params = {"carrera": "Derecho", "modalidad_academica": "Presencial", "idioma": "Inglés", "tipo_curso": "Intensivo"}
            selected = self.client.get("/api/encuesta-calidad/centro-idiomas/actual", params=params)
            self.assertEqual(selected.status_code, 200)
            self.assertEqual(selected.json()["respuestas"], 3)
            self.assertEqual(self.client.get("/api/encuesta-calidad/centro-idiomas/historico").status_code, 404)
            self.assertTrue(all(call.args[0] == "2026" for call in loader.call_args_list))


if __name__ == "__main__":
    unittest.main()
