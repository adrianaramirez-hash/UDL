import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from backend.SERVICES.calidad.continuing_education import CRITERIOS, BINARIOS, resumen_actual
from backend.SERVICES.calidad.kpi_service import calcular_kpis_preview


def mapping():
    definitions = [(key, name, "BINARIA" if key in BINARIOS else "LIKERT") for key, name in CRITERIOS.items()]
    definitions += [("EC_USO", "Participación", "BINARIA"), ("EC_MEDIO", "Medio de difusión", "CATEGORICA"), ("EC_MEJORA", "Sugerencias de mejora", "TEXTO")]
    return [{"SECCION_ID": "ESC_06_EDUC_CONT", "PREGUNTA_ID": f"ESC_{key}", "PREGUNTA_CANONICA_ID": key,
             "TEXTO_CORTO": name, "TIPO_DATO": kind, "COMPARABLE_HISTORICO": "NO",
             "ESCALA_ID": "CALIDAD_5" if kind == "LIKERT" else "BINARIA_SI_NO" if kind == "BINARIA" else "",
             "CALCULA_KPI_PREGUNTA": "SI" if kind in {"LIKERT", "BINARIA"} else "NO",
             "ENTRA_PROMEDIO_SECCION": "SI" if kind == "LIKERT" else "NO"} for key, name, kind in definitions]


def response(definition, value=75, number=1, career="Derecho", modality="Presencial"):
    numeric = definition["TIPO_DATO"] in {"LIKERT", "BINARIA"}
    return {**definition, "FILA_ORIGEN": number, "CARRERA_BASE": career, "MODALIDAD_ACADEMICA": modality,
            "SECCION_NOMBRE": "Educación Continua", "RESPUESTA_VALIDA": "SI" if value is not None else "NO",
            "INCLUIR_EN_KPI": "SI" if numeric and value is not None else "NO",
            "RESPUESTA_ORIGINAL": str(value) if value is not None else "",
            "RESPUESTA_NORMALIZADA": str(value) if value is not None else "",
            "INDICE_100": value if numeric else None,
            "VALOR_BASE": None if not numeric or value is None else value / 100 if definition["TIPO_DATO"] == "BINARIA" else value / 25 + 1}


def population(definitions, number=1, career="Derecho", modality="Presencial", score=75, usage=100, recommendation=100, knowledge=100, where=100, media="Correo institucional"):
    values = {**{key: score if usage == 100 else None for key in CRITERIOS if key not in BINARIOS},
              "EC_RECOMIENDA": recommendation if usage == 100 else None,
              "EC_CONOCE": knowledge, "EC_SABE_DONDE": where, "EC_USO": usage,
              "EC_MEDIO": media if usage == 100 else None,
              "EC_MEJORA": f"Comentario de {career} / {modality}" if usage == 100 else None}
    return [response(row, values[row["PREGUNTA_CANONICA_ID"]], number, career, modality) for row in definitions]


class ContinuingEducationTests(unittest.TestCase):
    def setUp(self):
        self.mapa = mapping()
        self.definitions = {r["PREGUNTA_CANONICA_ID"]: r for r in self.mapa}

    def test_global_has_exactly_six_named_indicators_without_combined_index(self):
        result = resumen_actual(population(self.mapa), self.mapa)
        self.assertEqual((result["encuestas_analizadas"], result["respuestas"]), (1, 6))
        self.assertEqual(len(result["criterios"]), 6)
        self.assertEqual({r["id"] for r in result["criterios"]}, set(CRITERIOS))
        self.assertTrue(all(r["nombre"].strip() for r in result["criterios"]))
        self.assertNotIn("indice", result)
        self.assertIsNone(result["poblacion"]["carrera"])

    def test_participation_and_media_do_not_change_scores_denominators_or_distributions(self):
        rows = population(self.mapa)
        baseline = resumen_actual(rows, self.mapa)
        ids = {self.definitions[k]["PREGUNTA_ID"] for k in ("EC_USO", "EC_MEDIO")}
        selected = resumen_actual([r for r in rows if r["PREGUNTA_ID"] not in ids], self.mapa)
        self.assertEqual(selected["criterios"], baseline["criterios"])
        self.assertEqual(selected["respuestas"], 6)
        stable = {r["pregunta_id"]: r for r in calcular_kpis_preview(rows[:3])["preguntas"]}
        for criterion in selected["criterios"][:3]:
            self.assertEqual(criterion["resultado"], stable[criterion["pregunta_id"]]["indice"])

    def test_recommendation_is_yes_percentage_not_nps_or_net_yes_minus_no(self):
        rows = [response(self.definitions["EC_RECOMIENDA"], value, i) for i, value in enumerate((100, 100, 100, 0), 1)]
        result = resumen_actual(rows, self.mapa)["criterios"][3]
        self.assertEqual((result["resultado"], result["unidad"], result["binaria"]["si"], result["binaria"]["no"]), (75, "porcentaje", 3, 1))
        self.assertNotEqual(result["resultado"], 50)
        self.assertNotIn("nps", result)

    def test_all_six_indicators_use_their_own_valid_response_denominators(self):
        rows = [response(self.definitions["EC_CALIDAD"], 100, 1), response(self.definitions["EC_CALIDAD"], None, 2),
                response(self.definitions["EC_CONOCE"], 100, 1), response(self.definitions["EC_CONOCE"], 0, 2),
                response(self.definitions["EC_SABE_DONDE"], 0, 3)]
        result = resumen_actual(rows, self.mapa)
        self.assertEqual((result["encuestas_analizadas"], result["respuestas"]), (3, 4))
        self.assertEqual(result["criterios"][0]["n"], 1)
        self.assertEqual(result["criterios"][4]["resultado"], 50)
        self.assertEqual(result["criterios"][5]["resultado"], 0)
        self.assertIsNone(result["criterios"][1]["resultado"])

    def test_participation_uses_valid_binary_responses_only(self):
        rows = [response(self.definitions["EC_USO"], value, i) for i, value in enumerate((100, 0, 0, None, 75), 1)]
        result = resumen_actual(rows, self.mapa)
        self.assertEqual(result["participacion"], {"si": 1, "no": 2, "total_valido": 3, "si_pct": 33.33, "no_pct": 66.67})
        self.assertEqual(result["encuestas_analizadas"], 3)
        self.assertEqual(result["respuestas"], 0)

    def test_media_is_descriptive_with_own_denominator_and_no_missing_other_bucket(self):
        definition = self.definitions["EC_MEDIO"]
        rows = [response(definition, value, i) for i, value in enumerate(("Correo", "Correo", "Redes", "", None), 1)]
        result = resumen_actual(rows, self.mapa)
        self.assertEqual(result["medios"]["n"], 3)
        self.assertEqual({r["nombre"]: r["n"] for r in result["medios"]["valores"]}, {"Correo": 2, "Redes": 1})
        self.assertEqual(result["respuestas"], 0)
        self.assertNotIn("Otro", str(result["medios"]))

    def test_career_modality_participation_filters_recalculate_every_metric_and_restore_global(self):
        rows = (population(self.mapa, score=100) + population(self.mapa, 2, "Nutrición", "Ejecutiva", score=25, usage=0, knowledge=0, where=0))
        original = copy.deepcopy(rows)
        baseline = resumen_actual(rows, self.mapa)
        for filters in ({"carrera": "Derecho"}, {"modalidad_academica": "Presencial"}, {"participacion": "si"},
                        {"carrera": "Derecho", "modalidad_academica": "Presencial", "participacion": "si"}):
            selected = resumen_actual(rows, self.mapa, **filters)
            self.assertEqual((selected["encuestas_analizadas"], selected["respuestas"]), (1, 6))
            self.assertTrue(all(r["resultado"] == 100 for r in selected["criterios"]))
            self.assertEqual(selected["participacion"]["si_pct"], 100)
            self.assertEqual(selected["medios"]["n"], 1)
            self.assertEqual(len(selected["comentarios"]), 1)
        no = resumen_actual(rows, self.mapa, participacion="no")
        self.assertEqual((no["respuestas"], no["encuestas_analizadas"]), (2, 1))
        self.assertTrue(all(r["resultado"] is None for r in no["criterios"][:4]))
        self.assertEqual(resumen_actual(rows, self.mapa), baseline)
        self.assertEqual(rows, original)

    def test_zero_and_no_responses_are_distinct(self):
        empty = resumen_actual([], self.mapa)
        self.assertTrue(all(r["resultado"] is None for r in empty["criterios"]))
        self.assertIsNone(empty["participacion"]["si_pct"])
        zero = resumen_actual(population(self.mapa, score=0, recommendation=0, knowledge=0, where=0), self.mapa)
        self.assertTrue(all(r["resultado"] == 0 for r in zero["criterios"]))

    def test_general_2025_never_creates_equivalence_even_with_identical_text(self):
        general = {**self.mapa[0], "SECCION_ID": "ESC25_03_SERVICIOS", "PREGUNTA_ID": "ESC25_SER_EDUC_CONT",
                   "PREGUNTA_CANONICA_ID": "SER_EDUC_CONT_2025", "TEXTO_CORTO": CRITERIOS["EC_CALIDAD"]}
        rows = population(self.mapa)
        self.assertEqual(resumen_actual(rows, self.mapa), resumen_actual(rows + [response(general, 0)], self.mapa + [general]))
        result = resumen_actual(rows, self.mapa)
        self.assertTrue(all(r["historico_disponible"] is False and "delta" not in r and "indice_2025" not in r for r in result["criterios"]))

    def test_comments_only_use_approved_question_and_active_population_without_semantic_selection(self):
        rows = population(self.mapa) + population(self.mapa, 2, "Nutrición", "Ejecutiva")
        selected = resumen_actual(rows, self.mapa, carrera="Derecho")
        self.assertEqual(len(selected["comentarios"]), 1)
        self.assertIn("Derecho", selected["comentarios"][0]["texto"])
        self.assertEqual(selected["comentarios"][0]["pregunta_id"], self.definitions["EC_MEJORA"]["PREGUNTA_ID"])

    def test_invalid_catalog_and_participation_filter_fail_explicitly(self):
        for invalid in (self.mapa[1:], self.mapa + [self.mapa[0]],
                        self.mapa + [{**self.mapa[0], "PREGUNTA_CANONICA_ID": "EXTRA"}],
                        [{**r, "TIPO_DATO": "LIKERT"} if r["PREGUNTA_CANONICA_ID"] == "EC_RECOMIENDA" else r for r in self.mapa]):
            with self.assertRaises(ValueError):
                resumen_actual([], invalid)
        with self.assertRaises(ValueError):
            resumen_actual([], self.mapa, participacion="unknown")

    def test_audit_preserves_empty_scores_and_unavailable_history_without_zero(self):
        from backend.TESTS.audit_continuing_education import audit
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "capture.json", Path(directory) / "audit.md"
            source.write_text(json.dumps({"2026": {"registros": [], "mapa": self.mapa}}), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                audit(source, output)
            text = output.read_text(encoding="utf-8")
            self.assertIn("Encuestas analizadas: **0**", text)
            self.assertIn("Recomendación NO se calcula como NPS", text)
            self.assertIn("no tienen comparación histórica válida", text)
            self.assertEqual(text.count("| No | Sin respuestas en este periodo |"), 6)


class ContinuingEducationRoutesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        with patch("backend.REPOSITORIES.calidad.calidad_repository.CalidadRepository"):
            from backend.GATEWAY import continuing_education_api as api
        cls.api = api
        app = FastAPI()
        app.include_router(api.router)
        cls.client = TestClient(app)

    def test_http_global_filters_and_no_historical_endpoint(self):
        definitions = mapping()
        with patch.object(self.api, "_fuente", return_value=(population(definitions), definitions)) as loader:
            result = self.client.get("/api/encuesta-calidad/educacion-continua/actual")
            self.assertEqual(result.status_code, 200)
            self.assertEqual(len(result.json()["criterios"]), 6)
            selected = self.client.get("/api/encuesta-calidad/educacion-continua/actual", params={"carrera": "Derecho", "modalidad_academica": "Presencial", "participacion": "si"})
            self.assertEqual(selected.json()["respuestas"], 6)
            self.assertEqual(self.client.get("/api/encuesta-calidad/educacion-continua/historico").status_code, 404)
            self.assertTrue(all(call.args[0] == "2026" for call in loader.call_args_list))


if __name__ == "__main__":
    unittest.main()
