import copy
import unittest
from unittest.mock import patch

from backend.SERVICES.calidad.institutional_services import resumen_actual, resumen_historico, nombre_desde_encabezado
from backend.SERVICES.calidad.kpi_service import calcular_kpis_preview


def definition(name="Admisiones", canonical="SER_ADMISIONES", year=2026, comparable="SÍ"):
    prefix = "ESC" if year == 2026 else "ESC25"
    return {"SECCION_ID": f"{prefix}_03_SERVICIOS", "PREGUNTA_ID": f"{prefix}_{canonical}",
            "PREGUNTA_CANONICA_ID": canonical, "TEXTO_CORTO": name,
            "ENTRA_PROMEDIO_SECCION": "SÍ", "CALCULA_KPI_PREGUNTA": "SÍ", "COMPARABLE_HISTORICO": comparable}


def response(definition, value=75, career="Derecho", modality="Presencial", number=1):
    return {**definition, "FILA_ORIGEN": number, "SECCION_NOMBRE": "Servicios institucionales",
            "TIPO_DATO": "LIKERT", "INDICE_100": value, "VALOR_BASE": value / 25 + 1 if value is not None else None,
            "INCLUIR_EN_KPI": "SÍ" if value is not None else "NO", "CARRERA_BASE": career,
            "MODALIDAD_ACADEMICA": modality, "RESPUESTA_ORIGINAL": str(value)}


class InstitutionalServicesTests(unittest.TestCase):
    def test_global_without_career_uses_all_services(self):
        first, second = definition(), definition("Becas", "SER_BECAS")
        result = resumen_actual([response(first, 50), response(second, 100)], [first, second])
        self.assertEqual(result["indice"], 75)
        self.assertEqual(len(result["servicios"]), 2)
        self.assertEqual(result["respuestas"], 2)
        self.assertEqual(result["encuestas_con_evaluacion"], 1)
        self.assertIsNone(result["poblacion"]["carrera"])

    def test_career_and_modality_scope_filter_both_years(self):
        current, old = definition(), definition(year=2025)
        a = [response(current, 100), response(current, 25, "Nutrición", number=2),
             response(current, 0, modality="Ejecutiva", number=3)]
        b = [response(old, 75), response(old, 0, "Nutrición", number=2),
             response(old, 100, modality="Ejecutiva", number=3)]
        actual = resumen_actual(a, [current], "Derecho", "Presencial")
        history = resumen_historico(a, [current], b, [old], "Derecho", "Presencial")
        self.assertEqual(actual["indice"], 100)
        self.assertEqual((history["indice_2025"], history["indice_2026"]), (75, 100))
        self.assertEqual(actual["poblacion"], history["poblacion"])
        # No fallback to global history when the selected population is absent.
        absent = resumen_historico(a, [current], b, [old], "Carrera futura", "Presencial")
        self.assertIsNone(absent["indice_2025"])
        self.assertIsNone(absent["indice_2026"])

    def test_service_without_history_remains_in_current_catalog(self):
        first, new = definition(), definition("Servicio nuevo", "SER_NUEVO")
        old = definition(year=2025)
        a = [response(first), response(new, 100)]
        actual = resumen_actual(a, [first, new])
        history = resumen_historico(a, [first, new], [response(old)], [old])
        self.assertEqual(len(actual["servicios"]), 2)
        self.assertIsNone(history["servicios"][1]["indice_2025"])
        self.assertEqual(actual["servicios"][1]["indice"], 100)

    def test_no_responses_stays_null_and_zero_is_a_valid_score(self):
        first, empty = definition(), definition("Becas", "SER_BECAS")
        result = resumen_actual([response(first, 0), response(empty, None)], [first, empty])
        self.assertEqual(result["servicios"][0]["indice"], 0)
        self.assertIsNone(result["servicios"][1]["indice"])
        self.assertEqual(result["servicios"][1]["n"], 0)
        self.assertEqual(result["encuestas_con_evaluacion"], 1)

    def test_institutional_label_comes_from_raw_without_changing_identity(self):
        original = definition("Servicio institucional 09", "SER_PARAMEDICO")
        corrected = nombre_desde_encabezado(original, " [Paramédico]")
        result = resumen_actual([response(original)], [corrected])
        self.assertEqual(result["servicios"][0]["nombre"], "Paramédico")
        self.assertEqual(result["servicios"][0]["id"], "SER_PARAMEDICO")
        self.assertEqual(original["TEXTO_CORTO"], "Servicio institucional 09")
        proper = definition("Paramédico", "SER_PARAMEDICO")
        self.assertEqual(nombre_desde_encabezado(proper, "[Otro nombre]"), proper)
        old = definition("Paramédico", "SER_PARAMEDICO", 2025)
        previous = [response(old)]
        self.assertEqual(resumen_historico([response(original)], [original], previous, [old]),
                         resumen_historico([response(original)], [corrected], previous, [old]))

    def test_history_cannot_create_additional_services(self):
        current = definition()
        old, historical_only = definition(year=2025), definition("Solo histórico", "SER_SOLO_2025", 2025)
        history = resumen_historico([response(current)], [current], [response(old), response(historical_only)], [old, historical_only])
        self.assertEqual([row["id"] for row in history["servicios"]], ["SER_ADMISIONES"])

    def test_empty_or_duplicate_catalog_names_fail_explicitly(self):
        for second in [definition(""), definition(), definition("Admisiones", "OTRO_ID")]:
            with self.subTest(second=second), self.assertRaises(ValueError):
                resumen_actual([], [definition(), second])

    def test_global_history_uses_only_approved_canonical_intersection(self):
        first, noncomparable, missing = definition(), definition("Otro", "SER_OTRO", comparable="NO"), definition("Sin dato", "SER_VACIO")
        old, old_other, old_missing = definition(year=2025), definition("Otro", "SER_OTRO", 2025), definition("Sin dato", "SER_VACIO", 2025)
        current = [response(first, 75), response(noncomparable, 0), response(missing, 100)]
        previous = [response(old, 50), response(old_other, 100)]
        result = resumen_historico(current, [first, noncomparable, missing], previous, [old, old_other, old_missing])
        self.assertEqual(result["criterios"], ["SER_ADMISIONES"])
        self.assertEqual((result["indice_2025"], result["indice_2026"]), (50, 75))

    def test_similar_names_do_not_create_equivalence(self):
        first, old = definition(), definition("Admisiones", "CANONICA_DIFERENTE", 2025)
        result = resumen_historico([response(first)], [first], [response(old)], [old])
        self.assertEqual(result["criterios"], [])
        self.assertIsNone(result["servicios"][0]["indice_2025"])

    def test_normalized_record_approval_is_also_required(self):
        first, old = definition(), definition(year=2025)
        result = resumen_historico([{**response(first), "COMPARABLE_HISTORICO": "NO"}], [first], [response(old)], [old])
        self.assertIsNone(result["indice_2026"])
        self.assertIsNone(result["servicios"][0]["indice_2025"])

    def test_reuses_stable_engine_without_mutating_records_or_personnel(self):
        first = definition()
        rows = [{**response(first, 75), "ANIO": 2026, "FUENTE_ID": "ESC_MAIN"}]
        before = copy.deepcopy(rows)
        with patch("backend.SERVICES.calidad.kpi_service.cargar_personas_evaluadas", return_value=[]) as loader:
            result = resumen_actual(rows, [first])
            loader.assert_not_called()
            baseline = calcular_kpis_preview(rows)
        self.assertEqual(rows, before)
        self.assertEqual(result["indice"], baseline["secciones"][0]["indice"])
        for field in ["indice", "n", "favorable_pct", "neutral_pct", "desfavorable_pct"]:
            self.assertEqual(result["servicios"][0][field], baseline["preguntas"][0][field])


class InstitutionalRoutesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The legacy router initializes its repository at import time. Keep this
        # HTTP contract test offline; source access is audited separately.
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        with patch("backend.REPOSITORIES.calidad.calidad_repository.CalidadRepository"):
            from backend.GATEWAY import institutional_services_api as api
        cls.api = api
        app = FastAPI()
        app.include_router(api.router)
        cls.client = TestClient(app)

    def test_http_endpoints_keep_the_same_scope_and_history_is_independent(self):
        current, old = definition(), definition(year=2025)
        rows = [response(current, 100), response(current, 25, "Nutrición", number=2)]
        previous = [response(old, 75), response(old, 0, "Nutrición", number=2)]
        def source(year, modality, program):
            self.assertEqual(modality, "Escolarizadas")
            self.assertEqual(program, "Programa vigente")
            return (rows, [current]) if year == "2026" else (previous, [old])
        params = {"carrera": "Derecho", "modalidad_academica": "Presencial", "servicio": "Programa vigente"}
        with patch.object(self.api, "_fuente", side_effect=source):
            actual = self.client.get("/api/encuesta-calidad/servicios-institucionales/actual", params=params)
            history = self.client.get("/api/encuesta-calidad/servicios-institucionales/historico", params=params)
        self.assertEqual(actual.status_code, 200)
        self.assertEqual(history.status_code, 200)
        self.assertEqual(actual.json()["indice"], 100)
        self.assertEqual(history.json()["indice_2025"], 75)
        self.assertEqual(actual.json()["poblacion"], history.json()["poblacion"])
        with patch.object(self.api, "_fuente", return_value=(rows, [current])):
            self.assertEqual(self.client.get("/api/encuesta-calidad/servicios-institucionales/actual").status_code, 200)

    def test_program_filter_never_falls_back_to_global(self):
        mapping = definition()
        rows = [{**response(mapping, 75), "SERVICIO_PROGRAMA": "Derecho"},
                {**response(mapping, 100), "SERVICIO_PROGRAMA": "Nutrición"}]
        with patch.object(self.api, "_obtener_base_preview", return_value={"registros": rows}), patch.object(self.api, "cargar_mapa_preguntas", return_value=[mapping]):
            filtered, _ = self.api._fuente("2026", "Escolarizadas", "Derecho")
            absent, _ = self.api._fuente("2025", "Escolarizadas", "Programa inexistente")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["INDICE_100"], 75)
        self.assertEqual(absent, [])

    def test_http_paramedico_reads_its_mapped_raw_column(self):
        mapped = {**definition("Servicio institucional 09", "SER_PARAMEDICO"), "COLUMNA_ORIGEN": "BX"}
        headers = [""] * 76
        headers[75] = " [Paramédico]"
        with patch.object(self.api, "_fuente", return_value=([response(mapped)], [mapped])), \
             patch.object(self.api.repository, "obtener_estado_fuente", return_value={"SPREADSHEET_ID": "book", "HOJA_RESPUESTAS": "sheet"}), \
             patch.object(self.api, "leer_encabezados", return_value=headers):
            result = self.client.get("/api/encuesta-calidad/servicios-institucionales/actual")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["servicios"][0]["nombre"], "Paramédico")


if __name__ == "__main__":
    unittest.main()
