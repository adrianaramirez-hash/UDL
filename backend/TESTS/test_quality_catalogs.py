import unittest
from unittest.mock import patch

from backend.SERVICES.calidad.kpi_service import calcular_kpis_preview


def row(career="Derecho", modality="Presencial"):
    return {
        "FUENTE_ID": "ESC_MAIN", "ANIO": 2026, "FILA_ORIGEN": 1,
        "CARRERA_BASE": career, "SERVICIO_PROGRAMA": career,
        "MODALIDAD_ACADEMICA": modality, "MODALIDAD_AGRUPADA": "Escolarizadas",
        "SECCION_ID": "ESC_02_DIRECCION", "PREGUNTA_ID": "ESC_DIR_01",
        "TIPO_DATO": "LIKERT", "INDICE_100": 75, "VALOR_BASE": 4,
        "ENTRA_PROMEDIO_SECCION": "SI", "INCLUIR_EN_KPI": "SI",
        "CALCULA_KPI_PREGUNTA": "SI", "ENTRA_INDICE_GENERAL": "SI",
    }


def assignment(career, modality, name="Responsable confirmado"):
    return {"CARRERA_BASE": career, "MODALIDAD_ACADEMICA": modality,
            "PERSONA_EVALUADA": name, "ROL_EVALUADO": "Asistente"}


def result(catalog, records=None):
    with patch("backend.SERVICES.calidad.kpi_service.cargar_personas_evaluadas", return_value=catalog):
        return calcular_kpis_preview(records or [row(), {**row("Psicología"), "FILA_ORIGEN": 2}])


class QualityCatalogTests(unittest.TestCase):
    def test_active_populations_with_no_answers_are_in_direction(self):
        populations = [
            ("Maestría en Desarrollo del Potencial Humano y Organizacional (Coaching)", "Posgrado"),
            ("Maestría en Psicoterapia Psicoanalítica", "Posgrado"),
            *((career, "Ejecutiva") for career in ["Administración de Empresas", "Contaduría", "Mercadotecnia", "Tecnologías de la Información"]),
        ]
        data = result([assignment(*population) for population in populations])
        rows = data["comparativos_direccion"]["por_carrera_modalidad"]
        for career, modality in populations:
            matches = [r for r in rows if (r["carrera_base"], r["modalidad_academica"]) == (career, modality)]
            self.assertEqual(len(matches), 1)
            self.assertIsNone(matches[0]["indice"])
            self.assertEqual(matches[0]["encuestas"], 0)

    def test_assignments_include_unanswered_careers(self):
        catalog = [assignment(c, "Presencial", "Grecia Ceballos Ibarra") for c in
                   ["Administración de Empresas", "Contaduría", "Finanzas", "Mercadotecnia"]]
        rows = result(catalog)["comparativos_personal"]["asignaciones"]
        greek = [r for r in rows if r["persona_evaluada"] == "Grecia Ceballos Ibarra"]
        self.assertEqual(len(greek), 4)
        self.assertTrue(all(r["indice"] is None and r["evaluaciones"] == 0 for r in greek))

    def test_confirmed_responsibilities_are_exposed_without_fabricated_scores(self):
        rows = result([])["comparativos_personal"]["asignaciones"]
        for person, career in [("Oscar Corona Amezcua", "Desarrollo del Potencial Humano"),
                               ("Lourdes Quiroga", "Psicoterapia"), ("Militza Fermoso", "Odontología Legal"),
                               ("Karla Sánchez Cruz", "Psicología"), ("Balbina García Arroyo", "Actuación")]:
            matches = [r for r in rows if r["persona_evaluada"] == person]
            self.assertEqual(len(matches), 1)
            self.assertIn(career, matches[0]["carrera_base"])
            self.assertIsNone(matches[0]["indice"])
        for person in ["Alejandra Romero Peña", "Brenda Ponce Pérez"]:
            matches = [r for r in rows if r["persona_evaluada"] == person]
            self.assertEqual(len(matches), 6)
            self.assertTrue(all(r["modalidad_academica"] == "Ejecutiva" for r in matches))

    def test_claudia_only_retains_postgraduate_assignments(self):
        catalog = [assignment("Derecho", "Presencial", "Claudia Diaz"),
                   assignment("Maestría en Psicoterapia", "Posgrado", "Claudia Diaz")]
        rows = result(catalog)["comparativos_personal"]["asignaciones"]
        claudia = [r for r in rows if r["persona_evaluada"] == "Claudia Diaz"]
        self.assertEqual(len(claudia), 1)
        self.assertEqual(claudia[0]["modalidad_academica"], "Posgrado")

    def test_service_filter_does_not_add_other_catalog_populations(self):
        data = result([assignment("Derecho", "Presencial"), assignment("Contaduría", "Ejecutiva")], [row()])
        self.assertTrue(all(r["carrera_base"] == "Derecho" for r in data["comparativos_direccion"]["por_carrera_modalidad"]))
        self.assertTrue(all(r["carrera_base"] == "Derecho" for r in data["comparativos_personal"]["asignaciones"]))

    def test_adding_unanswered_population_does_not_change_existing_indices(self):
        before, after = result([]), result([assignment("Contaduría", "Ejecutiva")])
        for key in ["indice_general", "secciones", "preguntas"]:
            self.assertEqual(before[key], after[key])


if __name__ == "__main__":
    unittest.main()
