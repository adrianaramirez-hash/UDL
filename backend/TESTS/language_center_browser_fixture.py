"""Offline populations for language/course joins and contextual detail."""
import json
from backend.TESTS.test_language_center import mapping, population
from backend.SERVICES.calidad.language_center import resumen_actual


def fixtures():
    definitions = mapping()
    rows = (population(definitions, 1, "Derecho", "Presencial", "Inglés", "Intensivo", 100, 100)
            + population(definitions, 2, "Derecho", "Ejecutiva", "Francés", "Sabatino", 50, 100)
            + population(definitions, 3, "Nutrición", "Presencial", "Inglés", "Intensivo", 25, 0))
    return {f"{career}|{modality}|{language}|{course}": {"actual": resumen_actual(rows, definitions, career, modality, language, course)}
            for career in ("", "Derecho", "Nutrición") for modality in ("", "Presencial", "Ejecutiva")
            for language in ("", "Inglés", "Francés") for course in ("", "Intensivo", "Sabatino")}


if __name__ == "__main__":
    print(json.dumps(fixtures(), ensure_ascii=True))
