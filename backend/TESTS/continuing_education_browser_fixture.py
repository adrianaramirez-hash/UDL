"""Offline populations computed by the section projection and stable engine."""
import json
from backend.TESTS.test_continuing_education import mapping, population
from backend.SERVICES.calidad.continuing_education import resumen_actual


def fixtures():
    definitions = mapping()
    rows = (population(definitions, 1, "Derecho", "Presencial", score=100)
            + population(definitions, 2, "Derecho", "Ejecutiva", score=50, recommendation=0, knowledge=0, where=100, media="Redes sociales")
            + population(definitions, 3, "Nutrición", "Presencial", usage=0, knowledge=0, where=0))
    return {f"{career}|{modality}|{participation}": {"actual": resumen_actual(rows, definitions, career, modality, participation)}
            for career in ("", "Derecho", "Nutrición") for modality in ("", "Presencial", "Ejecutiva") for participation in ("", "si", "no")}


if __name__ == "__main__":
    print(json.dumps(fixtures(), ensure_ascii=True))
