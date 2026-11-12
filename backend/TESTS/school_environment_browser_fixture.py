"""Offline populations calculated by the stable engine and environment projection."""
import json
from backend.TESTS.test_school_environment import mapping, response
from backend.SERVICES.calidad.school_environment import resumen_actual, resumen_historico


def fixtures():
    now_map, old_map = mapping(), mapping(2025)
    now, old = [], []
    populations = [("Derecho", "Presencial", 100, 75, 100, 0),
                   ("Derecho", "Ejecutiva", 25, 50, 0, 100),
                   ("Nutrición", "Presencial", 50, 25, 0, 0)]
    for number, (career, modality, current, previous, incidence, previous_incidence) in enumerate(populations, 1):
        for definitions, records, score, rate in [(now_map, now, current, incidence), (old_map, old, previous, previous_incidence)]:
            for index, definition in enumerate(definitions[:3]):
                records.append(response(definition, None if modality == "Ejecutiva" and index == 2 else score, number, career, modality))
            records.append(response(definitions[3], rate, number, career, modality))
        for definition in now_map[4:]:
            now.append(response(definition, f"Comentario de {career} / {modality}", number, career, modality))
    return {f"{career}|{modality}": {
        "actual": resumen_actual(now, now_map, career, modality),
        "historico": resumen_historico(now, now_map, old, old_map, career, modality)}
        for career in ("", "Derecho", "Nutrición") for modality in ("", "Presencial", "Ejecutiva")}


if __name__ == "__main__":
    print(json.dumps(fixtures(), ensure_ascii=True))
