"""Offline browser fixtures calculated by the independent indicator projection."""
import json
from backend.TESTS.test_satisfaction import mapping, response
from backend.SERVICES.calidad.satisfaction import resumen_actual, resumen_historico


def fixtures():
    now_map, old_map = mapping(), mapping(2025)
    now, old = [], []
    populations = [("Derecho", "Presencial", 100, 75, 100, 0),
                   ("Derecho", "Ejecutiva", 50, 25, 0, 100),
                   ("Nutrición", "Presencial", 25, 50, 0, 100)]
    for number, (career, modality, current, previous, recommendation, previous_recommendation) in enumerate(populations, 1):
        for definitions, records, score, rate in [(now_map, now, current, recommendation), (old_map, old, previous, previous_recommendation)]:
            records += [response(definitions[0], score, number, career, modality), response(definitions[1], rate, number, career, modality)]
        now.append(response(now_map[2], f"Motivo de {career} / {modality}", number, career, modality))
    return {f"{career}|{modality}": {
        "actual": resumen_actual(now, now_map, career, modality),
        "historico": resumen_historico(now, now_map, old, old_map, career, modality)}
        for career in ("", "Derecho", "Nutrición") for modality in ("", "Presencial", "Ejecutiva")}


if __name__ == "__main__":
    print(json.dumps(fixtures(), ensure_ascii=True))
