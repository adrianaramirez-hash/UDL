"""Offline browser fixtures computed by the facilities projection and stable engine."""
import json
from backend.TESTS.test_facilities import mapping, response
from backend.SERVICES.calidad.facilities import resumen_actual, resumen_historico


def fixtures():
    current_map, previous_map = mapping(), mapping(2025)
    current, previous = [], []
    populations = [("Derecho", "Presencial", 100, 75),
                   ("Derecho", "Ejecutiva", 25, 50),
                   ("Nutrición", "Presencial", 50, 25)]
    for number, (career, modality, now, before) in enumerate(populations, 1):
        for definitions, records, value in [(current_map, current, now), (previous_map, previous, before)]:
            for definition in definitions:
                applicable = not (modality == "Ejecutiva" and definition["PREGUNTA_CANONICA_ID"] in
                                  {"INS_LAB_COMPUTO", "INS_SALONES", "INS_HYFLEX"})
                score = value
                if definition["PREGUNTA_CANONICA_ID"] == "INS_CAFETERIA":
                    score = max(0, value - 25)
                records.append(response(definition, score if applicable else None, number, career, modality))
        current.append({"SECCION_ID": "ESC_07_INSTALACIONES", "TIPO_DATO": "TEXTO",
                        "PREGUNTA_ID": "INS_COMENTARIO", "TEXTO_CORTO": "Aspectos por mejorar",
                        "RESPUESTA_ORIGINAL": f"Comentario de {career} / {modality}",
                        "FILA_ORIGEN": number, "CARRERA_BASE": career, "MODALIDAD_ACADEMICA": modality})
    output = {}
    for career in ("", "Derecho", "Nutrición"):
        for modality in ("", "Presencial", "Ejecutiva"):
            output[f"{career}|{modality}"] = {
                "actual": resumen_actual(current, current_map, career, modality),
                "historico": resumen_historico(current, current_map, previous, previous_map, career, modality),
            }
    return output


if __name__ == "__main__":
    print(json.dumps(fixtures(), ensure_ascii=True))
