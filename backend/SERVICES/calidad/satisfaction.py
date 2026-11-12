"""Independent satisfaction index and binary recommendation, without aggregation."""
from backend.SERVICES.calidad.kpi_service import calcular_kpis_preview, _es_si, _numero
from backend.SERVICES.calidad.institutional_services import clave, _validos

INDICADORES = {"SAT_GENERAL": "Satisfacción general", "REC_UDL": "Recomendación"}


def es_seccion(row):
    return str(row.get("SECCION_ID", "")).replace("ESC25_", "ESC_", 1) == "ESC_09_SATISFACCION"


def catalogo(mapa, completo=True):
    result, questions = {}, set()
    for row in mapa:
        if not es_seccion(row) or not _es_si(row.get("CALCULA_KPI_PREGUNTA")):
            continue
        canonical = str(row.get("PREGUNTA_CANONICA_ID") or "").strip()
        question = str(row.get("PREGUNTA_ID") or "").strip()
        if canonical not in INDICADORES:
            if completo:
                raise ValueError("Indicador fuera de los dos autorizados en Satisfacción y recomendación")
            continue
        expected = "LIKERT" if canonical == "SAT_GENERAL" else "BINARIA"
        if not question or canonical in result or question in questions or row.get("TIPO_DATO") != expected:
            raise ValueError("Indicador duplicado, sin pregunta o con escala incompatible")
        result[canonical] = row
        questions.add(question)
    if completo and set(result) != set(INDICADORES):
        raise ValueError("Se requieren Satisfacción general y Recomendación como indicadores independientes")
    return result


def filtrar(registros, carrera="", modalidad_academica=""):
    return [row for row in registros if es_seccion(row)
            and (not carrera or clave(row.get("CARRERA_BASE")) == clave(carrera))
            and (not modalidad_academica or clave(row.get("MODALIDAD_ACADEMICA") or row.get("MODALIDAD_DETALLE")) == clave(modalidad_academica))]


def _rows(rows, definition):
    return [row for row in rows if definition and row.get("PREGUNTA_ID") == definition["PREGUNTA_ID"]]


def _binary_rows(rows):
    return [row for row in _validos(rows) if row.get("TIPO_DATO") == "BINARIA" and _numero(row.get("VALOR_BASE")) in (0, 1)]


def _surveys(rows):
    return len({row["FILA_ORIGEN"] for row in rows if row.get("FILA_ORIGEN") not in (None, "")})


def _engine(rows):
    # Source/year trigger unrelated personnel catalogs. Copies preserve source
    # records and delegate scoring and distributions to the unchanged engine.
    return calcular_kpis_preview([{key: value for key, value in row.items() if key not in {"ANIO", "FUENTE_ID"}} for row in rows])


def _satisfaction(rows, definition):
    question = next(iter(_engine(rows)["preguntas"]), {})
    return {"id": "SAT_GENERAL", "nombre": INDICADORES["SAT_GENERAL"],
            "pregunta_id": definition["PREGUNTA_ID"], "escala_id": definition.get("ESCALA_ID"),
            "indice": question.get("indice"), "n": question.get("n", 0),
            "encuestas_validas": _surveys(_validos(rows)),
            **{field: question.get(field) for field in ("favorable_pct", "neutral_pct", "desfavorable_pct")}}


def _recommendation(rows, definition):
    valid = _binary_rows(rows)
    yes = sum(_numero(row.get("VALOR_BASE")) == 1 for row in valid)
    no = len(valid) - yes
    return {"id": "REC_UDL", "nombre": INDICADORES["REC_UDL"],
            "pregunta_id": definition["PREGUNTA_ID"] if definition else None,
            "escala_id": definition.get("ESCALA_ID") if definition else None,
            "si": yes, "no": no, "total_valido": len(valid), "encuestas_validas": _surveys(valid),
            "si_pct": round(yes * 100 / len(valid), 2) if valid else None,
            "no_pct": round(no * 100 / len(valid), 2) if valid else None}


def resumen_actual(registros, mapa, carrera="", modalidad_academica=""):
    definitions = catalogo(mapa)
    selected, all_rows = filtrar(registros, carrera, modalidad_academica), filtrar(registros)
    satisfaction_rows = _rows(selected, definitions["SAT_GENERAL"])
    recommendation_rows = _rows(selected, definitions["REC_UDL"])
    reason_ids = {row["PREGUNTA_ID"] for row in mapa if es_seccion(row)
                  and row.get("PREGUNTA_CANONICA_ID") == "REC_PORQUE" and row.get("TIPO_DATO") == "TEXTO"}
    comments = _engine([row for row in selected if row.get("PREGUNTA_ID") in reason_ids])["comentarios"]
    return {"satisfaccion": _satisfaction(satisfaction_rows, definitions["SAT_GENERAL"]),
            "recomendacion": _recommendation(recommendation_rows, definitions["REC_UDL"]),
            "encuestas_analizadas": _surveys(_validos(satisfaction_rows) + _binary_rows(recommendation_rows)),
            "comentarios": comments,
            "opciones": {"carreras": sorted({str(row.get("CARRERA_BASE") or "").strip() for row in all_rows} - {""}),
                         "modalidades": sorted({str(row.get("MODALIDAD_ACADEMICA") or row.get("MODALIDAD_DETALLE") or "").strip() for row in all_rows} - {""})},
            "poblacion": {"carrera": carrera or None, "modalidad_academica": modalidad_academica or None}}


def resumen_historico(actuales, mapa_actual, anteriores, mapa_anterior, carrera="", modalidad_academica=""):
    now_map, old_map = catalogo(mapa_actual), catalogo(mapa_anterior, completo=False)
    now, old = filtrar(actuales, carrera, modalidad_academica), filtrar(anteriores, carrera, modalidad_academica)
    result = {"poblacion": {"carrera": carrera or None, "modalidad_academica": modalidad_academica or None}}
    for canonical, field in [("SAT_GENERAL", "satisfaccion"), ("REC_UDL", "recomendacion")]:
        approved = canonical in old_map and all(_es_si(definition.get("COMPARABLE_HISTORICO")) for definition in (now_map[canonical], old_map[canonical]))
        current = [r for r in _rows(now, now_map[canonical]) if _es_si(r.get("COMPARABLE_HISTORICO"))] if approved else []
        previous = [r for r in _rows(old, old_map[canonical]) if _es_si(r.get("COMPARABLE_HISTORICO"))] if approved else []
        if canonical == "SAT_GENERAL":
            new_value = _satisfaction(current, now_map[canonical])["indice"]
            old_value = _satisfaction(previous, old_map[canonical])["indice"] if approved else None
            old_key, new_key, delta_key = "indice_2025", "indice_2026", "delta"
        else:
            new_value = _recommendation(current, now_map[canonical])["si_pct"]
            old_value = _recommendation(previous, old_map.get(canonical))["si_pct"]
            old_key, new_key, delta_key = "porcentaje_2025", "porcentaje_2026", "delta_pp"
        comparable = old_value is not None and new_value is not None
        result[field] = {"id": canonical, "comparable": comparable,
                         old_key: old_value if comparable else None, new_key: new_value if comparable else None,
                         delta_key: round(new_value - old_value, 2) if comparable else None}
    return result
