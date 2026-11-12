"""Ambiente escolar: stable quantitative indices and independent incidence rates."""
from backend.SERVICES.calidad.kpi_service import calcular_kpis_preview, _es_si, _numero
from backend.SERVICES.calidad.institutional_services import clave, _validos, _distribucion


# Display names requested by the institution; canonical IDs come from the approved map.
CRITERIOS = {
    "AMB_REL_PROF": "Relación con profesores / asesores",
    "AMB_REL_COMP": "Relación con compañeros",
    "AMB_AMBIENTE": "Ambiente escolar en mi servicio",
}


def es_ambiente(row):
    return str(row.get("SECCION_ID", "")).replace("ESC25_", "ESC_", 1) == "ESC_08_AMBIENTE"


def catalogo(mapa, completo=True):
    result = {}
    question_ids = set()
    for row in mapa:
        if not es_ambiente(row) or not _es_si(row.get("ENTRA_PROMEDIO_SECCION")):
            continue
        if not _es_si(row.get("CALCULA_KPI_PREGUNTA")):
            continue
        canonical = str(row.get("PREGUNTA_CANONICA_ID") or "").strip()
        question = str(row.get("PREGUNTA_ID") or "").strip()
        if canonical not in CRITERIOS:
            if completo:
                raise ValueError("El mapa contiene un criterio de ambiente escolar fuera de los 3 autorizados")
            continue
        if not question or canonical in result or question in question_ids:
            raise ValueError("Criterio de ambiente escolar sin pregunta o duplicado en el mapa aprobado")
        result[canonical] = row
        question_ids.add(question)
    if completo and set(result) != set(CRITERIOS):
        raise ValueError("El mapa debe contener exactamente los 3 criterios de ambiente escolar autorizados")
    return result


def filtrar(registros, carrera="", modalidad_academica=""):
    return [row for row in registros if es_ambiente(row)
            and (not carrera or clave(row.get("CARRERA_BASE")) == clave(carrera))
            and (not modalidad_academica or clave(row.get("MODALIDAD_ACADEMICA") or row.get("MODALIDAD_DETALLE")) == clave(modalidad_academica))]


def _encuestas(rows):
    return len({row["FILA_ORIGEN"] for row in _validos(rows) if row.get("FILA_ORIGEN") not in (None, "")})


def _metricas(rows):
    # Source/year only trigger unrelated personnel catalogs. Project copies and
    # delegate all scores, valid-response denominators and weights to the engine.
    projected = [{key: value for key, value in row.items() if key not in {"FUENTE_ID", "ANIO"}} for row in rows]
    result = calcular_kpis_preview(projected)
    return {
        "indice": next((row["indice"] for row in result["secciones"] if str(row["seccion_id"]).replace("ESC25_", "ESC_", 1) == "ESC_08_AMBIENTE"), None),
        "respuestas": sum(row["n"] for row in result["preguntas"] if row["entra_seccion"]),
        "preguntas": {row["pregunta_id"]: row for row in result["preguntas"]},
        "comentarios": result["comentarios"], "distribucion": _distribucion(rows),
    }


def _incidence_definition(mapa, required=False):
    definitions = [row for row in mapa if es_ambiente(row) and row.get("PREGUNTA_CANONICA_ID") == "AMB_PROBLEMA"]
    if len(definitions) > 1 or (required and not definitions):
        raise ValueError("El mapa debe contener una única pregunta de incidencia")
    if not definitions:
        return None
    definition = definitions[0]
    if (not definition.get("PREGUNTA_ID") or definition.get("TIPO_DATO") != "BINARIA"
            or _es_si(definition.get("ENTRA_PROMEDIO_SECCION"))):
        raise ValueError("La incidencia debe ser binaria y estar excluida del promedio")
    if definition["PREGUNTA_ID"] in {row["PREGUNTA_ID"] for row in catalogo(mapa, completo=False).values()}:
        raise ValueError("La incidencia no puede compartir pregunta con un criterio cuantitativo")
    return definition


def _incidence_rows(rows, definition):
    if definition is None:
        return []
    # Read only normalized values approved by the existing scale. No new text
    # normalization or transformation of this rate into an index is introduced.
    return [row for row in _validos(rows) if row.get("PREGUNTA_ID") == definition["PREGUNTA_ID"]
            and row.get("TIPO_DATO") == "BINARIA" and _numero(row.get("VALOR_BASE")) in (0, 1)]


def _incidence(rows):
    yes = sum(_numero(row.get("VALOR_BASE")) == 1 for row in rows)
    no = len(rows) - yes
    return {"si": yes, "no": no, "total_valido": len(rows),
            "si_pct": round(yes * 100 / len(rows), 2) if rows else None,
            "no_pct": round(no * 100 / len(rows), 2) if rows else None}


def resumen_actual(registros, mapa, carrera="", modalidad_academica=""):
    definitions = catalogo(mapa)
    incidence_definition = _incidence_definition(mapa, required=True)
    selected = filtrar(registros, carrera, modalidad_academica)
    allowed = {row["PREGUNTA_ID"] for row in definitions.values()}
    criterion_rows = [row for row in selected if row.get("PREGUNTA_ID") in allowed]
    text_ids = {row["PREGUNTA_ID"] for row in mapa if es_ambiente(row)
                and row.get("PREGUNTA_CANONICA_ID") in {"AMB_DESC", "AMB_COMENTARIO"} and row.get("TIPO_DATO") == "TEXTO"}
    # Incidence is excluded before calling the stable index engine, regardless
    # of the flags carried by individual response records.
    metrics = _metricas([row for row in selected if row.get("PREGUNTA_ID") in allowed | text_ids])
    incidence_rows = _incidence_rows(selected, incidence_definition)
    criteria = []
    for canonical, label in CRITERIOS.items():
        definition = definitions[canonical]
        question = metrics["preguntas"].get(definition["PREGUNTA_ID"], {})
        criteria.append({"id": canonical, "nombre": label, "pregunta_id": definition["PREGUNTA_ID"],
                         "indice": question.get("indice"), "n": question.get("n", 0),
                         "encuestas_validas": _encuestas([row for row in criterion_rows if row.get("PREGUNTA_ID") == definition["PREGUNTA_ID"]]),
                         **{field: question.get(field) for field in ("favorable_pct", "neutral_pct", "desfavorable_pct")}})
    all_rows = filtrar(registros)
    return {"indice": metrics["indice"], "respuestas": metrics["respuestas"],
            "encuestas_analizadas": _encuestas(criterion_rows + incidence_rows), "criterios": criteria,
            "incidencia": _incidence(incidence_rows),
            "distribucion": metrics["distribucion"], "comentarios": metrics["comentarios"],
            "opciones": {"carreras": sorted({str(row.get("CARRERA_BASE") or "").strip() for row in all_rows} - {""}),
                         "modalidades": sorted({str(row.get("MODALIDAD_ACADEMICA") or row.get("MODALIDAD_DETALLE") or "").strip() for row in all_rows} - {""})},
            "poblacion": {"carrera": carrera or None, "modalidad_academica": modalidad_academica or None}}


def resumen_historico(actuales, mapa_actual, anteriores, mapa_anterior, carrera="", modalidad_academica=""):
    current_map, previous_map = catalogo(mapa_actual), catalogo(mapa_anterior, completo=False)
    approved = {key for key in current_map.keys() & previous_map.keys()
                if _es_si(current_map[key].get("COMPARABLE_HISTORICO")) and _es_si(previous_map[key].get("COMPARABLE_HISTORICO"))}
    current_ids = {current_map[key]["PREGUNTA_ID"]: key for key in approved}
    previous_ids = {previous_map[key]["PREGUNTA_ID"]: key for key in approved}
    current = [row for row in filtrar(actuales, carrera, modalidad_academica) if row.get("PREGUNTA_ID") in current_ids and _es_si(row.get("COMPARABLE_HISTORICO"))]
    previous = [row for row in filtrar(anteriores, carrera, modalidad_academica) if row.get("PREGUNTA_ID") in previous_ids and _es_si(row.get("COMPARABLE_HISTORICO"))]
    current_values = {current_ids[key]: row["indice"] for key, row in _metricas(current)["preguntas"].items()}
    previous_values = {previous_ids[key]: row["indice"] for key, row in _metricas(previous)["preguntas"].items()}
    intersection = {key for key in approved if current_values.get(key) is not None and previous_values.get(key) is not None}
    same_current = _metricas([row for row in current if current_ids[row["PREGUNTA_ID"]] in intersection])
    same_previous = _metricas([row for row in previous if previous_ids[row["PREGUNTA_ID"]] in intersection])
    now_definition = _incidence_definition(mapa_actual, required=True)
    old_definition = _incidence_definition(mapa_anterior)
    incidence_approved = old_definition is not None and all(_es_si(row.get("COMPARABLE_HISTORICO")) for row in (now_definition, old_definition))
    now_incidence = _incidence([row for row in _incidence_rows(filtrar(actuales, carrera, modalidad_academica), now_definition)
                                if _es_si(row.get("COMPARABLE_HISTORICO"))]) if incidence_approved else _incidence([])
    old_incidence = _incidence([row for row in _incidence_rows(filtrar(anteriores, carrera, modalidad_academica), old_definition)
                                if _es_si(row.get("COMPARABLE_HISTORICO"))]) if incidence_approved else _incidence([])
    comparable = bool(now_incidence["total_valido"] and old_incidence["total_valido"])
    return {"indice_2025": same_previous["indice"], "indice_2026": same_current["indice"],
            "incidencia": {"comparable": comparable,
                           "porcentaje_2025": old_incidence["si_pct"] if comparable else None,
                           "porcentaje_2026": now_incidence["si_pct"] if comparable else None,
                           "delta_pp": round(now_incidence["si_pct"] - old_incidence["si_pct"], 2) if comparable else None},
            "ids_comparables": sorted(intersection),
            "criterios": [{"id": key, "indice_2025": previous_values.get(key) if key in intersection else None,
                           "indice_2026": current_values.get(key)} for key in CRITERIOS],
            "poblacion": {"carrera": carrera or None, "modalidad_academica": modalidad_academica or None}}
