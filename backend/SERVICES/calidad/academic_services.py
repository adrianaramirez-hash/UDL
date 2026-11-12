"""Servicios académicos: projection of the unchanged stable KPI engine."""
from backend.SERVICES.calidad.kpi_service import calcular_kpis_preview, _es_si
from backend.SERVICES.calidad.institutional_services import clave, _validos, _distribucion


# Display names requested by the institution; canonical IDs come from the approved map.
CRITERIOS = {
    "ACA_PROFESORES": "Profesores / Asesores",
    "ACA_PLAN_ESTUDIOS": "Plan de estudios",
    "ACA_CLASES_PRESENCIALES": "Clases presenciales",
    "ACA_CLASES_ZOOM": "Clases en Zoom",
    "ACA_PRACTICAS": "Prácticas académicas",
    "ACA_CERTIFICACIONES": "Certificaciones",
    "ACA_SEAC_GENERAL": "Plataforma SEAC en general",
    "ACA_SEAC_APPS": "SEAC · Apps móviles",
    "ACA_SEAC_INGRESO": "SEAC · Facilidad para ingresar",
    "ACA_SEAC_TRAMITES": "SEAC · Trámites administrativos",
    "ACA_SEAC_TAREAS": "SEAC · Tareas y trabajos",
    "ACA_AULAS_VIRTUALES": "Contenidos de las Aulas Virtuales",
}


def es_academico(row):
    return str(row.get("SECCION_ID", "")).replace("ESC25_", "ESC_", 1) == "ESC_04_ACADEMICOS"


def catalogo(mapa, completo=True):
    result = {}
    question_ids = set()
    for row in mapa:
        if not es_academico(row) or not _es_si(row.get("ENTRA_PROMEDIO_SECCION")):
            continue
        if not _es_si(row.get("CALCULA_KPI_PREGUNTA")):
            continue
        canonical = str(row.get("PREGUNTA_CANONICA_ID") or "").strip()
        question = str(row.get("PREGUNTA_ID") or "").strip()
        if canonical not in CRITERIOS:
            if completo:
                raise ValueError("El mapa contiene un criterio académico fuera de los 12 autorizados")
            continue
        if not question or canonical in result or question in question_ids:
            raise ValueError("Criterio académico sin pregunta o duplicado en el mapa aprobado")
        result[canonical] = row
        question_ids.add(question)
    if completo and set(result) != set(CRITERIOS):
        raise ValueError("El mapa debe contener exactamente los 12 criterios académicos autorizados")
    return result


def filtrar(registros, carrera="", modalidad_academica=""):
    return [row for row in registros if es_academico(row)
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
        "indice": next((row["indice"] for row in result["secciones"] if str(row["seccion_id"]).replace("ESC25_", "ESC_", 1) == "ESC_04_ACADEMICOS"), None),
        "respuestas": sum(row["n"] for row in result["preguntas"] if row["entra_seccion"]),
        "preguntas": {row["pregunta_id"]: row for row in result["preguntas"]},
        "comentarios": result["comentarios"], "distribucion": _distribucion(rows),
    }


def resumen_actual(registros, mapa, carrera="", modalidad_academica=""):
    definitions = catalogo(mapa)
    selected = filtrar(registros, carrera, modalidad_academica)
    allowed = {row["PREGUNTA_ID"] for row in definitions.values()}
    criterion_rows = [row for row in selected if row.get("PREGUNTA_ID") in allowed]
    metrics = _metricas([row for row in selected if row.get("PREGUNTA_ID") in allowed or row.get("TIPO_DATO") == "TEXTO"])
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
            "encuestas_analizadas": _encuestas(criterion_rows), "criterios": criteria,
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
    return {"indice_2025": same_previous["indice"], "indice_2026": same_current["indice"],
            "ids_comparables": sorted(intersection),
            "criterios": [{"id": key, "indice_2025": previous_values.get(key) if key in intersection else None,
                           "indice_2026": current_values.get(key)} for key in CRITERIOS],
            "poblacion": {"carrera": carrera or None, "modalidad_academica": modalidad_academica or None}}
