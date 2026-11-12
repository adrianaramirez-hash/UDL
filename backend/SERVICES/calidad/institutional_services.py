"""Read-only projection of the stable engine for Servicios institucionales."""
from collections import Counter
import re
from backend.SERVICES.calidad.kpi_service import calcular_kpis_preview, _es_si, _numero


def es_servicios(row):
    return str(row.get("SECCION_ID", "")).replace("ESC25_", "ESC_", 1) == "ESC_03_SERVICIOS"


def clave(valor):
    return " ".join(str(valor or "").split()).casefold()


def nombre_desde_encabezado(definition, encabezado):
    """Use the existing RAW matrix label, without changing mapped identifiers."""
    if definition.get("PREGUNTA_CANONICA_ID") != "SER_PARAMEDICO":
        return definition
    if not re.fullmatch(r"Servicio institucional\s+\d+", str(definition.get("TEXTO_CORTO", "")).strip(), re.IGNORECASE):
        return definition
    matches = re.findall(r"\[([^\[\]]+)\]", str(encabezado or ""))
    if not matches or not matches[-1].strip():
        return definition
    return {**definition, "NOMBRE_INSTITUCIONAL": matches[-1].strip()}


def catalogo(mapa):
    """Only mapped current KPI services; historical data never expands the list."""
    result = {}
    ids = set()
    names = set()
    for row in mapa:
        if not es_servicios(row) or not _es_si(row.get("ENTRA_PROMEDIO_SECCION")):
            continue
        if not _es_si(row.get("CALCULA_KPI_PREGUNTA")):
            continue
        canonical = str(row.get("PREGUNTA_CANONICA_ID") or "").strip()
        question = str(row.get("PREGUNTA_ID") or "").strip()
        name = str(row.get("TEXTO_CORTO") or "").strip()
        if not canonical or not question or not name:
            raise ValueError("Servicio sin ID canónico, pregunta o nombre en el mapa aprobado")
        if canonical in result or question in ids or clave(name) in names:
            raise ValueError("Servicios duplicados en el mapa aprobado")
        result[canonical] = row
        ids.add(question)
        names.add(clave(name))
    return result


def filtrar(registros, carrera="", modalidad_academica=""):
    return [row for row in registros if es_servicios(row)
            and (not carrera or clave(row.get("CARRERA_BASE")) == clave(carrera))
            and (not modalidad_academica or clave(row.get("MODALIDAD_ACADEMICA") or row.get("MODALIDAD_DETALLE")) == clave(modalidad_academica))]


def opciones(registros):
    rows = filtrar(registros)
    return {
        "carreras": sorted({str(row.get("CARRERA_BASE") or "").strip() for row in rows} - {""}),
        "modalidades": sorted({str(row.get("MODALIDAD_ACADEMICA") or row.get("MODALIDAD_DETALLE") or "").strip() for row in rows} - {""}),
    }


def _validos(registros):
    return [row for row in registros if _es_si(row.get("INCLUIR_EN_KPI")) and _numero(row.get("INDICE_100")) is not None]


def _distribucion(registros):
    """Aggregate response counts with the same categories as the stable engine.

    This is a response distribution, separate from the equal-weight question index.
    """
    counts = Counter()
    rows = _validos(registros)
    for row in rows:
        index = _numero(row.get("INDICE_100"))
        value = _numero(row.get("VALOR_BASE"))
        if str(row.get("TIPO_DATO", "")).upper() == "BINARIA":
            category = "favorable" if index == 100 else "desfavorable" if index == 0 else None
        else:
            category = "favorable" if value is not None and value >= 4 else "neutral" if value == 3 else "desfavorable" if value is not None and value <= 2 else None
        if category:
            counts[category] += 1
    return {f"{name}_pct": round(counts[name] * 100 / len(rows), 2) if rows else None
            for name in ("favorable", "neutral", "desfavorable")}


def _metricas(registros):
    # Delegate all question indices, counts, distributions and section weights.
    # These fields only trigger the unrelated personnel catalog in the engine.
    # Project copies for this section; normalized source records remain untouched.
    result = calcular_kpis_preview([{key: value for key, value in row.items() if key not in {"FUENTE_ID", "ANIO"}}
                                   for row in registros])
    return {
        "indice": next((row["indice"] for row in result["secciones"] if str(row["seccion_id"]).replace("ESC25_", "ESC_", 1) == "ESC_03_SERVICIOS"), None),
        "respuestas": sum(row["n"] for row in result["preguntas"] if row["entra_seccion"]),
        "distribucion": _distribucion(registros),
        "preguntas": {row["pregunta_id"]: row for row in result["preguntas"]},
        "comentarios": result["comentarios"],
    }


def resumen_actual(registros, mapa, carrera="", modalidad_academica=""):
    services = catalogo(mapa)
    selected = filtrar(registros, carrera, modalidad_academica)
    allowed = {row["PREGUNTA_ID"] for row in services.values()}
    metrics = _metricas([row for row in selected if row.get("PREGUNTA_ID") in allowed or row.get("TIPO_DATO") == "TEXTO"])
    output = []
    for canonical, definition in services.items():
        question = metrics["preguntas"].get(definition["PREGUNTA_ID"], {})
        output.append({
            "id": canonical, "nombre": (definition.get("NOMBRE_INSTITUCIONAL") or definition["TEXTO_CORTO"]).strip(),
            "pregunta_id": definition["PREGUNTA_ID"],
            "indice": question.get("indice"), "n": question.get("n", 0),
            "favorable_pct": question.get("favorable_pct"),
            "neutral_pct": question.get("neutral_pct"),
            "desfavorable_pct": question.get("desfavorable_pct"),
        })
    # Preserve existing comment selection/classification, without exposing its summaries.
    return {"indice": metrics["indice"], "respuestas": metrics["respuestas"],
            "encuestas_con_evaluacion": len({row["FILA_ORIGEN"] for row in _validos(selected)
                                             if row.get("PREGUNTA_ID") in allowed and row.get("FILA_ORIGEN") not in (None, "")}),
            "distribucion": metrics["distribucion"], "servicios": output,
            "comentarios": metrics["comentarios"], "opciones": opciones(registros),
            "poblacion": {"carrera": carrera or None, "modalidad_academica": modalidad_academica or None}}


def resumen_historico(actuales, mapa_actual, anteriores, mapa_anterior, carrera="", modalidad_academica=""):
    current_map, previous_map = catalogo(mapa_actual), catalogo(mapa_anterior)
    approved = {key for key in current_map.keys() & previous_map.keys()
                if _es_si(current_map[key].get("COMPARABLE_HISTORICO"))
                and _es_si(previous_map[key].get("COMPARABLE_HISTORICO"))}
    current = filtrar(actuales, carrera, modalidad_academica)
    previous = filtrar(anteriores, carrera, modalidad_academica)
    ids_current = {current_map[key]["PREGUNTA_ID"]: key for key in approved}
    ids_previous = {previous_map[key]["PREGUNTA_ID"]: key for key in approved}
    # Honor both metadata approval and the flags carried by the normalized records.
    current = [row for row in current if row.get("PREGUNTA_ID") in ids_current and _es_si(row.get("COMPARABLE_HISTORICO"))]
    previous = [row for row in previous if row.get("PREGUNTA_ID") in ids_previous and _es_si(row.get("COMPARABLE_HISTORICO"))]
    current_metrics, previous_metrics = _metricas(current), _metricas(previous)
    current_values = {ids_current[key]: row for key, row in current_metrics["preguntas"].items()}
    previous_values = {ids_previous[key]: row for key, row in previous_metrics["preguntas"].items()}
    intersection = {key for key in approved if current_values.get(key, {}).get("indice") is not None
                    and previous_values.get(key, {}).get("indice") is not None}
    # Both global indices use exactly the same criteria with observations in both years.
    comparable_current = _metricas([row for row in current if ids_current[row["PREGUNTA_ID"]] in intersection])
    comparable_previous = _metricas([row for row in previous if ids_previous[row["PREGUNTA_ID"]] in intersection])
    return {
        "indice_2025": comparable_previous["indice"], "indice_2026": comparable_current["indice"],
        "criterios": sorted(intersection),
        "servicios": [{"id": key, "indice_2025": previous_values.get(key, {}).get("indice") if key in intersection else None,
                       "indice_2026": current_values.get(key, {}).get("indice")}
                      for key in current_map],
        "poblacion": {"carrera": carrera or None, "modalidad_academica": modalidad_academica or None},
    }
