"""Independent specialized indicators and descriptive context, without historical joins."""
from collections import Counter
from backend.SERVICES.calidad.kpi_service import calcular_kpis_preview, _es_si, _numero
from backend.SERVICES.calidad.institutional_services import clave, _validos

CRITERIOS = {
    "EC_CALIDAD": "Calidad de cursos y diplomados",
    "EC_TEMAS": "Relevancia de los temas para la formación profesional",
    "EC_HORARIOS": "Horarios y modalidades",
    "EC_RECOMIENDA": "Recomendación de programas de Educación Continua",
    "EC_CONOCE": "Conocimiento de cursos, talleres y diplomados",
    "EC_SABE_DONDE": "Conocimiento de dónde consultar la oferta académica",
}
BINARIOS = {"EC_RECOMIENDA", "EC_CONOCE", "EC_SABE_DONDE"}
CONTEXTOS = {"EC_USO", "EC_MEDIO", "EC_MEJORA"}


def es_seccion(row):
    return str(row.get("SECCION_ID", "")).replace("ESC25_", "ESC_", 1) == "ESC_06_EDUC_CONT"


def catalogo(mapa):
    result, ids = {}, set()
    for row in mapa:
        if not es_seccion(row):
            continue
        canonical = str(row.get("PREGUNTA_CANONICA_ID") or "").strip()
        if canonical not in {*CRITERIOS, *CONTEXTOS}:
            if _es_si(row.get("CALCULA_KPI_PREGUNTA")):
                raise ValueError("Indicador fuera de los seis especializados de Educación Continua")
            continue
        question = str(row.get("PREGUNTA_ID") or "").strip()
        expected = "BINARIA" if canonical in BINARIOS | {"EC_USO"} else "CATEGORICA" if canonical == "EC_MEDIO" else "TEXTO" if canonical == "EC_MEJORA" else "LIKERT"
        if not question or canonical in result or question in ids or row.get("TIPO_DATO") != expected:
            raise ValueError("Pregunta duplicada, sin ID o con escala incompatible en Educación Continua")
        if canonical in CRITERIOS and not _es_si(row.get("CALCULA_KPI_PREGUNTA")):
            raise ValueError("Los seis indicadores requieren sus respuestas cuantitativas aprobadas")
        if canonical in CONTEXTOS and _es_si(row.get("ENTRA_PROMEDIO_SECCION")):
            raise ValueError("Participación, medios y comentarios no entran a satisfacción")
        result[canonical] = row
        ids.add(question)
    if not set(CRITERIOS) <= result.keys():
        raise ValueError("La evaluación especializada requiere los seis indicadores de Educación Continua 2026")
    return result


def _form(row):
    return row.get("FUENTE_ID"), row.get("ANIO"), row.get("FILA_ORIGEN")


def _surveys(rows):
    return len({_form(row) for row in rows if row.get("FILA_ORIGEN") not in (None, "")})


def _rows(rows, definitions, canonical):
    definition = definitions.get(canonical)
    return [r for r in rows if definition and r.get("PREGUNTA_ID") == definition["PREGUNTA_ID"]]


def _value(row):
    return str(row.get("RESPUESTA_NORMALIZADA") or row.get("RESPUESTA_ORIGINAL") or "").strip()


def _descriptive(rows):
    return [r for r in rows if _es_si(r.get("RESPUESTA_VALIDA")) and _value(r)]


def _binary_rows(rows):
    return [r for r in _validos(rows) if r.get("TIPO_DATO") == "BINARIA" and _numero(r.get("VALOR_BASE")) in (0, 1)]


def _binary(rows):
    valid = _binary_rows(rows)
    yes = sum(_numero(r.get("VALOR_BASE")) == 1 for r in valid)
    return {"si": yes, "no": len(valid) - yes, "total_valido": len(valid),
            "si_pct": round(yes * 100 / len(valid), 2) if valid else None,
            "no_pct": round((len(valid) - yes) * 100 / len(valid), 2) if valid else None}


def resumen_actual(registros, mapa, carrera="", modalidad_academica="", participacion=""):
    if participacion not in ("", "si", "no"):
        raise ValueError("Participación debe ser si, no o vista global")
    definitions = catalogo(mapa)
    ids = {r["PREGUNTA_ID"] for r in definitions.values()}
    all_rows = [r for r in registros if es_seccion(r) and r.get("PREGUNTA_ID") in ids]
    selected = [r for r in all_rows if (not carrera or clave(r.get("CARRERA_BASE")) == clave(carrera))
                and (not modalidad_academica or clave(r.get("MODALIDAD_ACADEMICA") or r.get("MODALIDAD_DETALLE")) == clave(modalidad_academica))]
    if participacion:
        target = 1 if participacion == "si" else 0
        forms = {_form(r) for r in _binary_rows(_rows(selected, definitions, "EC_USO")) if _numero(r.get("VALOR_BASE")) == target}
        selected = [r for r in selected if _form(r) in forms]
    likert_rows = [r for key in CRITERIOS.keys() - BINARIOS for r in _rows(selected, definitions, key)]
    comment_rows = _descriptive(_rows(selected, definitions, "EC_MEJORA"))
    projected = [{key: value for key, value in r.items() if key not in {"FUENTE_ID", "ANIO"}} for r in likert_rows + comment_rows]
    engine = calcular_kpis_preview(projected)
    questions = {r["pregunta_id"]: r for r in engine["preguntas"]}
    criteria, valid_rows = [], _validos(likert_rows)
    for canonical, name in CRITERIOS.items():
        definition = definitions[canonical]
        rows = _rows(selected, definitions, canonical)
        binary = canonical in BINARIOS
        values = _binary(rows) if binary else None
        question = questions.get(definition["PREGUNTA_ID"], {})
        valid = _binary_rows(rows) if binary else _validos(rows)
        if binary:
            valid_rows += valid
        criteria.append({"id": canonical, "nombre": name, "pregunta_id": definition["PREGUNTA_ID"],
                         "tipo_dato": definition["TIPO_DATO"], "escala_id": definition.get("ESCALA_ID"),
                         "resultado": values["si_pct"] if binary else question.get("indice"),
                         "unidad": "porcentaje" if binary else "indice", "n": len(valid),
                         "encuestas_validas": _surveys(valid), "historico_disponible": False,
                         "binaria": values,
                         **{field: question.get(field) for field in ("favorable_pct", "neutral_pct", "desfavorable_pct")}})
    usage_rows = _rows(selected, definitions, "EC_USO")
    media_rows = _descriptive(_rows(selected, definitions, "EC_MEDIO"))
    media_counts = Counter(_value(r) for r in media_rows)
    return {"criterios": criteria, "respuestas": sum(r["n"] for r in criteria),
            "encuestas_analizadas": _surveys(valid_rows + _binary_rows(usage_rows) + media_rows + comment_rows),
            "participacion": _binary(usage_rows),
            "medios": {"n": len(media_rows), "valores": [{"nombre": name, "n": media_counts[name], "porcentaje": round(media_counts[name] * 100 / len(media_rows), 2)} for name in sorted(media_counts)]},
            "comentarios": engine["comentarios"],
            "opciones": {"carreras": sorted({str(r.get("CARRERA_BASE") or "").strip() for r in all_rows} - {""}),
                         "modalidades": sorted({str(r.get("MODALIDAD_ACADEMICA") or r.get("MODALIDAD_DETALLE") or "").strip() for r in all_rows} - {""}),
                         "participacion": [label for label, value in [("si", 1), ("no", 0)] if any(_numero(r.get("VALOR_BASE")) == value for r in _binary_rows(_rows(all_rows, definitions, "EC_USO")))]},
            "poblacion": {"carrera": carrera or None, "modalidad_academica": modalidad_academica or None, "participacion": participacion or None}}
