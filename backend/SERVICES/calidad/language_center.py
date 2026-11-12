"""Specialized 2026 experience and descriptive context, with no historical join."""
from collections import Counter

from backend.SERVICES.calidad.kpi_service import calcular_kpis_preview, _es_si, _numero
from backend.SERVICES.calidad.institutional_services import clave, _validos

CRITERIOS = {
    "IDI_GUSTO_CLASE": "¿En general te gusta tu clase?",
    "IDI_SAT_DOCENTE": "¿En general estás satisfecho con tu docente?",
    "IDI_SAT_OFICINA": "¿En general es satisfactorio para ti el servicio que te ofrecen en la oficina del Centro de Idiomas?",
}
CONTEXTOS = {"IDI_USO", "IDI_IDIOMA", "IDI_TIPO_CURSO", "IDI_PROFESOR"}
COMENTARIOS = {"IDI_MENOS_GUSTA", "IDI_MAS_GUSTA"}


def es_seccion(row):
    return str(row.get("SECCION_ID", "")).replace("ESC25_", "ESC_", 1) == "ESC_05_IDIOMAS"


def catalogo(mapa):
    result, questions = {}, set()
    for row in mapa:
        if not es_seccion(row):
            continue
        canonical = str(row.get("PREGUNTA_CANONICA_ID") or "").strip()
        if canonical not in {*CRITERIOS, *CONTEXTOS, *COMENTARIOS}:
            if row.get("TIPO_DATO") == "LIKERT" and _es_si(row.get("CALCULA_KPI_PREGUNTA")):
                raise ValueError("Criterio fuera de los tres especializados de Centro de Idiomas")
            continue
        question = str(row.get("PREGUNTA_ID") or "").strip()
        if not question or canonical in result or question in questions:
            raise ValueError("Pregunta de Centro de Idiomas sin ID o duplicada")
        if canonical in CRITERIOS and (row.get("TIPO_DATO") != "LIKERT" or not _es_si(row.get("CALCULA_KPI_PREGUNTA"))):
            raise ValueError("Escala incompatible para un criterio especializado")
        if canonical == "IDI_USO" and row.get("TIPO_DATO") != "BINARIA":
            raise ValueError("El uso del servicio requiere la escala binaria aprobada")
        if canonical in CONTEXTOS | COMENTARIOS and _es_si(row.get("ENTRA_PROMEDIO_SECCION")):
            raise ValueError("Las variables de contexto y comentarios no forman parte de satisfacción")
        result[canonical] = row
        questions.add(question)
    if not set(CRITERIOS) <= result.keys():
        raise ValueError("La evaluación especializada de Centro de Idiomas requiere los tres criterios 2026")
    return result


def _form(row):
    return (row.get("FUENTE_ID"), row.get("ANIO"), row.get("FILA_ORIGEN"))


def _surveys(rows):
    return len({_form(row) for row in rows if row.get("FILA_ORIGEN") not in (None, "")})


def _value(row):
    return str(row.get("RESPUESTA_NORMALIZADA") or row.get("RESPUESTA_ORIGINAL") or "").strip()


def _descriptive(rows):
    return [r for r in rows if _es_si(r.get("RESPUESTA_VALIDA")) and _value(r)]


def _rows(rows, definitions, canonical):
    definition = definitions.get(canonical)
    return [r for r in rows if definition and r.get("PREGUNTA_ID") == definition["PREGUNTA_ID"]]


def _usage_rows(rows):
    return [r for r in _validos(rows) if r.get("TIPO_DATO") == "BINARIA" and _numero(r.get("VALOR_BASE")) in (0, 1)]


def _usage(rows):
    valid = _usage_rows(rows)
    yes = sum(_numero(r.get("VALOR_BASE")) == 1 for r in valid)
    return {"si": yes, "no": len(valid) - yes, "total_valido": len(valid),
            "si_pct": round(yes * 100 / len(valid), 2) if valid else None,
            "no_pct": round((len(valid) - yes) * 100 / len(valid), 2) if valid else None}


def _distribution(rows):
    valid = _descriptive(rows)
    counts = Counter(_value(row) for row in valid)
    return {"n": len(valid), "valores": [{"nombre": name, "n": counts[name], "porcentaje": round(counts[name] * 100 / len(valid), 2)} for name in sorted(counts)]}


def resumen_actual(registros, mapa, carrera="", modalidad_academica="", idioma="", tipo_curso=""):
    definitions = catalogo(mapa)
    allowed = {r["PREGUNTA_ID"] for r in definitions.values()}
    all_rows = [r for r in registros if es_seccion(r) and r.get("PREGUNTA_ID") in allowed]
    selected = [r for r in all_rows if (not carrera or clave(r.get("CARRERA_BASE")) == clave(carrera))
                and (not modalidad_academica or clave(r.get("MODALIDAD_ACADEMICA") or r.get("MODALIDAD_DETALLE")) == clave(modalidad_academica))]
    # Language/course responses live on separate normalized question rows.
    # Filter survey identities, then project every metric and comment to them.
    for canonical, value in [("IDI_IDIOMA", idioma), ("IDI_TIPO_CURSO", tipo_curso)]:
        if value:
            forms = {_form(r) for r in _descriptive(_rows(selected, definitions, canonical)) if clave(_value(r)) == clave(value)}
            selected = [r for r in selected if _form(r) in forms]
    criterion_rows = [r for r in selected if r.get("PREGUNTA_ID") in {definitions[k]["PREGUNTA_ID"] for k in CRITERIOS}]
    comment_rows = [r for key in COMENTARIOS for r in _descriptive(_rows(selected, definitions, key))]
    projected = [{key: value for key, value in r.items() if key not in {"FUENTE_ID", "ANIO"}} for r in criterion_rows + comment_rows]
    metrics = calcular_kpis_preview(projected)
    questions = {r["pregunta_id"]: r for r in metrics["preguntas"]}
    criteria = []
    for canonical, name in CRITERIOS.items():
        definition = definitions[canonical]
        question = questions.get(definition["PREGUNTA_ID"], {})
        criteria.append({"id": canonical, "nombre": name, "pregunta_id": definition["PREGUNTA_ID"],
                         "indice": question.get("indice"), "n": question.get("n", 0),
                         "encuestas_validas": _surveys(_validos(_rows(selected, definitions, canonical))),
                         "historico_disponible": False,
                         **{field: question.get(field) for field in ("favorable_pct", "neutral_pct", "desfavorable_pct")}})
    usage_rows = _rows(selected, definitions, "IDI_USO")
    descriptive_rows = [r for key in CONTEXTOS - {"IDI_USO"} for r in _descriptive(_rows(selected, definitions, key))]
    teachers = _distribution(_rows(selected, definitions, "IDI_PROFESOR"))
    return {"criterios": criteria, "respuestas": sum(r["n"] for r in criteria),
            "encuestas_analizadas": _surveys(_validos(criterion_rows) + _usage_rows(usage_rows) + descriptive_rows + comment_rows),
            "uso": _usage(usage_rows), "idiomas": _distribution(_rows(selected, definitions, "IDI_IDIOMA")),
            "tipos_curso": _distribution(_rows(selected, definitions, "IDI_TIPO_CURSO")),
            "profesores": {"n": teachers["n"], "nombres": [r["nombre"] for r in teachers["valores"]]},
            "comentarios": metrics["comentarios"],
            "preguntas_comentarios": [{"pregunta_id": definitions[key]["PREGUNTA_ID"], "nombre": label} for key, label in [
                ("IDI_MENOS_GUSTA", "¿Qué es lo que menos te gusta del servicio del Centro de Idiomas?"),
                ("IDI_MAS_GUSTA", "¿Qué es lo que más te gusta del servicio que te ofrece el Centro de Idiomas?")]
                if key in definitions],
            "opciones": {"carreras": sorted({str(r.get("CARRERA_BASE") or "").strip() for r in all_rows} - {""}),
                         "modalidades": sorted({str(r.get("MODALIDAD_ACADEMICA") or r.get("MODALIDAD_DETALLE") or "").strip() for r in all_rows} - {""}),
                         "idiomas": [r["nombre"] for r in _distribution(_rows(all_rows, definitions, "IDI_IDIOMA"))["valores"]],
                         "tipos_curso": [r["nombre"] for r in _distribution(_rows(all_rows, definitions, "IDI_TIPO_CURSO"))["valores"]]},
            "poblacion": {"carrera": carrera or None, "modalidad_academica": modalidad_academica or None,
                          "idioma": idioma or None, "tipo_curso": tipo_curso or None}}
