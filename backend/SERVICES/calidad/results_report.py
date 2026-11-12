"""PDF-only projection. All scores come from the existing, approved services."""
from backend.SERVICES.calidad import (
    academic_services, continuing_education, facilities, institutional_services,
    language_center, satisfaction, school_environment,
)
from backend.SERVICES.calidad.kpi_service import calcular_kpis_preview, _es_si


def _key(value):
    return institutional_services.clave(value)


def scoped(rows, career="", modality=""):
    return [r for r in rows
            if (not career or _key(r.get("CARRERA_BASE")) == _key(career))
            and (not modality or _key(r.get("MODALIDAD_ACADEMICA") or r.get("MODALIDAD_DETALLE")) == _key(modality))]


def _section(summary, section_id):
    return next((r for r in summary["secciones"]
                 if r["seccion_id"].replace("ESC25_", "ESC_", 1) == section_id), {})


def _metric(identifier, name, current, n, previous=None, compared_current=None, unit="pts"):
    comparable = previous is not None and compared_current is not None
    return {"id": identifier, "name": name, "value": current, "n": n, "unit": unit,
            "previous": previous if comparable else None,
            "compared_current": compared_current if comparable else None,
            "delta": round(compared_current - previous, 2) if comparable else None}


def ordered(criteria):
    return sorted(criteria, key=lambda r: (r["value"] is None, r["value"] if r["value"] is not None else 0, r["name"]))


def prepare_report(current_rows, current_map, old_rows, old_map, *, career="", modality="", source="Escolarizadas", service="", institutional_current=None, language="", course="", participation=""):
    """Inputs already respect the existing service filter; apply the same scope to both years."""
    # A program may identify one career/modality without separate controls.
    # If we print that identity, constrain BOTH years to those exact source fields.
    if service:
        program_rows = scoped(current_rows, career, modality)
        careers = {r.get("CARRERA_BASE") for r in program_rows if r.get("CARRERA_BASE")}
        modalities = {r.get("MODALIDAD_ACADEMICA") or r.get("MODALIDAD_DETALLE") for r in program_rows} - {None, ""}
        if not career and len(careers) == 1:
            career = next(iter(careers))
        if not modality and len(modalities) == 1:
            modality = next(iter(modalities))
    now, old = scoped(current_rows, career, modality), scoped(old_rows, career, modality)
    current = calcular_kpis_preview(now)
    comparable_now = calcular_kpis_preview(now, solo_comparables_historico=True)
    comparable_old = calcular_kpis_preview(old, solo_comparables_historico=True)
    definitions = {r.get("PREGUNTA_ID"): r for r in current_map}
    old_definitions = {r.get("PREGUNTA_CANONICA_ID"): r for r in old_map if r.get("PREGUNTA_CANONICA_ID")}
    old_questions = {r["pregunta_id"]: r for r in comparable_old["preguntas"]}

    def global_history(section_id):
        return (_section(comparable_old, section_id).get("indice"),
                _section(comparable_now, section_id).get("indice"))

    modules = []
    # Direction uses only the same four approved IDs as the dashboard.
    direction = []
    for identifier, name in [("ESC_DIR_01", "Disponibilidad"), ("ESC_DIR_02", "Atención a necesidades"),
                             ("ESC_DIR_03", "Solución de problemas"), ("ESC_DIR_04", "Seguimiento / desempeño docente")]:
        question = next((r for r in current["preguntas"] if r["pregunta_id"] == identifier), {})
        definition = definitions.get(identifier, {})
        previous_definition = old_definitions.get(definition.get("PREGUNTA_CANONICA_ID"), {})
        approved = _es_si(definition.get("COMPARABLE_HISTORICO")) and _es_si(previous_definition.get("COMPARABLE_HISTORICO"))
        previous = old_questions.get(previous_definition.get("PREGUNTA_ID"), {}).get("indice") if approved else None
        direction.append(_metric(identifier, name, question.get("indice"), question.get("n", 0), previous,
                                 question.get("indice") if approved else None))
    direction_surveys = academic_services._encuestas(institutional_services._validos(
        [r for r in now if r.get("PREGUNTA_ID") in {"ESC_DIR_01", "ESC_DIR_02", "ESC_DIR_03", "ESC_DIR_04"}]))
    modules.append({"id": "direction", "name": "Dirección", "index": _section(current, "ESC_02_DIRECCION").get("indice"),
                    "responses": sum(r["n"] for r in direction), "surveys": direction_surveys,
                    "history": global_history("ESC_02_DIRECCION"), "criteria": ordered(direction)})

    # Do not construct a new aggregate for the four staff criteria. Preserve each
    # existing person's criteria and show people alphabetically, never as a ranking.
    staff_questions = [r for r in current["preguntas"] if r["seccion_id"] == "ESC_01_PERSONAL"]
    people = []
    for person in current["personas"]:
        if person["seccion_id"] != "ESC_01_PERSONAL":
            continue
        questions = [q for q in staff_questions if (q["persona_evaluada"], q["rol_evaluado"]) == (person["persona_evaluada"], person["rol_evaluado"])]
        people.append({"name": person["persona_evaluada"], "role": person["rol_evaluado"], "value": person["indice"],
                       "surveys": person["evaluaciones"], "criteria": ordered([
                           _metric(q["pregunta_id"], q["texto_corto"], q["indice"], q["n"]) for q in questions])})
    observed = {(p["name"], p["role"]) for p in people}
    observed_names = {p["name"] for p in people}
    for assignment in current["comparativos_personal"].get("asignaciones", []):
        identity = (assignment["persona_evaluada"], assignment["rol_evaluado"])
        if identity[0] in observed_names or identity in observed or (career and _key(assignment["carrera_base"]) != _key(career)) or (modality and _key(assignment["modalidad_academica"]) != _key(modality)):
            continue
        observed.add(identity)
        observed_names.add(identity[0])
        people.append({"name": identity[0], "role": identity[1], "value": None, "surveys": 0, "criteria": []})
    staff_counts = current["comparativos_personal"]["resumen"]
    modules.append({"id": "staff", "name": "Asistentes/coordinadores", "index": _section(current, "ESC_01_PERSONAL").get("indice"),
                    "responses": sum(q["n"] for q in staff_questions), "surveys": staff_counts["evaluaciones"],
                    "people_count": staff_counts["personas_evaluadas"], "history": global_history("ESC_01_PERSONAL"),
                    "people": sorted(people, key=lambda p: p["name"].casefold()), "criteria": []})

    for identifier, name, service_module, field in [
        ("institutional", "Servicios institucionales", institutional_services, "servicios"),
        ("academic", "Servicios académicos", academic_services, "criterios"),
        ("facilities", "Instalaciones y equipo tecnológico", facilities, "criterios"),
        ("environment", "Ambiente escolar", school_environment, "criterios"),
    ]:
        actual = institutional_current if identifier == "institutional" and institutional_current is not None else service_module.resumen_actual(now, current_map, career, modality)
        history = service_module.resumen_historico(now, current_map, old, old_map, career, modality)
        past = {r["id"]: r for r in history[field]}
        criteria = [_metric(r["id"], r["nombre"], r["indice"], r["n"], past.get(r["id"], {}).get("indice_2025"),
                            past.get(r["id"], {}).get("indice_2026")) for r in actual[field]]
        module = {"id": identifier, "name": name, "index": actual["indice"], "responses": actual["respuestas"],
                  "surveys": actual["encuestas_con_evaluacion"] if identifier == "institutional" else actual["encuestas_analizadas"], "history": (history["indice_2025"], history["indice_2026"]),
                  "criteria": ordered(criteria)}
        if identifier == "environment":
            incidence = actual["incidencia"]
            h = history["incidencia"]
            module["incidence"] = _metric("AMB_PROBLEMA", "Incidencia reportada", incidence["si_pct"], incidence["total_valido"],
                                          h["porcentaje_2025"], h["porcentaje_2026"], "pp")
            module["incidence"].update({"yes": incidence["si"], "no": incidence["no"]})
        modules.append(module)

    actual = satisfaction.resumen_actual(now, current_map, career, modality)
    history = satisfaction.resumen_historico(now, current_map, old, old_map, career, modality)
    sat, rec = actual["satisfaccion"], actual["recomendacion"]
    modules.append({"id": "satisfaction", "name": "Satisfacción y recomendación", "index": None,
                    "surveys": actual["encuestas_analizadas"], "history": None,
                    "criteria": ordered([
                        _metric("SAT_GENERAL", "Satisfacción general", sat["indice"], sat["n"], history["satisfaccion"]["indice_2025"], history["satisfaccion"]["indice_2026"]),
                        _metric("REC_UDL", "Recomendación", rec["si_pct"], rec["total_valido"], history["recomendacion"]["porcentaje_2025"], history["recomendacion"]["porcentaje_2026"], "pp")])})
    for identifier, name, service_module in [("languages", "Centro de Idiomas", language_center), ("continuing", "Educación Continua", continuing_education)]:
        actual = service_module.resumen_actual(now, current_map, career, modality, language, course) if identifier == "languages" else service_module.resumen_actual(now, current_map, career, modality, participation)
        criteria = [_metric(r["id"], r["nombre"], r.get("resultado") if identifier == "continuing" else r["indice"], r["n"],
                            unit="pp" if r.get("unidad") == "porcentaje" else "pts") for r in actual["criterios"]]
        usage = actual["uso"] if identifier == "languages" else actual["participacion"]
        modules.append({"id": identifier, "name": name, "index": None, "responses": actual["respuestas"],
                        "surveys": actual["encuestas_analizadas"], "history": None, "current_only": True,
                        "usage": usage, "criteria": ordered(criteria),
                        "local_filters": {"Idioma": language, "Tipo de curso": course} if identifier == "languages" else {"Participación": participation}})

    # Resolve labels only when the existing program selection has one identity.
    careers = {r.get("CARRERA_BASE") for r in now if r.get("CARRERA_BASE")}
    modalities = {r.get("MODALIDAD_ACADEMICA") or r.get("MODALIDAD_DETALLE") for r in now} - {None, ""}
    return {"title": "Informe de resultados", "period": "2026", "source": source, "service": service,
            "career": career or (next(iter(careers)) if service and len(careers) == 1 else service or "Todas las carreras"),
            "modality": modality or (next(iter(modalities)) if service and len(modalities) == 1 else "Todas las modalidades"),
            "index": current["indice_general"], "surveys": current["encuestas"], "responses": current["respuestas_kpi_validas"],
            "modules": modules}
