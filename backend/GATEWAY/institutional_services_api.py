"""Isolated routes: existing section endpoints and exports remain unchanged."""
from fastapi import APIRouter, HTTPException, Depends
from backend.GATEWAY.quality_access import require_quality
from backend.CORE.auth.quality_scope import quality_scope
from backend.GATEWAY.calidad_api import (
    _obtener_base_preview, _CONFIG_MODALIDADES_2025, _CONFIG_MODALIDADES_2026,
    _normalizar_servicio_filtro,
    repository,
)
from backend.FUENTES.calidad.google_sheets import leer_encabezados
from backend.SERVICES.calidad.normalizer import cargar_mapa_preguntas
from backend.SERVICES.calidad.institutional_services import resumen_actual, resumen_historico, clave, nombre_desde_encabezado

router = APIRouter(dependencies=[Depends(require_quality)], prefix="/api/encuesta-calidad/servicios-institucionales", tags=["Servicios institucionales"])


def _fuente_base(periodo, modalidad, servicio):
    configs = {"2025": _CONFIG_MODALIDADES_2025, "2026": _CONFIG_MODALIDADES_2026}
    if periodo not in configs or modalidad not in configs[periodo]:
        raise HTTPException(status_code=400, detail="Periodo o modalidad no soportados")
    base = _obtener_base_preview(modalidad, periodo)
    mapa = cargar_mapa_preguntas(configs[periodo][modalidad]["fuente_id"], int(periodo))
    registros = base["registros"]
    if servicio:
        registros = [row for row in registros if clave(_normalizar_servicio_filtro(row.get("SERVICIO_PROGRAMA"))) == clave(_normalizar_servicio_filtro(servicio))]
    return registros, mapa


def _fuente(periodo, modalidad, servicio):
    scope = quality_scope.get()
    registros, mapa = _fuente_base(periodo, modalidad, "" if scope else servicio)
    return (scope.select(registros) if scope else registros), mapa


@router.get("/actual")
def actual(periodo: str = "2026", modalidad: str = "Escolarizadas", servicio: str = "", carrera: str = "", modalidad_academica: str = ""):
    rows, mapa = _fuente(periodo, modalidad, servicio)
    # The current map has a generic matrix label. Read its institutional name
    # from the same source column; no alias or historical mapping is introduced.
    generic = [row for row in mapa if row.get("PREGUNTA_CANONICA_ID") == "SER_PARAMEDICO"
               and clave(row.get("TEXTO_CORTO")).startswith("servicio institucional ")]
    if generic:
        source = (_CONFIG_MODALIDADES_2025 if periodo == "2025" else _CONFIG_MODALIDADES_2026)[modalidad]["fuente_id"]
        state = repository.obtener_estado_fuente(source)
        headers = leer_encabezados(str(state["SPREADSHEET_ID"]).strip(), str(state["HOJA_RESPUESTAS"]).strip())
        labels = {}
        for definition in generic:
            column = 0
            for letter in str(definition["COLUMNA_ORIGEN"]).strip().upper():
                column = column * 26 + ord(letter) - ord("A") + 1
            if 0 < column <= len(headers):
                labels[definition["PREGUNTA_ID"]] = nombre_desde_encabezado(definition, headers[column - 1])
        mapa = [labels.get(row.get("PREGUNTA_ID"), row) for row in mapa]
    try:
        return resumen_actual(rows, mapa, carrera, modalidad_academica)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/historico")
def historico(modalidad: str = "Escolarizadas", servicio: str = "", carrera: str = "", modalidad_academica: str = ""):
    current, current_map = _fuente("2026", modalidad, servicio)
    previous, previous_map = _fuente("2025", modalidad, servicio)
    try:
        return resumen_historico(current, current_map, previous, previous_map, carrera, modalidad_academica)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
