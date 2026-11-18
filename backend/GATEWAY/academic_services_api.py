from fastapi import APIRouter, HTTPException, Depends
from backend.GATEWAY.quality_access import require_quality
from backend.GATEWAY.institutional_services_api import _fuente
from backend.SERVICES.calidad.academic_services import resumen_actual, resumen_historico

router = APIRouter(dependencies=[Depends(require_quality)], prefix="/api/encuesta-calidad/servicios-academicos", tags=["Servicios académicos"])


@router.get("/actual")
def actual(periodo: str = "2026", modalidad: str = "Escolarizadas", servicio: str = "", carrera: str = "", modalidad_academica: str = ""):
    rows, mapa = _fuente(periodo, modalidad, servicio)
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
