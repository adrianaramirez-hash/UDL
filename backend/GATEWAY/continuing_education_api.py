from fastapi import APIRouter, HTTPException
from backend.GATEWAY.institutional_services_api import _fuente
from backend.SERVICES.calidad.continuing_education import resumen_actual

router = APIRouter(prefix="/api/encuesta-calidad/educacion-continua", tags=["Educación Continua"])


@router.get("/actual")
def actual(periodo: str = "2026", modalidad: str = "Escolarizadas", servicio: str = "", carrera: str = "", modalidad_academica: str = "", participacion: str = ""):
    rows, mapa = _fuente(periodo, modalidad, servicio)
    try:
        return resumen_actual(rows, mapa, carrera, modalidad_academica, participacion)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
