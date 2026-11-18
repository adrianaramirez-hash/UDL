from fastapi import APIRouter, HTTPException, Depends
from backend.GATEWAY.quality_access import require_quality
from backend.GATEWAY.institutional_services_api import _fuente
from backend.SERVICES.calidad.language_center import resumen_actual

router = APIRouter(dependencies=[Depends(require_quality)], prefix="/api/encuesta-calidad/centro-idiomas", tags=["Centro de Idiomas"])


@router.get("/actual")
def actual(periodo: str = "2026", modalidad: str = "Escolarizadas", servicio: str = "", carrera: str = "", modalidad_academica: str = "", idioma: str = "", tipo_curso: str = ""):
    rows, mapa = _fuente(periodo, modalidad, servicio)
    try:
        return resumen_actual(rows, mapa, carrera, modalidad_academica, idioma, tipo_curso)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
