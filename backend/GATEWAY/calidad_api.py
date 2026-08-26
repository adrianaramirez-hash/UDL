import threading
import time

from fastapi import APIRouter, HTTPException

from backend.FUENTES.calidad.google_sheets import (
    leer_filas,
    obtener_ultima_fila_con_datos,
)
from backend.REPOSITORIES.calidad.calidad_repository import (
    CalidadRepository,
)
from backend.SERVICES.calidad.kpi_service import (
    calcular_kpis_preview,
)
from backend.SERVICES.calidad.normalizer import (
    normalizar_filas_esc_preview,
)
from backend.SERVICES.calidad.structure_validator import (
    validar_firmas_activas,
)


router = APIRouter(
    prefix="/api/encuesta-calidad",
    tags=["Encuesta de Calidad"],
)

repository = CalidadRepository()

_CACHE_TTL_SECONDS = 60
_cache_lock = threading.Lock()
_cache_resumen = None
_cache_expira = 0.0


def _numero_a_columna(numero: int) -> str:
    resultado = ""

    while numero:
        numero, residuo = divmod(
            numero - 1,
            26,
        )

        resultado = (
            chr(65 + residuo)
            + resultado
        )

    return resultado


def _calcular_resumen_preview():
    """
    Devuelve KPIs de las respuestas nuevas de ESC_MAIN.

    No escribe datos en el MASTER.
    """

    validaciones = validar_firmas_activas()

    validacion_esc = next(
        (
            item
            for item in validaciones
            if item["fuente_id"] == "ESC_MAIN"
        ),
        None,
    )

    if (
        validacion_esc is None
        or not validacion_esc["ok"]
    ):
        raise HTTPException(
            status_code=409,
            detail={
                "error": "SOURCE_STRUCTURE_INVALID",
                "validacion": validacion_esc,
            },
        )

    estado = repository.obtener_estado_fuente(
        "ESC_MAIN"
    )

    if estado is None:
        raise HTTPException(
            status_code=404,
            detail="ESC_MAIN no existe en 12_ESTADO_ETL.",
        )

    fila_inicio = int(
        estado["SIGUIENTE_FILA_A_LEER"]
    )

    fila_fin = obtener_ultima_fila_con_datos(
        str(
            estado["SPREADSHEET_ID"]
        ).strip(),
        str(
            estado["HOJA_RESPUESTAS"]
        ).strip(),
    )

    if fila_fin < fila_inicio:
        return {
            "encuestas": 0,
            "respuestas_normalizadas": 0,
            "respuestas_kpi_validas": 0,
            "preguntas_con_datos": 0,
            "indice_general": None,
            "secciones_general": 0,
            "secciones": [],
            "preguntas": [],
            "qa": [],
            "fila_inicio": fila_inicio,
            "fila_fin": fila_fin,
        }

    columna_fin = _numero_a_columna(
        int(
            validacion_esc[
                "columnas_actuales"
            ]
        )
    )

    filas = leer_filas(
        str(
            estado["SPREADSHEET_ID"]
        ).strip(),
        str(
            estado["HOJA_RESPUESTAS"]
        ).strip(),
        fila_inicio,
        fila_fin,
        columna_fin,
    )

    registros, qa = (
        normalizar_filas_esc_preview(
            filas,
            fila_inicio,
        )
    )

    kpis = calcular_kpis_preview(
        registros
    )

    return {
        **kpis,
        "qa": qa,
        "qa_total": len(qa),
        "fila_inicio": fila_inicio,
        "fila_fin": fila_fin,
        "fuente_id": "ESC_MAIN",
        "periodo_id": "ESC_2026_NUEVO",
        "version_instrumento": "FORM_2026",
    }


@router.get("/preview/resumen")
def obtener_resumen_preview():
    global _cache_resumen, _cache_expira

    ahora = time.monotonic()

    if (
        _cache_resumen is not None
        and ahora < _cache_expira
    ):
        return _cache_resumen

    with _cache_lock:
        ahora = time.monotonic()

        if (
            _cache_resumen is not None
            and ahora < _cache_expira
        ):
            return _cache_resumen

        resumen = _calcular_resumen_preview()

        _cache_resumen = resumen
        _cache_expira = (
            ahora + _CACHE_TTL_SECONDS
        )

        return resumen
