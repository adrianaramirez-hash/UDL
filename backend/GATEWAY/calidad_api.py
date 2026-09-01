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
    normalizar_filas_preview,
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
_cache_base = {}
_cache_expira = {}

_CONFIG_MODALIDADES_2026 = {
    "Escolarizadas": {
        "fuente_id": "ESC_MAIN",
        "periodo_id": "ESC_2026_NUEVO",
        "version_instrumento": "FORM_2026",
    },
    "Virtuales": {
        "fuente_id": "VIR_MAIN",
        "periodo_id": "VIR_2026",
        "version_instrumento": "FORM_2025",
    },
    "Prepa": {
        "fuente_id": "PRE_MAIN",
        "periodo_id": "PRE_2026",
        "version_instrumento": "FORM_2025",
    },
}

_CONFIG_MODALIDADES_2025 = {
    "Escolarizadas": {
        "fuente_id": "ESC_2025_ARCHIVE",
        "periodo_id": "ESC_2025_NOVDIC",
        "version_instrumento": "LEGACY_2025",
        "fila_inicio": 2,
        "fila_fin": 1755,
    },
    "Virtuales": {
        "fuente_id": "VIR_MAIN",
        "periodo_id": "VIR_2025",
        "version_instrumento": "FORM_2025",
        "fila_inicio": 2,
        "fila_fin": 24,
    },
    "Prepa": {
        "fuente_id": "PRE_MAIN",
        "periodo_id": "PRE_2025",
        "version_instrumento": "FORM_2025",
        "fila_inicio": 2,
        "fila_fin": 249,
    },
}



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


def _calcular_base_preview(
    modalidad_fuente: str = "Escolarizadas",
    periodo: str = "2026",
):
    """
    Lee y normaliza las respuestas nuevas de una
    modalidad de Encuesta de Calidad.

    No escribe datos en el MASTER.
    """

    if periodo == "2025":
        configuraciones = _CONFIG_MODALIDADES_2025
        anio = 2025
    elif periodo == "2026":
        configuraciones = _CONFIG_MODALIDADES_2026
        anio = 2026
    else:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "PERIODO_NO_SOPORTADO",
                "periodo": periodo,
            },
        )

    configuracion = configuraciones.get(
        modalidad_fuente
    )

    if configuracion is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "MODALIDAD_NO_SOPORTADA",
                "modalidad": modalidad_fuente,
            },
        )

    fuente_id = configuracion["fuente_id"]

    validacion = None

    if periodo == "2026":
        validaciones = validar_firmas_activas()

        validacion = next(
            (
                item
                for item in validaciones
                if item["fuente_id"] == fuente_id
            ),
            None,
        )

        if (
            validacion is None
            or not validacion["ok"]
        ):
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "SOURCE_STRUCTURE_INVALID",
                    "validacion": validacion,
                },
            )

    estado = repository.obtener_estado_fuente(
        fuente_id
    )

    if estado is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"{fuente_id} no existe "
                "en 12_ESTADO_ETL."
            ),
        )

    if periodo == "2025":
        fila_inicio = int(
            configuracion["fila_inicio"]
        )
        fila_fin = int(
            configuracion["fila_fin"]
        )
    else:
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
            "registros": [],
            "qa": [],
            "fila_inicio": fila_inicio,
            "fila_fin": fila_fin,
            "fuente_id": fuente_id,
            "periodo_id": configuracion[
                "periodo_id"
            ],
            "version_instrumento": configuracion[
                "version_instrumento"
            ],
        }

    if periodo == "2025":
        columnas_historicas = {
            "ESC_2025_ARCHIVE": "BK",
            "VIR_MAIN": "CD",
            "PRE_MAIN": "BU",
        }

        columna_fin = columnas_historicas.get(
            fuente_id
        )

        if columna_fin is None:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "COLUMNAS_HISTORICAS_NO_CONFIGURADAS",
                    "fuente_id": fuente_id,
                },
            )
    else:
        columna_fin = _numero_a_columna(
            int(
                validacion[
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

    registros, qa = normalizar_filas_preview(
        filas=filas,
        fila_inicio=fila_inicio,
        fuente_id=fuente_id,
        anio=anio,
        periodo_id=configuracion[
            "periodo_id"
        ],
        version_instrumento=configuracion[
            "version_instrumento"
        ],
    )

    return {
        "registros": registros,
        "qa": qa,
        "fila_inicio": fila_inicio,
        "fila_fin": fila_fin,
        "fuente_id": fuente_id,
        "periodo_id": configuracion[
            "periodo_id"
        ],
        "version_instrumento": configuracion[
            "version_instrumento"
        ],
    }


def _obtener_base_preview(
    modalidad_fuente: str = "Escolarizadas",
    periodo: str = "2026",
):
    ahora = time.monotonic()
    clave_cache = f"{periodo}:{modalidad_fuente}"

    base_cache = _cache_base.get(clave_cache)

    expira = _cache_expira.get(
            clave_cache,
        0.0,
    )

    if (
        base_cache is not None
        and ahora < expira
    ):
        return base_cache

    with _cache_lock:
        ahora = time.monotonic()

        base_cache = _cache_base.get(clave_cache)

        expira = _cache_expira.get(
            clave_cache,
            0.0,
        )

        if (
            base_cache is not None
            and ahora < expira
        ):
            return base_cache

        base = _calcular_base_preview(
            modalidad_fuente,
            periodo,
        )

        _cache_base[clave_cache] = base

        _cache_expira[clave_cache] = (
            ahora + _CACHE_TTL_SECONDS
        )

        return base

def _normalizar_servicio_filtro(valor: str) -> str:
    texto = " ".join(
        str(valor or "").strip().split()
    )

    if texto.casefold() == (
        "maestría:educación especial"
    ).casefold():
        return "Maestría en Educación Especial"

    return texto


@router.get("/preview/resumen")
def obtener_resumen_preview(
    periodo: str | None = None,
    modalidad: str | None = None,
    servicio: str | None = None,
):
    periodo_seleccionado = (
        periodo.strip()
        if periodo and periodo.strip()
        else "2026"
    )

    if periodo_seleccionado not in {"2025", "2026"}:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "PERIODO_NO_SOPORTADO_PREVIEW",
                "periodo": periodo_seleccionado,
            },
        )

    modalidad_seleccionada = (
        modalidad.strip()
        if modalidad and modalidad.strip()
        else "Escolarizadas"
    )

    configuraciones_periodo = (
        _CONFIG_MODALIDADES_2025
        if periodo_seleccionado == "2025"
        else _CONFIG_MODALIDADES_2026
    )

    if (
        modalidad_seleccionada
        not in configuraciones_periodo
    ):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "MODALIDAD_NO_SOPORTADA",
                "modalidad": modalidad_seleccionada,
            },
        )

    base = _obtener_base_preview(
        modalidad_seleccionada,
        periodo_seleccionado,
    )

    registros = base["registros"]
    qa = base["qa"]

    registros_filtrados = registros

    if servicio and servicio.strip():
        servicio_limpio = (
            _normalizar_servicio_filtro(
                servicio
            )
        )

        registros_filtrados = [
            registro
            for registro in registros
            if _normalizar_servicio_filtro(
                registro.get(
                    "SERVICIO_PROGRAMA",
                    "",
                )
            ).casefold()
            == servicio_limpio.casefold()
        ]

    kpis = calcular_kpis_preview(
        registros_filtrados
    )

    modalidades = (
        [
            "Escolarizadas",
            "Virtuales",
            "Prepa",
        ]
        if periodo_seleccionado == "2025"
        else [
            "Escolarizadas",
            "Virtuales",
            "Prepa",
        ]
    )

    servicios = sorted(
        {
            _normalizar_servicio_filtro(
                registro.get(
                    "SERVICIO_PROGRAMA",
                    "",
                )
            )
            for registro in registros
            if _normalizar_servicio_filtro(
                registro.get(
                    "SERVICIO_PROGRAMA",
                    "",
                )
            )
        },
        key=str.casefold,
    )

    return {
        **kpis,
        "qa": qa,
        "qa_total": len(qa),
        "fila_inicio": base["fila_inicio"],
        "fila_fin": base["fila_fin"],
        "fuente_id": base["fuente_id"],
        "periodo_id": base["periodo_id"],
        "version_instrumento": base[
            "version_instrumento"
        ],
        "filtros": {
            "periodos": ["2025", "2026"],
            "modalidades": modalidades,
            "servicios": servicios,
            "seleccion": {
                "periodo": periodo_seleccionado,
                "modalidad": modalidad_seleccionada,
                "servicio": (
                    servicio.strip()
                    if servicio
                    else None
                ),
            },
        },
    }
