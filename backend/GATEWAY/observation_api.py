import pandas as pd
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from backend.MODELS.observation.chat import (
    ObservationChatRequest,
    ObservationChatResponse,
)
from backend.MODELS.observation.summary import ObservationSummary
from backend.REPOSITORIES.observation.observation_repository import (
    ObservationRepository,
)
from backend.SERVICES.observation.observation_chat import (
    responder_pregunta_observacion,
)
from backend.SERVICES.observation.observation_insights import (
    generar_insights_observacion,
)
from backend.SERVICES.observation.observation_kpis import (
    calcular_kpis,
    preparar_observaciones,
)
from backend.SERVICES.observation.observation_pdf import (
    generar_pdf_docente,
)
from backend.SERVICES.observation.observation_priority import (
    obtener_casos_prioritarios,
)
from backend.SERVICES.observation.observation_query import (
    buscar_docentes,
    obtener_contexto_consulta,
)
from backend.SERVICES.observation.observation_teacher_detail import (
    obtener_detalle_docente,
)
from backend.SERVICES.observation.observation_trends import (
    calcular_tendencia_por_corte,
)

from backend.SERVICES.observation.observation_executive_pdf import (
    generar_pdf_ejecutivo_observacion,
)

router = APIRouter(
    prefix="/api/observacion",
    tags=["Observación de Clases"],
)

repository = ObservationRepository()


def aplicar_filtros(
    df: pd.DataFrame,
    corte: str | None = None,
    servicio: str | None = None,
    tipo: str | None = None,
) -> pd.DataFrame:
    df_filtrado = df.copy()

    if corte and corte != "Todos los cortes":
        df_filtrado = df_filtrado[
            df_filtrado["Corte"].astype(str).str.strip()
            == corte.strip()
        ]

    if (
        servicio
        and servicio != "Todos los servicios"
        and "Indica el servicio" in df_filtrado.columns
    ):
        df_filtrado = df_filtrado[
            df_filtrado["Indica el servicio"].astype(str).str.strip()
            == servicio.strip()
        ]

    tipo_col = None

    if "Tipo de observación" in df_filtrado.columns:
        tipo_col = "Tipo de observación"

    elif "Tipo de observación " in df_filtrado.columns:
        tipo_col = "Tipo de observación "

    if (
        tipo
        and tipo != "Todos los tipos"
        and tipo_col is not None
    ):
        df_filtrado = df_filtrado[
            df_filtrado[tipo_col].astype(str).str.strip()
            == tipo.strip()
        ]

    return df_filtrado


@router.get("/filtros")
def obtener_filtros():
    df_respuestas, df_cortes = repository.obtener_datos()

    df = preparar_observaciones(
        df_respuestas,
        df_cortes,
    )

    cortes = []

    if not df_cortes.empty and "Corte" in df_cortes.columns:
        cortes_ordenados = df_cortes.copy()

        cortes_ordenados.columns = [
            c.strip() if isinstance(c, str) else c
            for c in cortes_ordenados.columns
        ]

        if "Fecha_inicio" in cortes_ordenados.columns:
            cortes_ordenados["Fecha_inicio"] = pd.to_datetime(
                cortes_ordenados["Fecha_inicio"],
                errors="coerce",
                dayfirst=True,
            )

            cortes_ordenados = cortes_ordenados.sort_values(
                "Fecha_inicio",
                kind="stable",
            )

        cortes = (
            cortes_ordenados["Corte"]
            .dropna()
            .astype(str)
            .str.strip()
            .loc[
                lambda serie: (
                    (serie != "")
                    & (serie != "Sin corte")
                )
            ]
            .drop_duplicates()
            .tolist()
        )

    servicios = []

    if "Indica el servicio" in df.columns:
        servicios = sorted(
            [
                str(valor).strip()
                for valor in df["Indica el servicio"]
                .dropna()
                .unique()
                .tolist()
                if str(valor).strip()
            ]
        )

    tipo_col = None

    if "Tipo de observación" in df.columns:
        tipo_col = "Tipo de observación"

    elif "Tipo de observación " in df.columns:
        tipo_col = "Tipo de observación "

    tipos = []

    if tipo_col is not None:
        tipos = sorted(
            [
                str(valor).strip()
                for valor in df[tipo_col]
                .dropna()
                .unique()
                .tolist()
                if str(valor).strip()
            ]
        )

    return {
        "cortes": cortes,
        "servicios": servicios,
        "tipos": tipos,
    }


@router.get("/tendencia")
def obtener_tendencia(
    servicio: str | None = Query(default=None),
    tipo: str | None = Query(default=None),
):
    df_respuestas, df_cortes = repository.obtener_datos()

    df = preparar_observaciones(
        df_respuestas,
        df_cortes,
    )

    df = aplicar_filtros(
        df,
        servicio=servicio,
        tipo=tipo,
    )

    return calcular_tendencia_por_corte(
        df,
        df_cortes,
    )


@router.get("/casos-prioritarios")
def obtener_prioritarios(
    corte: str | None = Query(default=None),
    servicio: str | None = Query(default=None),
    tipo: str | None = Query(default=None),
):
    df_respuestas, df_cortes = repository.obtener_datos()

    df = preparar_observaciones(
        df_respuestas,
        df_cortes,
    )

    df = aplicar_filtros(
        df,
        corte=corte,
        servicio=servicio,
        tipo=tipo,
    )

    return obtener_casos_prioritarios(df)


@router.get("/insights")
def obtener_insights(
    corte: str | None = Query(default=None),
    servicio: str | None = Query(default=None),
    tipo: str | None = Query(default=None),
):
    df_respuestas, df_cortes = repository.obtener_datos()

    df = preparar_observaciones(
        df_respuestas,
        df_cortes,
    )

    df = aplicar_filtros(
        df,
        corte=corte,
        servicio=servicio,
        tipo=tipo,
    )

    tendencia = calcular_tendencia_por_corte(
        df,
        df_cortes,
    )

    return generar_insights_observacion(
        df,
        tendencia,
    )


@router.get("/consulta")
def consultar_observaciones(
    docente: str | None = Query(default=None),
    servicio: str | None = Query(default=None),
    materia: str | None = Query(default=None),
    corte: str | None = Query(default=None),
    tipo: str | None = Query(default=None),
    clasificacion: str | None = Query(default=None),
):
    df_respuestas, df_cortes = repository.obtener_datos()

    df = preparar_observaciones(
        df_respuestas,
        df_cortes,
    )

    return obtener_contexto_consulta(
        df=df,
        docente=docente,
        servicio=servicio,
        materia=materia,
        corte=corte,
        tipo=tipo,
        clasificacion=clasificacion,
    )


@router.get("/buscar-docentes")
def consultar_docentes(
    nombre: str = Query(..., min_length=1),
):
    df_respuestas, df_cortes = repository.obtener_datos()

    df = preparar_observaciones(
        df_respuestas,
        df_cortes,
    )

    return buscar_docentes(
        df,
        nombre,
    )


@router.post(
    "/chat",
    response_model=ObservationChatResponse,
)
def conversar_observacion(
    payload: ObservationChatRequest,
):
    df_respuestas, df_cortes = repository.obtener_datos()

    df = preparar_observaciones(
        df_respuestas,
        df_cortes,
    )

    resultado = responder_pregunta_observacion(
        df,
        payload.pregunta,
        contexto=payload.contexto,
    )

    return ObservationChatResponse(
        respuesta=resultado.get(
            "respuesta",
            "",
        ),
        tipo_respuesta=resultado.get(
            "tipo_respuesta",
            "resumen",
        ),
        filtros=resultado.get(
            "filtros",
            {},
        ),
        datos=resultado.get(
            "datos",
            {},
        ),
        contexto=resultado.get(
            "contexto",
            {},
        ),
    )


@router.get("/docente/{nombre_docente}/pdf")
def exportar_pdf_docente(
    nombre_docente: str,
    corte: str | None = Query(default=None),
    servicio: str | None = Query(default=None),
    tipo: str | None = Query(default=None),
):
    df_respuestas, df_cortes = repository.obtener_datos()

    df = preparar_observaciones(
        df_respuestas,
        df_cortes,
    )

    df = aplicar_filtros(
        df,
        corte=corte,
        servicio=servicio,
        tipo=tipo,
    )

    detalle = obtener_detalle_docente(
        df,
        nombre_docente,
    )

    pdf_buffer = generar_pdf_docente(
        detalle,
    )

    nombre_limpio = (
        nombre_docente
        .strip()
        .replace(" ", "_")
    )

    nombre_archivo = (
        f"reporte_observacion_{nombre_limpio}.pdf"
    )

    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{nombre_archivo}"'
            )
        },
    )


@router.get("/docente/{nombre_docente}")
def obtener_docente(
    nombre_docente: str,
    corte: str | None = Query(default=None),
    servicio: str | None = Query(default=None),
    tipo: str | None = Query(default=None),
):
    df_respuestas, df_cortes = repository.obtener_datos()

    df = preparar_observaciones(
        df_respuestas,
        df_cortes,
    )

    df = aplicar_filtros(
        df,
        corte=corte,
        servicio=servicio,
        tipo=tipo,
    )

    return obtener_detalle_docente(
        df,
        nombre_docente,
    )

@router.get("/reporte-ejecutivo/pdf")
def exportar_reporte_ejecutivo(
    corte: str | None = Query(default=None),
    servicio: str | None = Query(default=None),
    tipo: str | None = Query(default=None),
):
    df_respuestas, df_cortes = repository.obtener_datos()

    df = preparar_observaciones(
        df_respuestas,
        df_cortes,
    )

    df_filtrado = aplicar_filtros(
        df,
        corte=corte,
        servicio=servicio,
        tipo=tipo,
    )

    kpis = calcular_kpis(
        df_filtrado
    )

    tendencia = calcular_tendencia_por_corte(
        df_filtrado,
        df_cortes,
    )

    insights = generar_insights_observacion(
        df_filtrado,
        tendencia,
    )

    casos_prioritarios = obtener_casos_prioritarios(
        df_filtrado
    )

    filtros = {
        "corte": (
            corte
            if corte
            else "Todos los cortes"
        ),
        "servicio": (
            servicio
            if servicio
            else "Todos los servicios"
        ),
        "tipo": (
            tipo
            if tipo
            else "Todos los tipos"
        ),
    }

    pdf_buffer = generar_pdf_ejecutivo_observacion(
        kpis=kpis,
        insights=insights,
        casos_prioritarios=casos_prioritarios,
        filtros=filtros,
    )

    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                'attachment; filename="'
                'reporte_ejecutivo_observacion_clases.pdf"'
            )
        },
    )


@router.get(
    "/resumen",
    response_model=ObservationSummary,
)
def obtener_resumen(
    corte: str | None = Query(default=None),
    servicio: str | None = Query(default=None),
    tipo: str | None = Query(default=None),
):
    df_respuestas, df_cortes = repository.obtener_datos()

    df = preparar_observaciones(
        df_respuestas,
        df_cortes,
    )

    df = aplicar_filtros(
        df,
        corte=corte,
        servicio=servicio,
        tipo=tipo,
    )

    kpis = calcular_kpis(df)

    return ObservationSummary(
        observaciones=kpis["observaciones"],
        consolidado=kpis["consolidado"],
        en_proceso=kpis["en_proceso"],
        no_consolidado=kpis["no_consolidado"],
        promedio=kpis["promedio"],
        casos_criticos=kpis["no_consolidado"],
    )