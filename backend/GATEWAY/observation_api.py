from dataclasses import dataclass
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from backend.CORE.auth.quality_scope import service_key
from backend.GATEWAY.auth_api import DENIED, Principal, require_observation
from backend.MODELS.observation.chat import (
    ObservationChatRequest,
    ObservationChatResponse,
)
from backend.MODELS.observation.summary import ObservationSummary
from backend.REPOSITORIES.observation.observation_repository import (
    ObservationRepository,
)
from backend.SERVICES.observation.observation_chat import (
    _detectar_docente,
    _detectar_valor_columna,
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
    dependencies=[Depends(require_observation)],
)

repository = ObservationRepository()
SERVICE_COLUMN = "Indica el servicio"
TEACHER_COLUMN = "Nombre del docente"


@dataclass(frozen=True)
class ObservationData:
    rows: pd.DataFrame
    cuts: pd.DataFrame
    restricted: bool = False
    services: frozenset[str] = frozenset()
    chat_catalog: pd.DataFrame | None = None

    def service_filter(self, requested: str | None) -> str | None:
        # An assigned account has already been restricted by service.
        return None if self.restricted else requested

    def authorize_teacher(self, name: str) -> None:
        if self.restricted and (
            TEACHER_COLUMN not in self.rows.columns
            or not self.rows[TEACHER_COLUMN].astype(str).str.strip().eq(name.strip()).any()
        ):
            raise HTTPException(403, DENIED)

    def authorize_chat(self, payload: ObservationChatRequest) -> None:
        if not self.restricted or self.chat_catalog is None:
            return
        context = payload.contexto
        if context.get("alcance", "asignado") != "asignado":
            raise HTTPException(403, DENIED)
        service = _detectar_valor_columna(self.chat_catalog, payload.pregunta, [SERVICE_COLUMN])
        for requested in (service, context.get("ultimo_servicio")):
            if requested and service_key(requested) not in self.services:
                raise HTTPException(403, DENIED)
        if teacher := context.get("ultimo_docente"):
            self.authorize_teacher(str(teacher))
        teacher, matches = _detectar_docente(self.chat_catalog, payload.pregunta)
        if teacher:
            self.authorize_teacher(teacher)
        elif matches:
            teachers = set(self.rows[TEACHER_COLUMN].astype(str).str.strip())
            if not any(match["docente"] in teachers for match in matches):
                raise HTTPException(403, DENIED)


def load_observation_data(
    request: Request, principal: Principal = Depends(require_observation),
) -> ObservationData:
    if principal.grants.role == "DG":
        responses, cuts = repository.obtener_datos()
        return ObservationData(preparar_observaciones(responses, cuts), cuts)
    if principal.grants.role != "DC":
        raise HTTPException(403, DENIED)

    services = frozenset(filter(None, map(service_key, principal.grants.services)))
    params = request.query_params
    if (not services or params.get("alcance", "asignado") != "asignado" or "global" in params
            or any(len(params.getlist(key)) > 1 for key in ("servicio", "carrera", "alcance"))):
        raise HTTPException(403, DENIED)
    for key in ("servicio", "carrera"):
        if key in params:
            requested = service_key(params[key])
            if requested not in services:
                raise HTTPException(403, DENIED)
            services = frozenset({requested})

    responses, cuts = repository.obtener_datos()
    responses = responses.rename(columns=str.strip)
    if list(responses.columns).count(SERVICE_COLUMN) != 1:
        raise HTTPException(503, "No se pudo determinar el servicio de Observación")
    rows = preparar_observaciones(responses[responses[SERVICE_COLUMN].fillna("").map(service_key).isin(services)], cuts)
    cuts = cuts.rename(columns=str.strip)
    if "Corte" in cuts.columns:
        cuts = cuts[cuts["Corte"].astype(str).str.strip().isin(rows["Corte"])]
    catalog = None
    if request.url.path.endswith("/chat"):
        catalog = responses[[column for column in (SERVICE_COLUMN, TEACHER_COLUMN) if column in responses.columns]]
    return ObservationData(rows, cuts, restricted=True, services=services, chat_catalog=catalog)


ScopedData = Annotated[ObservationData, Depends(load_observation_data)]


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
def obtener_filtros(data: ScopedData):
    df, df_cortes = data.rows, data.cuts

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
    data: ScopedData,
    servicio: str | None = Query(default=None),
    tipo: str | None = Query(default=None),
):
    df, df_cortes = data.rows, data.cuts

    df = aplicar_filtros(
        df,
        servicio=data.service_filter(servicio),
        tipo=tipo,
    )

    return calcular_tendencia_por_corte(
        df,
        df_cortes,
    )


@router.get("/casos-prioritarios")
def obtener_prioritarios(
    data: ScopedData,
    corte: str | None = Query(default=None),
    servicio: str | None = Query(default=None),
    tipo: str | None = Query(default=None),
):
    df = data.rows

    df = aplicar_filtros(
        df,
        corte=corte,
        servicio=data.service_filter(servicio),
        tipo=tipo,
    )

    return obtener_casos_prioritarios(df)


@router.get("/insights")
def obtener_insights(
    data: ScopedData,
    corte: str | None = Query(default=None),
    servicio: str | None = Query(default=None),
    tipo: str | None = Query(default=None),
):
    df, df_cortes = data.rows, data.cuts

    df = aplicar_filtros(
        df,
        corte=corte,
        servicio=data.service_filter(servicio),
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
    data: ScopedData,
    docente: str | None = Query(default=None),
    servicio: str | None = Query(default=None),
    materia: str | None = Query(default=None),
    corte: str | None = Query(default=None),
    tipo: str | None = Query(default=None),
    clasificacion: str | None = Query(default=None),
):
    df = data.rows

    if data.restricted and docente is not None and not buscar_docentes(df, docente):
        raise HTTPException(403, DENIED)
    return obtener_contexto_consulta(
        df=df,
        docente=docente,
        servicio=data.service_filter(servicio),
        materia=materia,
        corte=corte,
        tipo=tipo,
        clasificacion=clasificacion,
    )


@router.get("/buscar-docentes")
def consultar_docentes(
    data: ScopedData,
    nombre: str = Query(..., min_length=1),
):
    df = data.rows

    matches = buscar_docentes(df, nombre)
    if data.restricted and not matches:
        raise HTTPException(403, DENIED)
    return matches


@router.post(
    "/chat",
    response_model=ObservationChatResponse,
)
def conversar_observacion(
    data: ScopedData,
    payload: ObservationChatRequest,
):
    df = data.rows

    data.authorize_chat(payload)
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
    data: ScopedData,
    nombre_docente: str,
    corte: str | None = Query(default=None),
    servicio: str | None = Query(default=None),
    tipo: str | None = Query(default=None),
):
    df = data.rows

    data.authorize_teacher(nombre_docente)
    df = aplicar_filtros(
        df,
        corte=corte,
        servicio=data.service_filter(servicio),
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
    data: ScopedData,
    nombre_docente: str,
    corte: str | None = Query(default=None),
    servicio: str | None = Query(default=None),
    tipo: str | None = Query(default=None),
):
    df = data.rows

    data.authorize_teacher(nombre_docente)
    df = aplicar_filtros(
        df,
        corte=corte,
        servicio=data.service_filter(servicio),
        tipo=tipo,
    )

    return obtener_detalle_docente(
        df,
        nombre_docente,
    )

@router.get("/reporte-ejecutivo/pdf")
def exportar_reporte_ejecutivo(
    data: ScopedData,
    corte: str | None = Query(default=None),
    servicio: str | None = Query(default=None),
    tipo: str | None = Query(default=None),
):
    df, df_cortes = data.rows, data.cuts

    df_filtrado = aplicar_filtros(
        df,
        corte=corte,
        servicio=data.service_filter(servicio),
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
            else ("Servicios asignados" if data.restricted else "Todos los servicios")
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
    data: ScopedData,
    corte: str | None = Query(default=None),
    servicio: str | None = Query(default=None),
    tipo: str | None = Query(default=None),
):
    df = data.rows

    df = aplicar_filtros(
        df,
        corte=corte,
        servicio=data.service_filter(servicio),
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
