from collections import defaultdict
from statistics import mean


def _es_si(valor) -> bool:
    return str(valor or "").strip().upper() in {
        "SÍ",
        "SI",
    }


def _numero(valor):
    if valor in ("", None):
        return None

    try:
        return float(valor)
    except (TypeError, ValueError):
        return None


def _promedio(valores):
    valores_validos = [
        valor
        for valor in valores
        if valor is not None
    ]

    if not valores_validos:
        return None

    return round(
        mean(valores_validos),
        2,
    )


def calcular_kpis_preview(
    registros: list[dict],
) -> dict:
    """
    Calcula una vista previa de KPIs.

    No escribe datos en el MASTER.
    Respeta:
    - promedio por pregunta;
    - igual peso de preguntas dentro de sección;
    - igual peso de secciones en el índice general.
    """

    filas_origen = {
        registro["FILA_ORIGEN"]
        for registro in registros
    }

    validos_kpi = [
        registro
        for registro in registros
        if _es_si(
            registro.get(
                "INCLUIR_EN_KPI",
                "",
            )
        )
        and _numero(
            registro.get(
                "INDICE_100"
            )
        )
        is not None
    ]

    por_pregunta = defaultdict(
        list
    )

    for registro in validos_kpi:
        por_pregunta[
            registro["PREGUNTA_ID"]
        ].append(
            registro
        )

    preguntas = []

    for pregunta_id, grupo in por_pregunta.items():
        indices = [
            _numero(
                registro.get(
                    "INDICE_100"
                )
            )
            for registro in grupo
        ]

        valores_base = [
            _numero(
                registro.get(
                    "VALOR_BASE"
                )
            )
            for registro in grupo
        ]

        valores_base = [
            valor
            for valor in valores_base
            if valor is not None
        ]

        n = len(indices)

        favorables = sum(
            1
            for valor in valores_base
            if valor >= 4
        )

        neutrales = sum(
            1
            for valor in valores_base
            if valor == 3
        )

        desfavorables = sum(
            1
            for valor in valores_base
            if valor <= 2
        )

        ejemplo = grupo[0]

        preguntas.append(
            {
                "pregunta_id": pregunta_id,
                "seccion_id": ejemplo.get(
                    "SECCION_ID",
                    "",
                ),
                "seccion_nombre": ejemplo.get(
                    "SECCION_NOMBRE",
                    "",
                ),
                "texto_corto": ejemplo.get(
                    "TEXTO_CORTO",
                    "",
                ),
                "tipo_dato": ejemplo.get(
                    "TIPO_DATO",
                    "",
                ),
                "indice": _promedio(
                    indices
                ),
                "n": n,
                "favorable_pct": round(
                    favorables / n * 100,
                    2,
                )
                if n
                else None,
                "neutral_pct": round(
                    neutrales / n * 100,
                    2,
                )
                if n
                else None,
                "desfavorable_pct": round(
                    desfavorables / n * 100,
                    2,
                )
                if n
                else None,
                "entra_seccion": _es_si(
                    ejemplo.get(
                        "ENTRA_PROMEDIO_SECCION",
                        "",
                    )
                ),
                "entra_general": _es_si(
                    ejemplo.get(
                        "ENTRA_PROMEDIO_GENERAL",
                        "",
                    )
                ),
            }
        )

    preguntas_seccion = defaultdict(
        list
    )

    for pregunta in preguntas:
        if (
            pregunta["entra_seccion"]
            and pregunta["indice"]
            is not None
        ):
            preguntas_seccion[
                pregunta["seccion_id"]
            ].append(
                pregunta
            )

    secciones = []

    for seccion_id, grupo in preguntas_seccion.items():
        secciones.append(
            {
                "seccion_id": seccion_id,
                "seccion_nombre": grupo[
                    0
                ][
                    "seccion_nombre"
                ],
                "indice": _promedio(
                    [
                        pregunta[
                            "indice"
                        ]
                        for pregunta in grupo
                    ]
                ),
                "preguntas_con_datos": len(
                    grupo
                ),
            }
        )

    preguntas_general = defaultdict(
        list
    )

    for pregunta in preguntas:
        if (
            pregunta["entra_general"]
            and pregunta["indice"]
            is not None
        ):
            preguntas_general[
                pregunta["seccion_id"]
            ].append(
                pregunta["indice"]
            )

    secciones_general = [
        _promedio(indices)
        for indices in preguntas_general.values()
        if indices
    ]

    indice_general = _promedio(
        secciones_general
    )

    return {
        "encuestas": len(
            filas_origen
        ),
        "respuestas_normalizadas": len(
            registros
        ),
        "respuestas_kpi_validas": len(
            validos_kpi
        ),
        "preguntas_con_datos": len(
            preguntas
        ),
        "indice_general": indice_general,
        "secciones_general": len(
            secciones_general
        ),
        "secciones": sorted(
            secciones,
            key=lambda x: x[
                "seccion_id"
            ],
        ),
        "preguntas": preguntas,
    }
