import pandas as pd


def obtener_detalle_docente(
    df: pd.DataFrame,
    nombre_docente: str,
) -> dict:
    """
    Devuelve el detalle histórico de un docente dentro del
    conjunto ya filtrado de Observación de Clases.
    """

    if df.empty:
        return {
            "docente": nombre_docente,
            "observaciones": 0,
            "promedio": 0,
            "clasificacion": "",
            "historial": [],
            "fortalezas": [],
            "areas_oportunidad": [],
            "recomendaciones": [],
        }

    if "Nombre del docente" not in df.columns:
        raise ValueError(
            "No se encontró la columna 'Nombre del docente'."
        )

    nombre_busqueda = nombre_docente.strip()

    df_docente = df[
        df["Nombre del docente"]
        .astype(str)
        .str.strip()
        == nombre_busqueda
    ].copy()

    if df_docente.empty:
        return {
            "docente": nombre_busqueda,
            "observaciones": 0,
            "promedio": 0,
            "clasificacion": "",
            "historial": [],
            "fortalezas": [],
            "areas_oportunidad": [],
            "recomendaciones": [],
        }

    promedio = float(
        round(
            df_docente["Total_puntos_observación"].mean(),
            1,
        )
    )

    if promedio >= 97:
        clasificacion = "Consolidado"
    elif promedio >= 76:
        clasificacion = "En proceso"
    else:
        clasificacion = "No consolidado"

    historial = []

    tipo_col = None

    if "Tipo de observación" in df_docente.columns:
        tipo_col = "Tipo de observación"
    elif "Tipo de observación " in df_docente.columns:
        tipo_col = "Tipo de observación "

    for _, fila in df_docente.iterrows():
        historial.append(
            {
                "corte": str(
                    fila.get("Corte", "")
                ).strip(),
                "servicio": str(
                    fila.get("Indica el servicio", "")
                ).strip(),
                "grupo": str(
                    fila.get("Grupo", "")
                ).strip(),
                "asignatura": str(
                    fila.get("Asignatura", "")
                ).strip(),
                "tipo": (
                    str(
                        fila.get(tipo_col, "")
                    ).strip()
                    if tipo_col
                    else ""
                ),
                "puntaje": float(
                    fila.get(
                        "Total_puntos_observación",
                        0,
                    )
                ),
                "clasificacion": str(
                    fila.get(
                        "Clasificación_observación",
                        "",
                    )
                ).strip(),
            }
        )

    def obtener_textos(
        posibles_columnas: list[str],
    ) -> list[str]:
        textos = []

        for columna in posibles_columnas:
            if columna in df_docente.columns:
                valores = (
                    df_docente[columna]
                    .dropna()
                    .astype(str)
                    .str.strip()
                )

                textos.extend(
                    [
                        texto
                        for texto in valores.tolist()
                        if texto
                    ]
                )

        return list(dict.fromkeys(textos))

    fortalezas = obtener_textos(
        [
            "Fortalezas observadas en la sesión",
            "Fortalezas observadas en la sesión ",
            "Fortalezas",
        ]
    )

    areas_oportunidad = obtener_textos(
        [
            "Áreas de oportunidad observadas en la sesión",
            "Areas de oportunidad observadas en la sesión",
            "Áreas de oportunidad",
        ]
    )

    recomendaciones = obtener_textos(
        [
            "Recomendaciones generales para la mejora continua",
            "Recomendaciones generales",
        ]
    )

    return {
        "docente": nombre_busqueda,
        "observaciones": int(len(df_docente)),
        "promedio": promedio,
        "clasificacion": clasificacion,
        "historial": historial,
        "fortalezas": fortalezas,
        "areas_oportunidad": areas_oportunidad,
        "recomendaciones": recomendaciones,
    }