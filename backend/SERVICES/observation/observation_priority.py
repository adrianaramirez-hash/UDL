import pandas as pd


def obtener_casos_prioritarios(
    df: pd.DataFrame,
) -> list[dict]:
    """
    Devuelve docentes que requieren seguimiento.

    Criterios:
    - En proceso
    - No consolidado

    La clasificación del docente se calcula usando
    el promedio de puntos de sus observaciones.
    """

    if df.empty:
        return []

    if "Nombre del docente" not in df.columns:
        raise ValueError(
            "No se encontró la columna 'Nombre del docente'."
        )

    resumen = (
        df.groupby("Nombre del docente")
        .agg(
            observaciones=("Total_puntos_observación", "count"),
            promedio=("Total_puntos_observación", "mean"),
        )
        .reset_index()
    )

    def clasificar(promedio: float) -> str:
        if promedio >= 97:
            return "Consolidado"

        if promedio >= 76:
            return "En proceso"

        return "No consolidado"

    resumen["clasificacion"] = resumen["promedio"].apply(
        clasificar
    )

    resumen = resumen[
        resumen["clasificacion"].isin(
            ["En proceso", "No consolidado"]
        )
    ]

    resultado = []

    for _, fila in resumen.iterrows():
        docente = fila["Nombre del docente"]

        registros_docente = df[
            df["Nombre del docente"] == docente
        ]

        servicio = ""

        if "Indica el servicio" in registros_docente.columns:
            valores = (
                registros_docente["Indica el servicio"]
                .dropna()
                .astype(str)
                .str.strip()
            )

            if not valores.empty:
                servicio = valores.iloc[-1]

        tipo = ""

        tipo_col = None

        if "Tipo de observación" in registros_docente.columns:
            tipo_col = "Tipo de observación"

        elif "Tipo de observación " in registros_docente.columns:
            tipo_col = "Tipo de observación "

        if tipo_col:
            valores = (
                registros_docente[tipo_col]
                .dropna()
                .astype(str)
                .str.strip()
            )

            if not valores.empty:
                tipo = valores.iloc[-1]

        resultado.append(
            {
                "docente": str(docente),
                "servicio": servicio,
                "tipo": tipo,
                "promedio": float(
                    round(fila["promedio"], 1)
                ),
                "observaciones": int(
                    fila["observaciones"]
                ),
                "clasificacion": str(
                    fila["clasificacion"]
                ),
            }
        )

    resultado.sort(
        key=lambda item: item["promedio"]
    )

    return resultado