import pandas as pd


def calcular_tendencia_por_corte(
    df: pd.DataFrame,
    df_cortes: pd.DataFrame,
) -> list[dict]:
    """
    Calcula la tendencia histórica por corte de 30 días.

    Devuelve una lista con:
    - corte
    - promedio
    - observaciones
    - consolidado
    - en_proceso
    - no_consolidado
    - pct_consolidado
    """

    if df.empty:
        return []

    datos = df.copy()
    cortes = df_cortes.copy()

    cortes.columns = [
        c.strip() if isinstance(c, str) else c
        for c in cortes.columns
    ]

    if "Fecha_inicio" in cortes.columns:
        cortes["Fecha_inicio"] = pd.to_datetime(
            cortes["Fecha_inicio"],
            errors="coerce",
            dayfirst=True,
        )

    if "Corte" not in datos.columns:
        raise ValueError(
            "El DataFrame no contiene la columna 'Corte'."
        )

    resumen = (
        datos[
            datos["Corte"].notna()
            & (datos["Corte"].astype(str) != "Sin corte")
        ]
        .groupby("Corte", dropna=False)
        .agg(
            observaciones=("Total_puntos_observación", "count"),
            promedio=("Total_puntos_observación", "mean"),
        )
        .reset_index()
    )

    clasificaciones = (
        datos[
            datos["Corte"].notna()
            & (datos["Corte"].astype(str) != "Sin corte")
        ]
        .groupby(
            ["Corte", "Clasificación_observación"],
            dropna=False,
        )
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    resumen = resumen.merge(
        clasificaciones,
        on="Corte",
        how="left",
    )

    for columna in [
        "Consolidado",
        "En proceso",
        "No consolidado",
    ]:
        if columna not in resumen.columns:
            resumen[columna] = 0

    resumen["pct_consolidado"] = (
        resumen["Consolidado"]
        * 100
        / resumen["observaciones"]
    )

    if (
        not cortes.empty
        and "Corte" in cortes.columns
        and "Fecha_inicio" in cortes.columns
    ):
        orden_cortes = (
            cortes[
                ["Corte", "Fecha_inicio"]
            ]
            .drop_duplicates(subset=["Corte"])
        )

        resumen = resumen.merge(
            orden_cortes,
            on="Corte",
            how="left",
        )

        resumen = resumen.sort_values(
            "Fecha_inicio",
            kind="stable",
        )

    resultado = []

    for _, fila in resumen.iterrows():
        resultado.append(
            {
                "corte": str(fila["Corte"]).strip(),
                "promedio": float(
                    round(fila["promedio"], 1)
                ),
                "observaciones": int(
                    fila["observaciones"]
                ),
                "consolidado": int(
                    fila["Consolidado"]
                ),
                "en_proceso": int(
                    fila["En proceso"]
                ),
                "no_consolidado": int(
                    fila["No consolidado"]
                ),
                "pct_consolidado": float(
                    round(fila["pct_consolidado"], 1)
                ),
            }
        )

    return resultado