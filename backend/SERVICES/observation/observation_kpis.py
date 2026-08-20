import pandas as pd

from backend.SERVICES.observation.observation_scoring import (
    asignar_corte,
    clasificar_por_puntos,
    respuesta_a_puntos,
)


def preparar_observaciones(
    df_respuestas: pd.DataFrame,
    df_cortes: pd.DataFrame,
) -> pd.DataFrame:
    df = df_respuestas.copy()
    cortes = df_cortes.copy()

    df.columns = [
        c.strip() if isinstance(c, str) else c
        for c in df.columns
    ]

    cortes.columns = [
        c.strip() if isinstance(c, str) else c
        for c in cortes.columns
    ]

    # En tu Google Sheet actual la fecha viene como "Marca temporal".
    if "Fecha" in df.columns:
        col_fecha = "Fecha"
    elif "Marca temporal" in df.columns:
        col_fecha = "Marca temporal"
    else:
        raise ValueError(
            "No se encontró una columna de fecha "
            "('Fecha' o 'Marca temporal')."
        )

    df[col_fecha] = pd.to_datetime(
        df[col_fecha],
        errors="coerce",
        dayfirst=True,
    )

    if "Fecha_inicio" in cortes.columns:
        cortes["Fecha_inicio"] = pd.to_datetime(
            cortes["Fecha_inicio"],
            errors="coerce",
            dayfirst=True,
        )

    if "Fecha_fin" in cortes.columns:
        cortes["Fecha_fin"] = pd.to_datetime(
            cortes["Fecha_fin"],
            errors="coerce",
            dayfirst=True,
        )

    if "Fecha_inicio" in cortes.columns:
        cortes = cortes.sort_values(
            "Fecha_inicio",
            kind="stable",
        )

    df["Corte"] = df[col_fecha].apply(
        lambda fecha: asignar_corte(fecha, cortes)
    )

    todas_cols = list(df.columns)

    rubrica_inicio = (
        "El docente va acorde con el programa del curso."
    )

    rubrica_fin = (
        "Se usaron estrategias para mantener la atención "
        "(dinámicas, pausas activas, preguntas detonadoras)."
    )

    if (
        rubrica_inicio in todas_cols
        and rubrica_fin in todas_cols
    ):
        inicio = todas_cols.index(rubrica_inicio)
        fin = todas_cols.index(rubrica_fin)

        cols_puntaje = todas_cols[inicio : fin + 1]

    else:
        # Respaldo heredado del módulo Streamlit.
        cols_puntaje = todas_cols[12:52]

    if not cols_puntaje:
        raise ValueError(
            "No se detectaron columnas de la rúbrica."
        )

    def calcular_total(row):
        total = 0

        for columna in cols_puntaje:
            puntos = respuesta_a_puntos(row.get(columna))

            if puntos is not None:
                total += puntos

        return total

    df["Total_puntos_observación"] = df.apply(
        calcular_total,
        axis=1,
    )

    df["Clasificación_observación"] = (
        df["Total_puntos_observación"].apply(
            clasificar_por_puntos
        )
    )

    return df


def calcular_kpis(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "observaciones": 0,
            "consolidado": 0,
            "en_proceso": 0,
            "no_consolidado": 0,
            "pct_consolidado": 0,
            "pct_en_proceso": 0,
            "pct_no_consolidado": 0,
            "promedio": 0,
        }

    total = len(df)

    consolidado = int(
        (
            df["Clasificación_observación"]
            == "Consolidado"
        ).sum()
    )

    en_proceso = int(
        (
            df["Clasificación_observación"]
            == "En proceso"
        ).sum()
    )

    no_consolidado = int(
        (
            df["Clasificación_observación"]
            == "No consolidado"
        ).sum()
    )

    promedio = float(
    round(
        df["Total_puntos_observación"].mean(),
        1,
    )
)

    return {
        "observaciones": total,
        "consolidado": consolidado,
        "en_proceso": en_proceso,
        "no_consolidado": no_consolidado,
        "pct_consolidado": round(
            consolidado * 100 / total,
            1,
        ),
        "pct_en_proceso": round(
            en_proceso * 100 / total,
            1,
        ),
        "pct_no_consolidado": round(
            no_consolidado * 100 / total,
            1,
        ),
        "promedio": promedio,
    }
