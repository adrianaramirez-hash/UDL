import pandas as pd


def generar_insights_observacion(
    df: pd.DataFrame,
    tendencia: list[dict],
) -> dict:
    """
    Genera una lectura ejecutiva del módulo de Observación de Clases.

    Devuelve:
    - resumen
    - fortalezas
    - alertas
    - recomendaciones
    """

    if df.empty:
        return {
            "resumen": (
                "No hay información disponible para los filtros seleccionados."
            ),
            "fortalezas": [],
            "alertas": [],
            "recomendaciones": [],
        }

    fortalezas: list[str] = []
    alertas: list[str] = []
    recomendaciones: list[str] = []

    total_observaciones = len(df)

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

    promedio_actual = float(
        round(
            df["Total_puntos_observación"].mean(),
            1,
        )
    )

    pct_consolidado = (
        consolidado * 100 / total_observaciones
        if total_observaciones
        else 0
    )

    pct_en_proceso = (
        en_proceso * 100 / total_observaciones
        if total_observaciones
        else 0
    )

    pct_no_consolidado = (
        no_consolidado * 100 / total_observaciones
        if total_observaciones
        else 0
    )

    # -------------------------------------------------
    # Comparativo temporal
    # -------------------------------------------------

    cambio_promedio = None
    cambio_consolidado = None

    if len(tendencia) >= 2:
        corte_actual = tendencia[-1]
        corte_anterior = tendencia[-2]

        cambio_promedio = round(
            corte_actual["promedio"]
            - corte_anterior["promedio"],
            1,
        )

        cambio_consolidado = round(
            corte_actual["pct_consolidado"]
            - corte_anterior["pct_consolidado"],
            1,
        )

        if cambio_promedio > 0:
            fortalezas.append(
                f"El promedio del último corte mejoró "
                f"{cambio_promedio} puntos respecto al corte anterior."
            )

        elif cambio_promedio < 0:
            alertas.append(
                f"El promedio del último corte disminuyó "
                f"{abs(cambio_promedio)} puntos respecto al corte anterior."
            )

        if cambio_consolidado > 0:
            fortalezas.append(
                f"La proporción de observaciones consolidadas aumentó "
                f"{cambio_consolidado} puntos porcentuales."
            )

        elif cambio_consolidado < 0:
            alertas.append(
                f"La proporción de observaciones consolidadas disminuyó "
                f"{abs(cambio_consolidado)} puntos porcentuales."
            )

    # -------------------------------------------------
    # Lectura del desempeño
    # -------------------------------------------------

    if pct_consolidado >= 85:
        fortalezas.append(
            f"El desempeño general es favorable: "
            f"{round(pct_consolidado, 1)}% de las observaciones "
            f"se encuentra en nivel Consolidado."
        )

    elif pct_consolidado >= 70:
        fortalezas.append(
            f"La mayoría de las observaciones presenta un desempeño "
            f"aceptable, con {round(pct_consolidado, 1)}% "
            f"en nivel Consolidado."
        )

    else:
        alertas.append(
            f"La proporción de observaciones consolidadas es de "
            f"{round(pct_consolidado, 1)}%, por lo que conviene reforzar "
            f"el seguimiento académico."
        )

    if no_consolidado == 0:
        fortalezas.append(
            "No se detectaron observaciones No consolidado "
            "en el conjunto analizado."
        )

    else:
        termino_observacion = (
            "observación"
            if no_consolidado == 1
            else "observaciones"
        )

        verbo_detectar = (
            "Se detectó"
            if no_consolidado == 1
            else "Se detectaron"
        )

        alertas.append(
            f"{verbo_detectar} {no_consolidado} "
            f"{termino_observacion} No consolidado "
            f"({round(pct_no_consolidado, 1)}%)."
        )

        recomendaciones.append(
            "Priorizar la revisión de los casos No consolidado "
            "antes del cierre del siguiente corte."
        )

    if en_proceso > 0:
        termino_observacion = (
            "observación"
            if en_proceso == 1
            else "observaciones"
        )

        verbo_existir = (
            "Existe"
            if en_proceso == 1
            else "Existen"
        )

        alertas.append(
            f"{verbo_existir} {en_proceso} "
            f"{termino_observacion} En proceso "
            f"({round(pct_en_proceso, 1)}%)."
        )

        recomendaciones.append(
            "Mantener seguimiento sobre los casos En proceso "
            "para verificar su evolución en el siguiente corte de 30 días."
        )

    # -------------------------------------------------
    # Concentración de riesgo por servicio
    # -------------------------------------------------

    servicio_principal = None
    cantidad_servicio = 0

    if "Indica el servicio" in df.columns:
        df_riesgo = df[
            df["Clasificación_observación"].isin(
                ["En proceso", "No consolidado"]
            )
        ]

        if not df_riesgo.empty:
            servicios_riesgo = (
                df_riesgo["Indica el servicio"]
                .astype(str)
                .str.strip()
                .value_counts()
            )

            if not servicios_riesgo.empty:
                servicio_principal = servicios_riesgo.index[0]
                cantidad_servicio = int(
                    servicios_riesgo.iloc[0]
                )

                termino_observacion = (
                    "observación"
                    if cantidad_servicio == 1
                    else "observaciones"
                )

                verbo_requerir = (
                    "requiere"
                    if cantidad_servicio == 1
                    else "requieren"
                )

                alertas.append(
                    f"{servicio_principal} concentra la mayor cantidad "
                    f"de {termino_observacion} que {verbo_requerir} "
                    f"seguimiento ({cantidad_servicio})."
                )

                recomendaciones.append(
                    f"Revisar primero los casos de {servicio_principal}, "
                    f"ya que concentra la mayor cantidad de observaciones "
                    f"en seguimiento."
                )

    # -------------------------------------------------
    # Resumen ejecutivo
    # -------------------------------------------------

    if pct_consolidado >= 85:
        tono = "favorable"

    elif pct_consolidado >= 70:
        tono = "aceptable, aunque con oportunidades de mejora"

    else:
        tono = "con áreas que requieren atención"

    partes_resumen = [
        (
            f"Se analizaron {total_observaciones} observaciones "
            f"con un promedio de {promedio_actual} puntos."
        ),
        (
            f"El desempeño general es {tono}: "
            f"{round(pct_consolidado, 1)}% se encuentra "
            f"en nivel Consolidado."
        ),
    ]

    # -------------------------------------------------
    # Lectura temporal integrada
    # -------------------------------------------------

    if (
        cambio_promedio is not None
        and cambio_consolidado is not None
    ):
        if cambio_promedio > 0 and cambio_consolidado > 0:
            partes_resumen.append(
                f"El último corte muestra una mejora integral: "
                f"el promedio aumentó {cambio_promedio} puntos "
                f"y la proporción de observaciones consolidadas creció "
                f"{cambio_consolidado} puntos porcentuales."
            )

        elif cambio_promedio > 0 and cambio_consolidado < 0:
            partes_resumen.append(
                f"El promedio del último corte mejoró "
                f"{cambio_promedio} puntos; sin embargo, "
                f"la proporción de observaciones consolidadas disminuyó "
                f"{abs(cambio_consolidado)} puntos porcentuales."
            )

        elif cambio_promedio < 0 and cambio_consolidado > 0:
            partes_resumen.append(
                f"El promedio del último corte disminuyó "
                f"{abs(cambio_promedio)} puntos; no obstante, "
                f"la proporción de observaciones consolidadas aumentó "
                f"{cambio_consolidado} puntos porcentuales."
            )

        elif cambio_promedio < 0 and cambio_consolidado < 0:
            partes_resumen.append(
                f"El último corte presenta una señal de atención: "
                f"el promedio disminuyó {abs(cambio_promedio)} puntos "
                f"y la proporción de observaciones consolidadas bajó "
                f"{abs(cambio_consolidado)} puntos porcentuales."
            )

        elif cambio_promedio == 0 and cambio_consolidado > 0:
            partes_resumen.append(
                f"El promedio se mantiene estable; sin embargo, "
                f"la proporción de observaciones consolidadas aumentó "
                f"{cambio_consolidado} puntos porcentuales."
            )

        elif cambio_promedio == 0 and cambio_consolidado < 0:
            partes_resumen.append(
                f"El promedio se mantiene estable; sin embargo, "
                f"la proporción de observaciones consolidadas disminuyó "
                f"{abs(cambio_consolidado)} puntos porcentuales."
            )

        elif cambio_promedio > 0 and cambio_consolidado == 0:
            partes_resumen.append(
                f"El promedio del último corte mejoró "
                f"{cambio_promedio} puntos, mientras que la proporción "
                f"de observaciones consolidadas se mantuvo estable."
            )

        elif cambio_promedio < 0 and cambio_consolidado == 0:
            partes_resumen.append(
                f"El promedio del último corte disminuyó "
                f"{abs(cambio_promedio)} puntos, mientras que la proporción "
                f"de observaciones consolidadas se mantuvo estable."
            )

        else:
            partes_resumen.append(
                "El último corte se mantiene sin variaciones relevantes "
                "respecto al anterior."
            )

    elif cambio_promedio is not None:
        if cambio_promedio > 0:
            partes_resumen.append(
                f"El promedio del último corte mejoró "
                f"{cambio_promedio} puntos frente al anterior."
            )

        elif cambio_promedio < 0:
            partes_resumen.append(
                f"El promedio del último corte disminuyó "
                f"{abs(cambio_promedio)} puntos frente al anterior."
            )

        else:
            partes_resumen.append(
                "El promedio del último corte se mantiene estable "
                "respecto al anterior."
            )

    # -------------------------------------------------
    # Prioridades
    # -------------------------------------------------

    if no_consolidado > 0:
        termino_observacion = (
            "observación"
            if no_consolidado == 1
            else "observaciones"
        )

        verbo_identificar = (
            "Se identificó"
            if no_consolidado == 1
            else "Se identificaron"
        )

        verbo_requerir = (
            "requiere"
            if no_consolidado == 1
            else "requieren"
        )

        partes_resumen.append(
            f"{verbo_identificar} {no_consolidado} "
            f"{termino_observacion} No consolidado que "
            f"{verbo_requerir} atención prioritaria."
        )

    elif en_proceso > 0:
        termino_observacion = (
            "observación"
            if en_proceso == 1
            else "observaciones"
        )

        partes_resumen.append(
            f"No hay casos No consolidado, aunque permanecen "
            f"{en_proceso} {termino_observacion} En proceso."
        )

    else:
        partes_resumen.append(
            "No se identificaron observaciones que requieran "
            "seguimiento prioritario."
        )

    if servicio_principal:
        partes_resumen.append(
            f"La mayor concentración de seguimiento se encuentra en "
            f"{servicio_principal}."
        )

    resumen = " ".join(partes_resumen)

    # -------------------------------------------------
    # Evitar duplicados
    # -------------------------------------------------

    fortalezas = list(dict.fromkeys(fortalezas))
    alertas = list(dict.fromkeys(alertas))
    recomendaciones = list(dict.fromkeys(recomendaciones))

    return {
        "resumen": resumen,
        "fortalezas": fortalezas,
        "alertas": alertas,
        "recomendaciones": recomendaciones,
    }