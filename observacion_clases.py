def render_observacion_clases(vista: str = "Dirección General", carrera: str | None = None):

    # --------------------------------------------------
    # CARGA DE DATOS
    # --------------------------------------------------
    try:
        with st.spinner("Cargando datos (Google Sheets)…"):
            df_respuestas, df_cortes = cargar_datos_desde_sheets()
    except Exception as e:
        st.error("No se pudieron cargar los datos desde Google Sheets.")
        st.exception(e)
        st.stop()

    if df_respuestas.empty:
        st.warning("La hoja de respuestas está vacía.")
        st.stop()

    st.subheader("Observación de clases — Reportes por corte")

    # --------------------------------------------------
    # FECHA OFICIAL (SIEMPRE COLUMNA 'Fecha')
    # --------------------------------------------------
    if "Fecha" not in df_respuestas.columns:
        st.error("No se encontró la columna obligatoria 'Fecha' en la hoja.")
        st.stop()

    df_respuestas["Fecha_dt"] = pd.to_datetime(
        df_respuestas["Fecha"],
        errors="coerce",
        dayfirst=True
    )

    col_fecha = "Fecha_dt"

    # --------------------------------------------------
    # VALIDACIÓN DE FECHAS
    # --------------------------------------------------
    if df_respuestas["Fecha_dt"].isna().all():
        st.error("No se pudieron convertir las fechas. Verifica el formato de la columna 'Fecha'.")
        st.stop()

    # --------------------------------------------------
    # NORMALIZACIÓN
    # --------------------------------------------------
    COL_SERVICIO = "Indica el servicio"
    COL_DOCENTE = "Nombre del docente"

    for col in [COL_SERVICIO, COL_DOCENTE]:
        if col not in df_respuestas.columns:
            st.error(f"No se encontró la columna '{col}'.")
            st.stop()

    df_respuestas["Servicio_norm"] = df_respuestas[COL_SERVICIO].astype(str).str.strip().str.lower()

    carrera_norm = None
    if vista == "Director de carrera" and carrera:
        carrera_norm = carrera.strip().lower()

    # --------------------------------------------------
    # CORTES
    # --------------------------------------------------
    if not df_cortes.empty:
        df_cortes["Fecha_inicio"] = pd.to_datetime(df_cortes["Fecha_inicio"], errors="coerce", dayfirst=True)
        df_cortes["Fecha_fin"] = pd.to_datetime(df_cortes["Fecha_fin"], errors="coerce", dayfirst=True)

    def asignar_corte(fecha):
        if pd.isna(fecha) or df_cortes.empty:
            return "Sin corte"
        for _, fila in df_cortes.iterrows():
            if pd.notna(fila["Fecha_inicio"]) and pd.notna(fila["Fecha_fin"]):
                if fila["Fecha_inicio"] <= fecha <= fila["Fecha_fin"]:
                    return str(fila["Corte"])
        return "Sin corte"

    df_respuestas["Corte"] = df_respuestas[col_fecha].apply(asignar_corte)

    # --------------------------------------------------
    # DETECCIÓN DE RÚBRICA POR NOMBRE
    # --------------------------------------------------
    todas_cols = list(df_respuestas.columns)

    rubrica_inicio = "El docente va acorde con el programa del curso."
    rubrica_fin = "Se usaron estrategias para mantener la atención (dinámicas, pausas activas, preguntas detonadoras)."

    if rubrica_inicio in todas_cols and rubrica_fin in todas_cols:
        i0 = todas_cols.index(rubrica_inicio)
        i1 = todas_cols.index(rubrica_fin)
        cols_puntaje = todas_cols[i0:i1 + 1]
    else:
        st.error("No se pudieron detectar correctamente las columnas de rúbrica.")
        st.stop()

    # --------------------------------------------------
    # CÁLCULO DE PUNTOS
    # --------------------------------------------------
    def respuesta_a_puntos(valor):
        if pd.isna(valor):
            return None
        texto = str(valor).strip().lower()
        if texto in ["sí", "si", "x"]:
            return 3
        if "sin evidencia" in texto:
            return 2
        if texto == "no":
            return 1
        try:
            return float(texto)
        except:
            return None

    def clasificar(total):
        if pd.isna(total):
            return ""
        if total >= 97:
            return "Consolidado"
        elif total >= 76:
            return "En proceso"
        else:
            return "No consolidado"

    def calcular_total(row):
        total = 0
        for col in cols_puntaje:
            p = respuesta_a_puntos(row[col])
            if p is not None:
                total += p
        return total

    df_respuestas["Total_puntos_observación"] = df_respuestas.apply(calcular_total, axis=1)
    df_respuestas["Clasificación_observación"] = df_respuestas["Total_puntos_observación"].apply(clasificar)

    # ==================================================
    # SIDEBAR - FILTROS
    # ==================================================
    st.sidebar.header("Filtros")

    opciones_cortes = ["Todos los cortes"] + sorted(df_respuestas["Corte"].unique().tolist())
    corte_sel = st.sidebar.selectbox("Corte", opciones_cortes)

    df_filtrado = df_respuestas.copy()

    if corte_sel != "Todos los cortes":
        df_filtrado = df_filtrado[df_filtrado["Corte"] == corte_sel]

    if carrera_norm:
        df_filtrado = df_filtrado[df_filtrado["Servicio_norm"] == carrera_norm]
        st.sidebar.markdown(f"**Servicio:** {carrera}")
    else:
        servicios = ["Todos los servicios"] + sorted(df_filtrado[COL_SERVICIO].dropna().unique().tolist())
        servicio_sel = st.sidebar.selectbox("Servicio", servicios)

        if servicio_sel != "Todos los servicios":
            df_filtrado = df_filtrado[df_filtrado[COL_SERVICIO] == servicio_sel]

    if "Tipo de observación" in df_filtrado.columns:
        tipos = ["Todos"] + sorted(df_filtrado["Tipo de observación"].dropna().unique().tolist())
        tipo_sel = st.sidebar.selectbox("Tipo de observación", tipos)

        if tipo_sel != "Todos":
            df_filtrado = df_filtrado[df_filtrado["Tipo de observación"] == tipo_sel]

    if df_filtrado.empty:
        st.warning("No hay observaciones con el filtro seleccionado.")
        st.stop()

    # ==================================================
    # KPIs
    # ==================================================
    total = len(df_filtrado)

    n_consol = (df_filtrado["Clasificación_observación"] == "Consolidado").sum()
    n_proc = (df_filtrado["Clasificación_observación"] == "En proceso").sum()
    n_no = (df_filtrado["Clasificación_observación"] == "No consolidado").sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Observaciones", total)
    col2.metric("% Consolidado", f"{(n_consol*100/total):.0f}%")
    col3.metric("% En proceso", f"{(n_proc*100/total):.0f}%")
    col4.metric("% No consolidado", f"{(n_no*100/total):.0f}%")

    st.divider()

    # ==================================================
    # TABLA FINAL
    # ==================================================
    columnas_mostrar = [
        col_fecha,
        COL_SERVICIO,
        COL_DOCENTE,
        "Total_puntos_observación",
        "Clasificación_observación",
        "Corte"
    ]

    columnas_mostrar = [c for c in columnas_mostrar if c in df_filtrado.columns]

    st.dataframe(
        df_filtrado.sort_values(col_fecha, ascending=False)[columnas_mostrar],
        use_container_width=True
    )
