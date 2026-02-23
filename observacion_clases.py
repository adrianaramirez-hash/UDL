import streamlit as st
import pandas as pd
import gspread
import json
from collections.abc import Mapping
from google.oauth2.service_account import Credentials
import altair as alt

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]


@st.cache_data(ttl=300, show_spinner=False)
def cargar_datos_desde_sheets(_refresh_key: int = 0):
    raw = st.secrets["gcp_service_account_json"]
    creds_dict = dict(raw) if isinstance(raw, Mapping) else json.loads(raw)

    creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    client = gspread.authorize(creds)

    sheet_url = st.secrets.get("OC_SHEET_URL", "").strip()
    if not sheet_url:
        raise KeyError("Falta configurar OC_SHEET_URL en Secrets.")

    sh = client.open_by_url(sheet_url)

    ws_resp = sh.worksheet("Respuestas de formulario 1")
    df_resp = pd.DataFrame(ws_resp.get_all_records())

    ws_cortes = sh.worksheet("Cortes")
    df_cortes = pd.DataFrame(ws_cortes.get_all_records())

    return df_resp, df_cortes


def respuesta_a_puntos(valor):
    if pd.isna(valor):
        return None
    texto = str(valor).strip().lower()
    if texto in ("sí", "si", "x"):
        return 3
    if "sin evidencia" in texto or "sin evidencias" in texto:
        return 2
    if texto == "no":
        return 1
    try:
        return float(texto)
    except ValueError:
        return None


def clasificar_por_puntos(total_puntos):
    if pd.isna(total_puntos):
        return ""
    if total_puntos >= 97:
        return "Consolidado"
    if total_puntos >= 76:
        return "En proceso"
    return "No consolidado"


def asignar_corte(fecha, df_cortes):
    if pd.isna(fecha) or df_cortes.empty:
        return "Sin corte"
    for _, fila in df_cortes.iterrows():
        fi = fila.get("Fecha_inicio")
        ff = fila.get("Fecha_fin")
        if pd.notna(fi) and pd.notna(ff) and fi <= fecha <= ff:
            return str(fila.get("Corte"))
    return "Sin corte"


def obtener_texto(fila, posibles_nombres):
    for nombre in posibles_nombres:
        if nombre in fila.index:
            valor = fila[nombre]
            if isinstance(valor, str) and valor.strip():
                return valor
    return ""


def normalizar_texto(valor):
    return str(valor).strip().lower() if pd.notna(valor) else ""


def render_observacion_clases(vista: str = "Dirección General", carrera: str | None = None):
    if "oc_refresh_key" not in st.session_state:
        st.session_state.oc_refresh_key = 0

    st.sidebar.header("Filtros")
    if st.sidebar.button("Recargar datos"):
        st.session_state.oc_refresh_key += 1
        st.cache_data.clear()

    try:
        with st.spinner("Cargando datos (Google Sheets)…"):
            df_respuestas, df_cortes = cargar_datos_desde_sheets(st.session_state.oc_refresh_key)
    except Exception as e:
        st.error("No se pudieron cargar los datos desde Google Sheets.")
        st.exception(e)
        st.stop()

    if df_respuestas.empty:
        st.warning("La hoja de respuestas está vacía.")
        st.stop()

    df_respuestas.columns = [c.strip() if isinstance(c, str) else c for c in df_respuestas.columns]
    df_cortes.columns = [c.strip() if isinstance(c, str) else c for c in df_cortes.columns]

    st.subheader("Observación de clases — Reportes por corte")

    if "Fecha" not in df_respuestas.columns:
        st.error("No se encontró la columna obligatoria 'Fecha' en la hoja de respuestas.")
        st.stop()

    col_fecha = "Fecha"
    df_respuestas[col_fecha] = pd.to_datetime(df_respuestas[col_fecha], errors="coerce", dayfirst=True)

    COL_SERVICIO = "Indica el servicio"
    COL_DOCENTE = "Nombre del docente"

    for col in (COL_SERVICIO, COL_DOCENTE):
        if col not in df_respuestas.columns:
            st.error(f"No se encontró la columna '{col}' en la hoja de respuestas.")
            st.stop()

    df_respuestas["Servicio_norm"] = df_respuestas[COL_SERVICIO].apply(normalizar_texto)

    carrera_norm = None
    if vista == "Director de carrera" and carrera:
        carrera_norm = normalizar_texto(carrera)

    if not df_cortes.empty:
        if "Fecha_inicio" in df_cortes.columns:
            df_cortes["Fecha_inicio"] = pd.to_datetime(df_cortes["Fecha_inicio"], errors="coerce", dayfirst=True)
        if "Fecha_fin" in df_cortes.columns:
            df_cortes["Fecha_fin"] = pd.to_datetime(df_cortes["Fecha_fin"], errors="coerce", dayfirst=True)
        if "Fecha_inicio" in df_cortes.columns:
            df_cortes = df_cortes.sort_values("Fecha_inicio", kind="stable")
    else:
        df_cortes = pd.DataFrame(columns=["Corte", "Fecha_inicio", "Fecha_fin"])

    df_respuestas["Corte"] = df_respuestas[col_fecha].apply(lambda f: asignar_corte(f, df_cortes))

    todas_cols = list(df_respuestas.columns)
    rubrica_inicio = "El docente va acorde con el programa del curso."
    rubrica_fin = "Se usaron estrategias para mantener la atención (dinámicas, pausas activas, preguntas detonadoras)."

    if rubrica_inicio in todas_cols and rubrica_fin in todas_cols:
        i0 = todas_cols.index(rubrica_inicio)
        i1 = todas_cols.index(rubrica_fin)
        cols_puntaje = todas_cols[i0 : i1 + 1]
    else:
        start_idx = 12
        end_idx = 52
        cols_puntaje = todas_cols[start_idx:end_idx]
        if not cols_puntaje:
            st.error("No se detectaron columnas de rúbrica.")
            st.stop()

    AREAS = {
        "A. Planeación de sesión en el aula virtual": cols_puntaje[0:14],
        "B. Presentación y desarrollo de la sesión": cols_puntaje[14:30],
        "C. Dinámicas interpersonales": cols_puntaje[30:34],
        "D. Administración de la sesión": cols_puntaje[34:40],
    }

    def calcular_total_puntos_fila(row):
        total = 0
        for col in cols_puntaje:
            if col not in row.index:
                continue
            puntos = respuesta_a_puntos(row[col])
            if puntos is not None:
                total += puntos
        return total

    df_respuestas = df_respuestas.copy()
    df_respuestas["Total_puntos_observación"] = df_respuestas.apply(calcular_total_puntos_fila, axis=1)
    df_respuestas["Clasificación_observación"] = df_respuestas["Total_puntos_observación"].apply(clasificar_por_puntos)

    excluir_sin_corte = st.sidebar.checkbox("Excluir 'Sin corte'", value=True)
    excluir_fecha_invalida = st.sidebar.checkbox("Excluir fecha inválida", value=True)

    opciones_cortes = ["Todos los cortes"]
    if not df_cortes.empty and "Corte" in df_cortes.columns:
        opciones_cortes += list(df_cortes["Corte"].astype(str))
    if "Sin corte" in df_respuestas["Corte"].unique():
        opciones_cortes.append("Sin corte")

    corte_seleccionado = st.sidebar.selectbox("Corte", opciones_cortes)

    df_para_filtros = df_respuestas.copy()
    if excluir_fecha_invalida:
        df_para_filtros = df_para_filtros[df_para_filtros[col_fecha].notna()]
    if excluir_sin_corte:
        df_para_filtros = df_para_filtros[df_para_filtros["Corte"] != "Sin corte"]
    if corte_seleccionado != "Todos los cortes":
        df_para_filtros = df_para_filtros[df_para_filtros["Corte"] == corte_seleccionado]
    if carrera_norm:
        df_para_filtros = df_para_filtros[df_para_filtros["Servicio_norm"] == carrera_norm]

    servicios_base = sorted(df_para_filtros[COL_SERVICIO].dropna().unique().tolist())

    if carrera_norm:
        st.sidebar.markdown(f"**Servicio:** {carrera} (Director de carrera)")
        servicio_seleccionado = "(director)"
    else:
        servicios_disponibles = ["Todos los servicios"] + servicios_base
        servicio_seleccionado = st.sidebar.selectbox("Servicio", servicios_disponibles)

    tipo_obs_col = None
    if "Tipo de observación" in df_respuestas.columns:
        tipo_obs_col = "Tipo de observación"
    elif "Tipo de observación " in df_respuestas.columns:
        tipo_obs_col = "Tipo de observación "

    if tipo_obs_col:
        tipos_disponibles = ["Todos los tipos"] + sorted(df_para_filtros[tipo_obs_col].dropna().unique().tolist())
        tipo_seleccionado = st.sidebar.selectbox("Tipo de observación", tipos_disponibles)
    else:
        tipo_seleccionado = "Todos los tipos"

    df_filtrado = df_respuestas.copy()

    if excluir_fecha_invalida:
        df_filtrado = df_filtrado[df_filtrado[col_fecha].notna()]
    if excluir_sin_corte:
        df_filtrado = df_filtrado[df_filtrado["Corte"] != "Sin corte"]

    if corte_seleccionado != "Todos los cortes":
        df_filtrado = df_filtrado[df_filtrado["Corte"] == corte_seleccionado]

    if carrera_norm:
        df_filtrado = df_filtrado[df_filtrado["Servicio_norm"] == carrera_norm]
    elif servicio_seleccionado != "Todos los servicios":
        df_filtrado = df_filtrado[df_filtrado[COL_SERVICIO] == servicio_seleccionado]

    if tipo_seleccionado != "Todos los tipos" and tipo_obs_col:
        df_filtrado = df_filtrado[df_filtrado[tipo_obs_col] == tipo_seleccionado]

    if df_filtrado.empty:
        st.warning("No hay observaciones para el filtro seleccionado.")
        st.stop()

    with st.expander("Diagnóstico", expanded=False):
        n_total = len(df_respuestas)
        n_sin_corte = (df_respuestas["Corte"] == "Sin corte").sum()
        n_fecha_nula = df_respuestas[col_fecha].isna().sum()
        st.write(f"Total registros: {n_total}")
        st.write(f"Fecha inválida (no parseó): {n_fecha_nula}")
        st.write(f"Sin corte: {n_sin_corte}")

        if not df_cortes.empty and "Fecha_inicio" in df_cortes.columns and "Fecha_fin" in df_cortes.columns:
            min_corte = df_cortes["Fecha_inicio"].min()
            max_corte = df_cortes["Fecha_fin"].max()
            if pd.notna(min_corte) and pd.notna(max_corte):
                n_fuera = df_respuestas[
                    df_respuestas[col_fecha].notna()
                    & ((df_respuestas[col_fecha] < min_corte) | (df_respuestas[col_fecha] > max_corte))
                ].shape[0]
                st.write(f"Fuera de rango de cortes ({min_corte.date()} a {max_corte.date()}): {n_fuera}")

        cols_dbg = [c for c in [col_fecha, "Corte", COL_SERVICIO, COL_DOCENTE] if c in df_respuestas.columns]
        st.dataframe(df_respuestas[df_respuestas["Corte"] == "Sin corte"][cols_dbg].tail(10), use_container_width=True)

    rango_fechas = df_filtrado[col_fecha].agg(["min", "max"])
    st.caption(
        f"Observaciones: **{len(df_filtrado)}** | Rango: "
        f"{rango_fechas['min'].date() if pd.notna(rango_fechas['min']) else '—'} a "
        f"{rango_fechas['max'].date() if pd.notna(rango_fechas['max']) else '—'}"
    )

    df_base = df_filtrado.copy()
    total_obs = len(df_base)

    n_consol = (df_base["Clasificación_observación"] == "Consolidado").sum()
    n_proceso = (df_base["Clasificación_observación"] == "En proceso").sum()
    n_no = (df_base["Clasificación_observación"] == "No consolidado").sum()

    pct_consol = n_consol * 100 / total_obs if total_obs else 0
    pct_proceso = n_proceso * 100 / total_obs if total_obs else 0
    pct_no = n_no * 100 / total_obs if total_obs else 0

    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    col_kpi1.metric("Obs. totales", total_obs)
    col_kpi2.metric("% Consolidado", f"{pct_consol:.0f} %")
    col_kpi3.metric("% En proceso", f"{pct_proceso:.0f} %")
    col_kpi4.metric("% No consolidado", f"{pct_no:.0f} %")

    tab_resumen, tab_servicios, tab_docentes, tab_detalle = st.tabs(
        ["Resumen general", "Por servicio", "Por docente", "Detalle por docente"]
    )

    with tab_resumen:
        df_trend = df_respuestas.copy()

        if excluir_fecha_invalida:
            df_trend = df_trend[df_trend[col_fecha].notna()]
        if excluir_sin_corte:
            df_trend = df_trend[df_trend["Corte"] != "Sin corte"]

        if carrera_norm:
            df_trend = df_trend[df_trend["Servicio_norm"] == carrera_norm]
        elif servicio_seleccionado != "Todos los servicios":
            df_trend = df_trend[df_trend[COL_SERVICIO] == servicio_seleccionado]

        if tipo_seleccionado != "Todos los tipos" and tipo_obs_col:
            df_trend = df_trend[df_trend[tipo_obs_col] == tipo_seleccionado]

        df_trend = df_trend[df_trend["Corte"] != "Sin corte"]

        st.subheader("Evolución de la clasificación por corte")
        if not df_trend.empty:
            df_graf_cortes = (
                df_trend.groupby(["Corte", "Clasificación_observación"])
                .size()
                .reset_index(name="conteo")
            )
            totales_corte = df_graf_cortes.groupby("Corte")["conteo"].transform("sum")
            df_graf_cortes["porcentaje"] = df_graf_cortes["conteo"] * 100 / totales_corte

            chart_cortes = (
                alt.Chart(df_graf_cortes)
                .mark_bar()
                .encode(
                    x=alt.X("Corte:N", title="Corte"),
                    y=alt.Y("porcentaje:Q", title="Porcentaje"),
                    color=alt.Color("Clasificación_observación:N", title="Clasificación"),
                    tooltip=[
                        "Corte",
                        "Clasificación_observación",
                        alt.Tooltip("porcentaje:Q", format=".1f", title="Porcentaje (%)"),
                        "conteo",
                    ],
                )
                .properties(height=300)
            )
            st.altair_chart(chart_cortes, use_container_width=True)
        else:
            st.info("No hay información suficiente para mostrar la evolución por corte.")

    with tab_servicios:
        st.subheader("Clasificación por servicio")

        if total_obs > 0:
            df_graf = (
                df_base.groupby([COL_SERVICIO, "Clasificación_observación"])
                .size()
                .reset_index(name="conteo")
            )
            totales_serv = df_graf.groupby(COL_SERVICIO)["conteo"].transform("sum")
            df_graf["porcentaje"] = df_graf["conteo"] * 100 / totales_serv

            chart = (
                alt.Chart(df_graf)
                .mark_bar()
                .encode(
                    x=alt.X(f"{COL_SERVICIO}:N", title="Servicio"),
                    y=alt.Y("porcentaje:Q", title="Porcentaje"),
                    color=alt.Color("Clasificación_observación:N", title="Clasificación"),
                    tooltip=[
                        COL_SERVICIO,
                        "Clasificación_observación",
                        alt.Tooltip("porcentaje:Q", format=".1f", title="Porcentaje (%)"),
                        "conteo",
                    ],
                )
                .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)

        st.markdown("#### Resumen por servicio")
        resumen_servicio = (
            df_filtrado.groupby(COL_SERVICIO)
            .agg(
                Observaciones=("Total_puntos_observación", "count"),
                Docentes_observados=(COL_DOCENTE, "nunique"),
                Total_puntos=("Total_puntos_observación", "sum"),
            )
            .reset_index()
        )
        resumen_servicio["Promedio_puntos_por_obs"] = (
            resumen_servicio["Total_puntos"] / resumen_servicio["Observaciones"]
        )
        st.dataframe(resumen_servicio, use_container_width=True)

    with tab_docentes:
        st.subheader("Resumen por docente (en el filtro seleccionado)")
        resumen_docente = (
            df_filtrado.groupby(COL_DOCENTE)
            .agg(
                N_observaciones=("Total_puntos_observación", "count"),
                Total_puntos=("Total_puntos_observación", "sum"),
            )
            .reset_index()
        )
        resumen_docente["Promedio_puntos_por_obs"] = (
            resumen_docente["Total_puntos"] / resumen_docente["N_observaciones"]
        )
        resumen_docente["Clasificación_docente"] = resumen_docente["Promedio_puntos_por_obs"].apply(
            clasificar_por_puntos
        )
        cat_tipo = pd.CategoricalDtype(["Consolidado", "En proceso", "No consolidado"], ordered=True)
        resumen_docente["Clasificación_docente"] = resumen_docente["Clasificación_docente"].astype(cat_tipo)
        resumen_docente = resumen_docente.sort_values(
            ["Clasificación_docente", "Promedio_puntos_por_obs"],
            ascending=[True, False],
        )
        st.dataframe(resumen_docente, use_container_width=True)

    with tab_detalle:
        st.subheader("Historial y detalle de observaciones por docente")

        resumen_docente = (
            df_filtrado.groupby(COL_DOCENTE)
            .agg(
                N_observaciones=("Total_puntos_observación", "count"),
                Total_puntos=("Total_puntos_observación", "sum"),
            )
            .reset_index()
        )
        docentes_lista = sorted(resumen_docente[COL_DOCENTE].dropna().unique().tolist())
        docente_sel = st.selectbox("Selecciona un docente", ["(ninguno)"] + docentes_lista)

        if docente_sel != "(ninguno)":
            df_doc = df_filtrado[df_filtrado[COL_DOCENTE] == docente_sel].copy()
            df_doc = df_doc.sort_values(col_fecha)

            etiqueta_base = df_doc[col_fecha].dt.strftime("%Y-%m-%d").fillna("sin fecha")
            if "Grupo" in df_doc.columns:
                etiqueta_base = (
                    etiqueta_base
                    + " | "
                    + df_doc[COL_SERVICIO].astype(str)
                    + " | Grupo: "
                    + df_doc["Grupo"].astype(str)
                )
            else:
                etiqueta_base = etiqueta_base + " | " + df_doc[COL_SERVICIO].astype(str)

            df_doc["Etiqueta_obs"] = etiqueta_base

            cols_hist = [col_fecha, COL_SERVICIO, "Grupo", "Total_puntos_observación", "Clasificación_observación", "Corte"]
            cols_hist = [c for c in cols_hist if c in df_doc.columns]

            st.markdown(f"**Observaciones de {docente_sel} en el filtro actual:**")
            st.dataframe(df_doc[cols_hist], use_container_width=True)

            idx_sel = st.selectbox(
                "Elige una observación para ver el detalle por área",
                df_doc.index,
                format_func=lambda i: df_doc.loc[i, "Etiqueta_obs"],
            )
            fila_obs = df_doc.loc[idx_sel]

            def calcular_resumen_areas(df, columnas_area):
                puntos_totales = 0
                max_puntos = 0
                for col in columnas_area:
                    if col in df.columns:
                        serie = df[col].apply(respuesta_a_puntos)
                        puntos_totales += serie.fillna(0).sum()
                        max_puntos += 3 * serie.notna().sum()
                porcentaje = puntos_totales * 100 / max_puntos if max_puntos > 0 else None
                return puntos_totales, max_puntos, porcentaje

            resumen_areas_global = []
            for area, columnas in AREAS.items():
                p_tot, p_max, p_pct = calcular_resumen_areas(df_doc, columnas)
                resumen_areas_global.append(
                    {"Área": area, "Puntos (todas las observaciones)": p_tot, "Máx. posible": p_max, "% logro": p_pct}
                )
            df_areas_global = pd.DataFrame(resumen_areas_global)

            st.subheader("Resumen por área del docente (todas las observaciones)")
            st.dataframe(df_areas_global, use_container_width=True)

            chart_areas_global = (
                alt.Chart(df_areas_global)
                .mark_bar()
                .encode(
                    x=alt.X("Área:N", title="Área evaluada"),
                    y=alt.Y("% logro:Q", title="% de logro"),
                    tooltip=["Área", "Puntos (todas las observaciones)", "Máx. posible", "% logro"],
                )
                .properties(height=300)
            )
            st.altair_chart(chart_areas_global, use_container_width=True)

            resumen_areas_obs = []
            for area, columnas in AREAS.items():
                puntos = 0
                max_puntos = 0
                for col in columnas:
                    if col in fila_obs.index:
                        p = respuesta_a_puntos(fila_obs[col])
                        if p is not None:
                            puntos += p
                            max_puntos += 3
                porcentaje = puntos * 100 / max_puntos if max_puntos > 0 else None
                resumen_areas_obs.append({"Área": area, "Puntos": puntos, "Máx. posible": max_puntos, "% logro": porcentaje})

            df_areas_obs = pd.DataFrame(resumen_areas_obs)

            st.subheader("Detalle por área de la observación seleccionada")
            st.dataframe(df_areas_obs, use_container_width=True)

            chart_areas_obs = (
                alt.Chart(df_areas_obs)
                .mark_bar()
                .encode(
                    x=alt.X("Área:N", title="Área evaluada"),
                    y=alt.Y("% logro:Q", title="% de logro"),
                    tooltip=["Área", "Puntos", "Máx. posible", "% logro"],
                )
                .properties(height=300)
            )
            st.altair_chart(chart_areas_obs, use_container_width=True)

            st.subheader("Comentarios cualitativos de la observación seleccionada")

            fortalezas = obtener_texto(
                fila_obs,
                ["Fortalezas observadas en la sesión", "Fortalezas observadas en la sesión ", "Fortalezas"],
            )
            areas_op = obtener_texto(
                fila_obs,
                ["Áreas de oportunidad observadas en la sesión", "Areas de oportunidad observadas en la sesión", "Áreas de oportunidad"],
            )
            recom = obtener_texto(
                fila_obs,
                ["Recomendaciones generales para la mejora continua", "Recomendaciones generales"],
            )

            st.markdown("**Fortalezas observadas:**")
            st.write(fortalezas if fortalezas else "Sin registro.")

            st.markdown("**Áreas de oportunidad observadas:**")
            st.write(areas_op if areas_op else "Sin registro.")

            st.markdown("**Recomendaciones generales para la mejora continua:**")
            st.write(recom if recom else "Sin registro.")
