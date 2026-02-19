import pandas as pd
import streamlit as st
import altair as alt
import gspread
from io import BytesIO

SECTION_LABELS = {
    "DIR": "Director / Coordinación",
    "SER": "Servicios administrativos y generales",
    "ADM": "Acceso a soporte administrativo",
    "ACD": "Servicios académicos",
    "APR": "Aprendizaje",
    "EVA": "Evaluación del conocimiento",
    "SEAC": "Plataforma SEAC",
    "PLAT": "Plataforma SEAC",
    "SAT": "Plataforma SEAC",
    "MAT": "Materiales en la plataforma",
    "UDL": "Comunicación con la Universidad",
    "COM": "Comunicación con compañeros",
    "INS": "Instalaciones y equipo tecnológico",
    "AMB": "Ambiente escolar",
    "REC": "Recomendación y satisfacción",
    "OTR": "Otros",
}

SHEET_PROCESADO = "PROCESADO"
SHEET_MAPA = "MAPA_PREGUNTAS"

FINANZAS_SHEET_ID = "11qszwEcEA6vvy7XYGo-w_WkqPp1kxoNG5GfJB_Wcc4A"
FINANZAS_SHEET_NAME = "VISTA_FINANZAS_NUM"


def _to_datetime_safe(s):
    return pd.to_datetime(s, errors="coerce", dayfirst=True)


def _pick_fecha_col(df):
    for c in ["Marca temporal", "Marca Temporal", "Fecha", "fecha"]:
        if c in df.columns:
            return c
    return None


def _mean_numeric(series):
    return pd.to_numeric(series, errors="coerce").mean()


def _auto_classify_numcols(df, cols):
    if not cols:
        return [], []
    dnum = df[cols].apply(pd.to_numeric, errors="coerce")
    maxs = dnum.max(axis=0, skipna=True)
    likert = [c for c in cols if c in maxs.index and pd.notna(maxs[c]) and float(maxs[c]) > 1]
    yesno = [c for c in cols if c not in likert]
    return likert, yesno


def _section_from_numcol(col):
    return col.split("_", 1)[0]


def _create_excel_download(df, kpis_df, filename):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="DATA_FILTRADA", index=False)
        if kpis_df is not None:
            kpis_df.to_excel(writer, sheet_name="KPIS", index=False)
    output.seek(0)

    st.download_button(
        "⬇️ DESCARGAR REPORTE COMPLETO (EXCEL)",
        data=output,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )


def _download_buttons(df, filename_prefix, kpis_df=None):

    if df.empty:
        return

    st.markdown("### 📥 Exportación de información")

    col1, col2 = st.columns(2)

    with col1:
        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Descargar base completa (CSV)",
            data=csv,
            file_name=f"{filename_prefix}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col2:
        _create_excel_download(
            df,
            kpis_df,
            f"{filename_prefix}.xlsx"
        )


@st.cache_data(ttl=300)
def _load_general(url):
    sa = dict(st.secrets["gcp_service_account_json"])
    gc = gspread.service_account_from_dict(sa)
    sh = gc.open_by_url(url)

    def ws_df(name):
        ws = sh.worksheet(name)
        data = ws.get_all_values()
        return pd.DataFrame(data[1:], columns=data[0]).replace("", pd.NA)

    return ws_df(SHEET_PROCESADO), ws_df(SHEET_MAPA)


@st.cache_data(ttl=300)
def _load_finanzas():
    sa = dict(st.secrets["gcp_service_account_json"])
    gc = gspread.service_account_from_dict(sa)
    sh = gc.open_by_key(FINANZAS_SHEET_ID)
    ws = sh.worksheet(FINANZAS_SHEET_NAME)
    data = ws.get_all_values()
    return pd.DataFrame(data[1:], columns=data[0]).replace("", pd.NA)
def render_encuesta_calidad(vista=None, carrera=None):

    st.subheader("Encuesta de calidad")
    vista = (vista or "Dirección General").strip()

    # =====================================================
    # ================= VISTA FINANZAS ====================
    # =====================================================

    if vista == "Dirección Finanzas":

        df = _load_finanzas()

        if df.empty:
            st.warning("Sin datos disponibles.")
            return

        fecha_col = _pick_fecha_col(df)
        years = ["(Todos)"]

        if fecha_col:
            df[fecha_col] = _to_datetime_safe(df[fecha_col])
            years += sorted(df[fecha_col].dt.year.dropna().unique(), reverse=True)

        with st.sidebar:
            year_sel = st.selectbox("Año", years)

        f = df.copy()

        if year_sel != "(Todos)" and fecha_col:
            f = f[f[fecha_col].dt.year == int(year_sel)]

        kpis = pd.DataFrame({
            "Indicador": ["Total registros"],
            "Valor": [len(f)]
        })

        _download_buttons(f, f"finanzas_{year_sel}", kpis)

        st.dataframe(f, use_container_width=True)
        return

    # =====================================================
    # ================= VISTA GENERAL =====================
    # =====================================================

    modalidad = "Escolarizado / Ejecutivas"

    if vista == "Dirección General":
        modalidad = st.sidebar.selectbox(
            "Modalidad",
            ["Virtual / Mixto", "Escolarizado / Ejecutivas", "Preparatoria"]
        )

    url = st.secrets.get({
        "Virtual / Mixto": "EC_VIRTUAL_URL",
        "Escolarizado / Ejecutivas": "EC_ESCOLAR_URL",
        "Preparatoria": "EC_PREPA_URL"
    }[modalidad])

    df, mapa = _load_general(url)

    if df.empty:
        st.warning("Sin datos disponibles.")
        return

    fecha_col = _pick_fecha_col(df)
    years = ["(Todos)"]

    if fecha_col:
        df[fecha_col] = _to_datetime_safe(df[fecha_col])
        years += sorted(df[fecha_col].dt.year.dropna().unique(), reverse=True)

    carrera_col = None
    for c in ["Carrera_Catalogo", "Servicio", "Programa", "Carrera"]:
        if c in df.columns:
            carrera_col = c
            break

    with st.sidebar:
        year_sel = st.selectbox("Año", years)

        if vista == "Dirección General" and carrera_col:
            carrera_sel = st.selectbox(
                "Carrera",
                ["(Todas)"] + sorted(df[carrera_col].dropna().unique())
            )
        else:
            carrera_sel = carrera

    f = df.copy()

    if year_sel != "(Todos)" and fecha_col:
        f = f[f[fecha_col].dt.year == int(year_sel)]

    if carrera_sel and carrera_sel != "(Todas)" and carrera_col:
        f = f[f[carrera_col] == carrera_sel]

    if f.empty:
        st.warning("No hay datos con los filtros seleccionados.")
        return

    # =====================================================
    # NORMALIZAR MAPA (LAB O NUEVO)
    # =====================================================

    if "header_num" not in mapa.columns:

        if {"header_raw", "header_id", "tipo"}.issubset(set(mapa.columns)):

            mapa = mapa.copy()

            mapa["header_num"] = (
                mapa["header_id"].astype(str).str.strip() +
                mapa["tipo"].astype(str).str.upper().map({
                    "ABIERTA": "_txt"
                }).fillna("_num")
            )
        else:
            st.error("Mapa_Preguntas no tiene estructura válida.")
            st.stop()

    mapa["section_code"] = mapa["header_num"].apply(_section_from_numcol)

    mapa["section_name"] = (
        mapa["section_code"]
        .map(SECTION_LABELS)
        .fillna(mapa["section_code"])
        .astype(str)
    )

    # =====================================================
    # KPIs
    # =====================================================

    numeric_cols = [c for c in f.columns if c.endswith("_num")]
    likert_cols, yesno_cols = _auto_classify_numcols(f, numeric_cols)

    promedio_global = None
    if likert_cols:
        promedio_global = round(pd.to_numeric(f[likert_cols].stack(), errors="coerce").mean(), 2)

    pct_si = None
    if yesno_cols:
        pct_si = round(pd.to_numeric(f[yesno_cols].stack(), errors="coerce").mean() * 100, 1)

    kpis = pd.DataFrame({
        "Indicador": ["Total respuestas", "Promedio global Likert", "% Sí (Sí/No)"],
        "Valor": [len(f), promedio_global, pct_si]
    })

    _download_buttons(
        f,
        f"encuesta_{modalidad}_{year_sel}",
        kpis
    )

    # =====================================================
    # TABS
    # =====================================================

    if vista == "Dirección General":
        tab1, tab2, tab4, tab3 = st.tabs(
            ["Resumen", "Por sección", "Comparativo entre carreras", "Comentarios"]
        )
    else:
        tab1, tab2, tab3 = st.tabs(["Resumen", "Por sección", "Comentarios"])
        tab4 = None

    # =====================================================
    # RESUMEN + GRÁFICA POR SECCIÓN
    # =====================================================

    with tab1:

        c1, c2, c3 = st.columns(3)

        c1.metric("Respuestas", len(f))

        if promedio_global is not None:
            c2.metric("Promedio global", promedio_global)

        if pct_si is not None:
            c3.metric("% Sí", f"{pct_si}%")

        rows = []

        for (sec_code, sec_name), g in mapa.groupby(["section_code", "section_name"]):

            cols = [c for c in g["header_num"] if c in likert_cols]

            if not cols:
                continue

            val = pd.to_numeric(f[cols].stack(), errors="coerce").mean()

            if pd.notna(val):
                rows.append({
                    "Sección": sec_name,
                    "Promedio": round(val, 2)
                })

        if rows:
            sec_df = pd.DataFrame(rows).sort_values("Promedio", ascending=False)

            st.markdown("### Promedio por sección")
            st.dataframe(sec_df, use_container_width=True)

            chart = (
                alt.Chart(sec_df)
                .mark_bar()
                .encode(
                    x=alt.X("Sección:N", sort="-y"),
                    y="Promedio:Q",
                    tooltip=["Sección", "Promedio"]
                )
                .properties(height=350)
            )

            st.altair_chart(chart, use_container_width=True)

    # =====================================================
    # DETALLE POR SECCIÓN
    # =====================================================

    with tab2:

        for sec_code, sec_name in mapa[["section_code", "section_name"]].drop_duplicates().values:

            with st.expander(str(sec_name)):

                cols = mapa[
                    (mapa["section_code"] == sec_code) &
                    (mapa["header_num"].isin(likert_cols))
                ]["header_num"].tolist()

                if not cols:
                    st.info("Sin datos.")
                    continue

                rows = []

                for col in cols:
                    val = _mean_numeric(f[col])
                    if pd.notna(val):
                        rows.append({
                            "Pregunta": col,
                            "Promedio": round(val, 2)
                        })

                if rows:
                    qdf = pd.DataFrame(rows).sort_values("Promedio", ascending=False)
                    st.dataframe(qdf, use_container_width=True)

    # =====================================================
    # COMPARATIVO DG
    # =====================================================

    if tab4:

        with tab4:

            if not carrera_col:
                st.info("No hay columna carrera.")
            else:

                for (sec_code, sec_name), g in mapa.groupby(["section_code", "section_name"]):

                    cols = [c for c in g["header_num"] if c in likert_cols]

                    if not cols:
                        continue

                    rows = []

                    for carrera_val, df_c in f.groupby(carrera_col):

                        val = pd.to_numeric(df_c[cols].stack(), errors="coerce").mean()

                        if pd.notna(val):
                            rows.append({
                                "Carrera": carrera_val,
                                "Promedio": round(val, 2)
                            })

                    if rows:
                        comp_df = pd.DataFrame(rows).sort_values("Promedio", ascending=False)

                        with st.expander(str(sec_name)):
                            st.dataframe(comp_df, use_container_width=True)

    # =====================================================
    # COMENTARIOS
    # =====================================================

    with tab3:

        open_cols = [c for c in f.columns if c.endswith("_txt")]
        open_cols = [c for c in open_cols if f[c].notna().any()]

        if not open_cols:
            st.info("No hay comentarios.")
            return

        section_map = {}
        for col in open_cols:
            sec = col.split("_")[0]
            section_map.setdefault(sec, []).append(col)

        section_names = {
            sec: SECTION_LABELS.get(sec, sec)
            for sec in section_map
        }

        section_sel = st.selectbox(
            "Sección",
            ["(Todas)"] + list(section_names.values())
        )

        if section_sel == "(Todas)":
            cols = open_cols
        else:
            sec_code = next(k for k, v in section_names.items() if v == section_sel)
            cols = section_map[sec_code]

        col_sel = st.selectbox("Campo de comentario", cols)

        textos = f[col_sel].dropna()
        textos = textos[textos.astype(str).str.strip() != ""]

        st.caption(f"Comentarios encontrados: {len(textos)}")

        st.dataframe(
            pd.DataFrame({"Comentario": textos}),
            use_container_width=True
        )
