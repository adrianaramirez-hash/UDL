import pandas as pd
import streamlit as st
import altair as alt
import gspread
import textwrap
import unicodedata

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
SHEET_CATALOGO = "Catalogo_Servicio"

FINANZAS_SHEET_ID = "11qszwEcEA6vvy7XYGo-w_WkqPp1kxoNG5GfJB_Wcc4A"
FINANZAS_SHEET_NAME = "VISTA_FINANZAS_NUM"


def _to_datetime_safe(s):
    return pd.to_datetime(s, errors="coerce", dayfirst=True)


def _pick_fecha_col(df):
    for c in ["Marca temporal", "Fecha", "timestamp"]:
        if c in df.columns:
            return c
    return None


def _mean_numeric(series):
    return pd.to_numeric(series, errors="coerce").mean()


def _best_carrera_col(df):
    for c in ["Carrera_Catalogo", "Servicio", "Programa", "Carrera"]:
        if c in df.columns:
            return c
    return None


def _download_button(df, filename):
    if df.empty:
        return
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ Descargar dataset filtrado (CSV)",
        data=csv,
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


@st.cache_data(ttl=300)
def _load_finanzas():
    sa = dict(st.secrets["gcp_service_account_json"])
    gc = gspread.service_account_from_dict(sa)
    sh = gc.open_by_key(FINANZAS_SHEET_ID)
    ws = sh.worksheet(FINANZAS_SHEET_NAME)
    data = ws.get_all_values()
    return pd.DataFrame(data[1:], columns=data[0]).replace("", pd.NA)


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


def render_encuesta_calidad(vista=None, carrera=None):

    st.subheader("Encuesta de calidad")
    vista = vista or "Dirección General"

    if vista == "Dirección Finanzas":

        df = _load_finanzas()
        if df.empty:
            st.warning("Sin datos disponibles.")
            return

        fecha_col = _pick_fecha_col(df)
        if fecha_col:
            df[fecha_col] = _to_datetime_safe(df[fecha_col])
            years = ["(Todos)"] + sorted(df[fecha_col].dt.year.dropna().unique(), reverse=True)
        else:
            years = ["(Todos)"]

        with st.sidebar:
            year = st.selectbox("Año", years)

        f = df.copy()
        if year != "(Todos)" and fecha_col:
            f = f[f[fecha_col].dt.year == int(year)]

        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption(f"Registros filtrados: {len(f)}")
        with col2:
            _download_button(f, f"encuesta_finanzas_{year}.csv")

        st.dataframe(f, use_container_width=True)
        return

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

    fecha_col = _pick_fecha_col(df)
    if fecha_col:
        df[fecha_col] = _to_datetime_safe(df[fecha_col])
        years = ["(Todos)"] + sorted(df[fecha_col].dt.year.dropna().unique(), reverse=True)
    else:
        years = ["(Todos)"]

    carrera_col = _best_carrera_col(df)

    with st.sidebar:
        year = st.selectbox("Año", years)
        if vista == "Dirección General" and carrera_col:
            carrera_sel = st.selectbox("Carrera", ["(Todas)"] + sorted(df[carrera_col].dropna().unique()))
        else:
            carrera_sel = carrera

    f = df.copy()

    if year != "(Todos)" and fecha_col:
        f = f[f[fecha_col].dt.year == int(year)]

    if carrera_sel and carrera_sel != "(Todas)" and carrera_col:
        f = f[f[carrera_col] == carrera_sel]

    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption(f"Registros filtrados: {len(f)}")
    with col2:
        _download_button(f, f"encuesta_{modalidad}_{year}.csv")

    if f.empty:
        st.warning("No hay datos con los filtros seleccionados.")
        return

    numeric_cols = [c for c in f.columns if c.endswith("_num")]
    likert_cols = [c for c in numeric_cols if pd.to_numeric(f[c], errors="coerce").max() > 1]
    yesno_cols = [c for c in numeric_cols if c not in likert_cols]

    tab1, tab2 = st.tabs(["Resumen", "Comentarios"])

    with tab1:
        colA, colB, colC = st.columns(3)
        colA.metric("Respuestas", len(f))

        if likert_cols:
            colB.metric("Promedio global", round(pd.to_numeric(f[likert_cols].stack(), errors="coerce").mean(), 2))

        if yesno_cols:
            colC.metric("% Sí", round(pd.to_numeric(f[yesno_cols].stack(), errors="coerce").mean() * 100, 1))

    with tab2:

        open_cols = [c for c in f.columns if c.endswith("_txt")]
        open_cols = [c for c in open_cols if f[c].notna().any()]

        if not open_cols:
            st.info("No hay comentarios disponibles.")
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
            sec_code = [k for k, v in section_names.items() if v == section_sel][0]
            cols = section_map[sec_code]

        col_sel = st.selectbox("Campo", cols)

        textos = f[col_sel].dropna()
        textos = textos[textos.astype(str).str.strip() != ""]

        st.caption(f"Comentarios encontrados: {len(textos)}")
        st.dataframe(pd.DataFrame({"Comentario": textos}), use_container_width=True)
