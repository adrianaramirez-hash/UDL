import pandas as pd
import streamlit as st
import altair as alt
import gspread
import textwrap
import unicodedata
from io import BytesIO

# =========================
# CONFIGURACIÓN GENERAL
# =========================

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

MAX_VERTICAL_QUESTIONS = 7
MAX_VERTICAL_SECTIONS = 7

SHEET_PROCESADO = "PROCESADO"
SHEET_MAPA = "MAPA_PREGUNTAS"
SHEET_CATALOGO = "Catalogo_Servicio"

FINANZAS_SHEET_ID = "11qszwEcEA6vvy7XYGo-w_WkqPp1kxoNG5GfJB_Wcc4A"
FINANZAS_SHEET_NAME = "VISTA_FINANZAS_NUM"


# =========================
# UTILIDADES GENERALES
# =========================

def _to_datetime_safe(s):
    return pd.to_datetime(s, errors="coerce", dayfirst=True)


def _pick_fecha_col(df):
    for c in ["Marca temporal", "Marca Temporal", "Fecha", "fecha", "timestamp", "Timestamp"]:
        if c in df.columns:
            return c
    return None


def _mean_numeric(series):
    return pd.to_numeric(series, errors="coerce").mean()


def _section_from_numcol(col: str) -> str:
    return col.split("_", 1)[0] if "_" in col else "OTR"


def _wrap_text(s: str, width: int = 18, max_lines: int = 3) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip()
    lines = textwrap.wrap(s, width=width)
    if len(lines) <= max_lines:
        return "\n".join(lines)
    kept = lines[:max_lines]
    kept[-1] = kept[-1][:-1] + "…"
    return "\n".join(kept)


def _auto_classify_numcols(df, cols):
    if not cols:
        return [], []
    dnum = df[cols].apply(pd.to_numeric, errors="coerce")
    maxs = dnum.max(axis=0, skipna=True)
    likert = [c for c in cols if c in maxs.index and pd.notna(maxs[c]) and float(maxs[c]) > 1]
    yesno = [c for c in cols if c not in likert]
    return likert, yesno


# =========================
# EXPORTACIÓN PROFESIONAL
# =========================

def _create_excel_download(df, kpis_df, filename):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="DATA_FILTRADA", index=False)

        if kpis_df is not None and not kpis_df.empty:
            kpis_df.to_excel(writer, sheet_name="KPIS", index=False)

            workbook = writer.book
            worksheet = writer.sheets["KPIS"]
            header_format = workbook.add_format({
                "bold": True,
                "bg_color": "#002147",
                "font_color": "white"
            })
            for col_num, value in enumerate(kpis_df.columns.values):
                worksheet.write(0, col_num, value, header_format)

    output.seek(0)

    st.download_button(
        "⬇️ DESCARGAR REPORTE COMPLETO (EXCEL)",
        data=output,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )


def _download_buttons(df, filename_prefix, kpis_df=None):

    if df is None or df.empty:
        return

    st.markdown("## ")
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

    st.markdown("---")


# =========================
# CARGA DE DATOS
# =========================

@st.cache_data(show_spinner=False, ttl=300)
def _load_from_gsheets_by_url(url):
    sa = dict(st.secrets["gcp_service_account_json"])
    gc = gspread.service_account_from_dict(sa)
    sh = gc.open_by_url(url)

    def normalize(x):
        return str(x).strip().lower().replace(" ", "").replace("_", "")

    titles = {normalize(ws.title): ws.title for ws in sh.worksheets()}

    def get_ws(expected_name):
        key = normalize(expected_name)
        if key not in titles:
            raise ValueError(f"No encontré hoja {expected_name}")
        ws = sh.worksheet(titles[key])
        data = ws.get_all_values()
        return pd.DataFrame(data[1:], columns=data[0]).replace("", pd.NA)

    df = get_ws(SHEET_PROCESADO)
    mapa = get_ws(SHEET_MAPA)

    return df, mapa


@st.cache_data(show_spinner=False, ttl=300)
def _load_finanzas():
    sa = dict(st.secrets["gcp_service_account_json"])
    gc = gspread.service_account_from_dict(sa)
    sh = gc.open_by_key(FINANZAS_SHEET_ID)
    ws = sh.worksheet(FINANZAS_SHEET_NAME)
    data = ws.get_all_values()
    return pd.DataFrame(data[1:], columns=data[0]).replace("", pd.NA)
def render_encuesta_calidad(vista: str | None = None, carrera: str | None = None):

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

        st.caption(f"Registros filtrados: {len(f)}")

        kpis = pd.DataFrame({
            "Indicador": ["Total registros"],
            "Valor": [len(f)]
        })

        _download_buttons(
            f,
            f"encuesta_finanzas_{year_sel}",
            kpis
        )

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

    df, mapa = _load_from_gsheets_by_url(url)

    if df.empty:
        st.warning("Sin datos disponibles.")
        return

    fecha_col = _pick_fecha_col(df)
    years = ["(Todos)"]

    if fecha_col:
        df[fecha_col] = _to_datetime_safe(df[fecha_col])
        years += sorted(df[fecha_col].dt.year.dropna().unique(), reverse=True)

    carrera_col = None
    for c in [
        "Carrera_Catalogo",
        "Servicio",
        "Selecciona el programa académico que estudias",
        "Programa",
        "Carrera",
    ]:
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

    st.caption(f"Registros filtrados: {len(f)}")

    # =====================================================
    # ===== NORMALIZAR MAPA (COMPATIBLE LAB Y NUEVO) =====
    # =====================================================

    if "header_num" not in mapa.columns:

        if {"header_raw", "header_id", "tipo"}.issubset(set(mapa.columns)):

            mapa = mapa.copy()

            mapa["header_exacto"] = mapa["header_raw"].astype(str).str.strip()

            mapa["scale_code"] = (
                mapa["tipo"]
                .astype(str)
                .str.upper()
                .map({
                    "LIKERT": "LIKERT_1_5",
                    "YESNO": "YESNO_0_1",
                    "ABIERTA": "ABIERTA"
                })
            )

            mapa["header_num"] = (
                mapa["header_id"].astype(str).str.strip() +
                mapa["tipo"].astype(str).str.upper().map({
                    "ABIERTA": "_txt"
                }).fillna("_num")
            )

        else:
            st.error("La hoja MAPA_PREGUNTAS no tiene estructura válida.")
            st.stop()

    # =====================================================
    # ================= KPIs GENERALES ====================
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

    if f.empty:
        st.warning("No hay datos con los filtros seleccionados.")
        return

    # =====================================================
    # ==================== TABS ===========================
    # =====================================================

    tab1, tab2, tab3 = st.tabs(["Resumen", "Por sección", "Comentarios"])

    # ================= RESUMEN =================

    with tab1:

        col1, col2, col3 = st.columns(3)

        col1.metric("Respuestas", len(f))

        if promedio_global is not None:
            col2.metric("Promedio global", promedio_global)

        if pct_si is not None:
            col3.metric("% Sí", f"{pct_si}%")

    # ================= POR SECCIÓN =================

    with tab2:

        mapa = mapa.copy()
        mapa["section_code"] = mapa["header_num"].apply(_section_from_numcol)
        mapa["section_name"] = mapa["section_code"].map(SECTION_LABELS)

        mapa_ok = mapa[mapa["header_num"].isin(f.columns)]

        rows = []

        for (sec_code, sec_name), g in mapa_ok.groupby(["section_code", "section_name"]):

            cols = [c for c in g["header_num"].tolist() if c in likert_cols]

            if not cols:
                continue

            val = pd.to_numeric(f[cols].stack(), errors="coerce").mean()

            if pd.isna(val):
                continue

            rows.append({
                "Sección": sec_name,
                "Promedio": round(val, 2)
            })

        if not rows:
            st.info("Sin datos suficientes.")
            return

        sec_df = pd.DataFrame(rows).sort_values("Promedio", ascending=False)

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

    # ================= COMENTARIOS =================

    with tab3:

        open_cols = [c for c in f.columns if c.endswith("_txt")]
        open_cols = [c for c in open_cols if f[c].notna().any()]

        if not open_cols:
            st.info("No hay comentarios disponibles.")
            return

        section_map = {}
        for col in open_cols:
            sec = col.split("_")[0]
            section_map.setdefault(sec, []).append(col)

        section_names = {sec: SECTION_LABELS.get(sec, sec) for sec in section_map}

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
