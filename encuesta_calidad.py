# encuesta_calidad.py
import pandas as pd
import streamlit as st
import altair as alt
import gspread
import re
import io

# ============================================================
# Etiquetas de secciones (fallback)
# ============================================================
SECTION_LABELS = {
    "DIR": "Director/Coordinación",
    "SER": "Servicios (Administrativos/Generales)",
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
    "REC": "Recomendación / Satisfacción",
    "OTR": "Otros",
}

# ============================================================
# Restricción para Dirección Finanzas
# ============================================================
DF_ALLOWED_PREFIX = {
    "Escolarizado / Ejecutivas": {"SER", "INS", "SEAC", "REC"},
    "Preparatoria": {"SER", "INS", "SEAC", "REC"},
    "Virtual / Mixto": {"SEAC", "ADM", "PLAT", "UDL", "REC"},
}

# ============================================================
# Nombres de pestañas
# ============================================================
SHEET_PROCESADO_DEFAULT = "PROCESADO"
SHEET_PROCESADO_DF = "VISTA_FINANZAS_NUM"
SHEET_MAPA = "Mapa_Preguntas"

# ============================================================
# Helpers
# ============================================================

def _download_excel_report(tabs_dict: dict, filename: str):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, df in tabs_dict.items():
            if df is not None and not df.empty:
                df.to_excel(writer, sheet_name=name[:31], index=False)

    st.download_button(
        "⬇️ Descargar informe completo (Excel)",
        buffer.getvalue(),
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )


def _to_datetime_safe(s):
    return pd.to_datetime(s, errors="coerce", dayfirst=True)


def _pick_fecha_col(df):
    for c in ["Marca temporal", "Marca Temporal", "Fecha"]:
        if c in df.columns:
            return c
    return None


def _get_url_for_modalidad(modalidad):
    return st.secrets[
        {
            "Virtual / Mixto": "EC_VIRTUAL_URL",
            "Escolarizado / Ejecutivas": "EC_ESCOLAR_URL",
            "Preparatoria": "EC_PREPA_URL",
        }[modalidad]
    ]


@st.cache_data(ttl=300)
def _load_from_gsheets_by_url(url, sheet_procesado):
    sa = dict(st.secrets["gcp_service_account_json"])
    gc = gspread.service_account_from_dict(sa)
    sh = gc.open_by_url(url)

    df = pd.DataFrame(sh.worksheet(sheet_procesado).get_all_records())
    mapa = pd.DataFrame(sh.worksheet(SHEET_MAPA).get_all_records())
    return df, mapa


# ============================================================
# Render principal
# ============================================================
def render_encuesta_calidad(vista=None, carrera=None):

    st.subheader("Encuesta de calidad")

    if not vista:
        vista = "Dirección General"

    # ========================================================
    # SIDEBAR
    # ========================================================
    with st.sidebar:
        modalidad = st.selectbox(
            "Modalidad",
            ["Virtual / Mixto", "Escolarizado / Ejecutivas", "Preparatoria"],
        )

    url = _get_url_for_modalidad(modalidad)
    sheet = SHEET_PROCESADO_DF if vista == "Dirección Finanzas" else SHEET_PROCESADO_DEFAULT

    df, mapa = _load_from_gsheets_by_url(url, sheet)

    if df.empty:
        st.warning("Sin datos.")
        return

    fecha_col = _pick_fecha_col(df)
    if fecha_col:
        df[fecha_col] = _to_datetime_safe(df[fecha_col])

    # ========================================================
    # Restricción DF
    # ========================================================
    if vista == "Dirección Finanzas":
        allowed = DF_ALLOWED_PREFIX.get(modalidad, set())
        mapa = mapa[mapa["section_code"].isin(allowed)]

    # ========================================================
    # Export dict
    # ========================================================
    export_tabs = {}

    # ========================================================
    # TABS
    # ========================================================
    tab1, tab2 = st.tabs(["Resumen", "Por sección"])

    # ========================================================
    # RESUMEN
    # ========================================================
    with tab1:

        numeric_cols = [c for c in df.columns if c.endswith("_num")]
        overall = pd.to_numeric(df[numeric_cols].stack(), errors="coerce").mean()

        st.metric("Respuestas", len(df))
        st.metric("Promedio global", f"{overall:.2f}" if pd.notna(overall) else "—")

        rows = []

        for sec in mapa["section_code"].unique():
            cols = mapa[mapa["section_code"] == sec]["header_num"]
            cols = [c for c in cols if c in df.columns]

            if cols:
                val = pd.to_numeric(df[cols].stack(), errors="coerce").mean()
                rows.append({
                    "Sección": SECTION_LABELS.get(sec, sec),
                    "Promedio": round(val, 2)
                })

        sec_df = pd.DataFrame(rows).sort_values("Promedio")
        st.dataframe(sec_df, use_container_width=True)

        if not sec_df.empty:
            export_tabs["Resumen_Secciones"] = sec_df.copy()

    # ========================================================
    # POR SECCIÓN
    # ========================================================
    with tab2:

        for sec in mapa["section_code"].unique():

            sec_name = SECTION_LABELS.get(sec, sec)
            sec_map = mapa[mapa["section_code"] == sec]
            cols = sec_map["header_num"]
            cols = [c for c in cols if c in df.columns]

            if not cols:
                continue

            sec_avg = pd.to_numeric(df[cols].stack(), errors="coerce").mean()

            with st.expander(f"{sec_name} — Promedio: {sec_avg:.2f}"):

                q_rows = []
                for _, row in sec_map.iterrows():
                    col = row["header_num"]
                    if col in df.columns:
                        val = pd.to_numeric(df[col], errors="coerce").mean()
                        q_rows.append({
                            "Sección": sec_name,
                            "Pregunta": row["header_raw"],
                            "Promedio": round(val, 2)
                        })

                q_df = pd.DataFrame(q_rows).sort_values("Promedio")
                st.dataframe(q_df, use_container_width=True)

                if not q_df.empty:
                    if "Preguntas" not in export_tabs:
                        export_tabs["Preguntas"] = q_df.copy()
                    else:
                        export_tabs["Preguntas"] = pd.concat(
                            [export_tabs["Preguntas"], q_df],
                            ignore_index=True
                        )

                # Comentarios
                open_cols = [c for c in df.columns if not c.endswith("_num")]

                comments = []
                for c in open_cols:
                    s = df[c].dropna().astype(str)
                    s = s[s.str.strip() != ""]
                    for val in s:
                        comments.append({
                            "Sección": sec_name,
                            "Comentario": val
                        })

                com_df = pd.DataFrame(comments)
                st.dataframe(com_df, use_container_width=True)

                if not com_df.empty:
                    if "Comentarios" not in export_tabs:
                        export_tabs["Comentarios"] = com_df.copy()
                    else:
                        export_tabs["Comentarios"] = pd.concat(
                            [export_tabs["Comentarios"], com_df],
                            ignore_index=True
                        )

    # ========================================================
    # EXPORTAR
    # ========================================================
    st.divider()
    st.markdown("### Exportar informe")

    if export_tabs:
        filename = f"Encuesta_Calidad_{modalidad.replace(' ','_')}_{vista.replace(' ','_')}.xlsx"
        _download_excel_report(export_tabs, filename)
    else:
        st.info("No hay datos disponibles para exportar.")
