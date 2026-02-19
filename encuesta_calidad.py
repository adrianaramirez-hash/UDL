import streamlit as st
import pandas as pd
import gspread
import io
import re

# ============================================================
# CONFIGURACIÓN
# ============================================================

SHEET_PROCESADO_DEFAULT = "PROCESADO"
SHEET_PROCESADO_DF = "VISTA_FINANZAS_NUM"
SHEET_MAPA = "Mapa_Preguntas"

SECTION_LABELS = {
    "DIR": "Director/Coordinación",
    "SER": "Servicios institucionales",
    "ADM": "Soporte administrativo",
    "APR": "Aprendizaje",
    "EVA": "Evaluación del conocimiento",
    "SEAC": "Plataforma SEAC",
    "PLAT": "Plataforma SEAC",
    "UDL": "Comunicación con la Universidad",
    "COM": "Comunicación con compañeros",
    "INS": "Instalaciones y equipo tecnológico",
    "REC": "Satisfacción y recomendación",
}

DF_ALLOWED_PREFIX = {
    "Escolarizado / Ejecutivas": {"SER", "INS", "SEAC", "REC"},
    "Preparatoria": {"SER", "INS", "SEAC", "REC"},
    "Virtual / Mixto": {"SEAC", "ADM", "PLAT", "UDL", "REC"},
}

# ============================================================
# UTILIDADES
# ============================================================

def _get_url(modalidad):
    return st.secrets[f"EC_{modalidad.upper().replace(' / ', '_').replace(' ', '_')}_URL"]

def _mean(series):
    return pd.to_numeric(series, errors="coerce").mean()

def _looks_open(col):
    c = col.lower()
    return any(x in c for x in ["coment", "suger", "¿por qué", "por qué"])

def _download_excel(tabs_dict, filename):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, df in tabs_dict.items():
            if df is not None and not df.empty:
                df.to_excel(writer, sheet_name=name[:31], index=False)

    st.download_button(
        "⬇️ Descargar Excel",
        buffer.getvalue(),
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

# ============================================================
# CARGA
# ============================================================

@st.cache_data(ttl=300)
def load_data(url, sheet_name):
    sa = dict(st.secrets["gcp_service_account_json"])
    gc = gspread.service_account_from_dict(sa)
    sh = gc.open_by_url(url)

    df = pd.DataFrame(sh.worksheet(sheet_name).get_all_records())
    mapa = pd.DataFrame(sh.worksheet(SHEET_MAPA).get_all_records())

    return df, mapa

# ============================================================
# RENDER PRINCIPAL
# ============================================================

def render_encuesta_calidad(vista="Dirección General", carrera=None):

    st.subheader("Encuesta de calidad")

    # ========================================================
    # SIDEBAR
    # ========================================================

    with st.sidebar:
        modalidad = st.selectbox(
            "Modalidad",
            ["Escolarizado / Ejecutivas", "Preparatoria", "Virtual / Mixto"]
        )

        year = st.selectbox("Año", ["(Todos)"])

        carrera_sel = st.text_input("Carrera/Servicio", value=carrera or "")

    # ========================================================
    # CARGA
    # ========================================================

    sheet = SHEET_PROCESADO_DF if vista == "Dirección Finanzas" else SHEET_PROCESADO_DEFAULT

    url_key = {
        "Escolarizado / Ejecutivas": "EC_ESCOLAR_URL",
        "Preparatoria": "EC_PREPA_URL",
        "Virtual / Mixto": "EC_VIRTUAL_URL",
    }

    url = st.secrets[url_key[modalidad]]

    df, mapa = load_data(url, sheet)

    if df.empty:
        st.warning("Sin datos.")
        return

    # ========================================================
    # FILTRO DF
    # ========================================================

    if vista == "Dirección Finanzas":
        allowed = DF_ALLOWED_PREFIX.get(modalidad, set())
        mapa = mapa[mapa["header_num"].str.split("_").str[0].isin(allowed)]

    # ========================================================
    # RESUMEN
    # ========================================================

    tabs_export = {}

    tab1, tab2 = st.tabs(["Resumen", "Por sección"])

    with tab1:

        numeric_cols = [c for c in df.columns if c.endswith("_num")]

        overall = _mean(df[numeric_cols].stack())

        st.metric("Respuestas", len(df))
        st.metric("Promedio global", f"{overall:.2f}" if pd.notna(overall) else "—")

        rows = []

        for sec in mapa["section_code"].unique():
            cols = mapa[mapa["section_code"] == sec]["header_num"]
            cols = [c for c in cols if c in df.columns]

            if cols:
                val = _mean(df[cols].stack())
                rows.append({
                    "Sección": SECTION_LABELS.get(sec, sec),
                    "Promedio": round(val, 2)
                })

        sec_df = pd.DataFrame(rows).sort_values("Promedio")
        st.dataframe(sec_df, use_container_width=True)

        tabs_export["Resumen_Secciones"] = sec_df

    # ========================================================
    # POR SECCIÓN
    # ========================================================

    with tab2:

        export_preguntas = []
        export_comentarios = []

        for sec in mapa["section_code"].unique():

            sec_name = SECTION_LABELS.get(sec, sec)
            sec_map = mapa[mapa["section_code"] == sec]

            cols = sec_map["header_num"]
            cols = [c for c in cols if c in df.columns]

            if not cols:
                continue

            sec_avg = _mean(df[cols].stack())

            with st.expander(f"{sec_name} — Promedio: {sec_avg:.2f}"):

                q_rows = []

                for _, row in sec_map.iterrows():
                    col = row["header_num"]
                    if col in df.columns:
                        val = _mean(df[col])
                        q_rows.append({
                            "Pregunta": row["header_exacto"],
                            "Resultado": round(val, 2)
                        })

                q_df = pd.DataFrame(q_rows).sort_values("Resultado")
                st.dataframe(q_df, use_container_width=True)

                export_preguntas.append(q_df.assign(Sección=sec_name))

                # -------------------------
                # COMENTARIOS
                # -------------------------

                open_cols = [c for c in df.columns if not c.endswith("_num") and _looks_open(c)]

                show_all = st.checkbox("Ver todos los comentarios", key=f"{sec}_all")
                query = "" if show_all else st.text_input("Buscar en comentarios…", key=f"{sec}_q")

                comments = []

                for col in open_cols:
                    tmp = df[col].dropna().astype(str)
                    tmp = tmp[tmp.str.strip() != ""]
                    if query:
                        tmp = tmp[tmp.str.contains(query, case=False)]

                    comments.extend(tmp.tolist())

                com_df = pd.DataFrame({"Comentario": comments})

                st.caption(f"Comentarios encontrados: {len(com_df)}")
                st.dataframe(com_df, use_container_width=True)

                if not com_df.empty:
                    export_comentarios.append(
                        com_df.assign(Sección=sec_name)
                    )

        if export_preguntas:
            tabs_export["Preguntas"] = pd.concat(export_preguntas)

        if export_comentarios:
            tabs_export["Comentarios"] = pd.concat(export_comentarios)

    # ========================================================
    # BOTÓN DESCARGA GLOBAL
    # ========================================================

    st.divider()
    st.markdown("### Exportar informe")

    filename = f"Encuesta_Calidad_{modalidad.replace(' ','_')}.xlsx"
    _download_excel(tabs_export, filename)
