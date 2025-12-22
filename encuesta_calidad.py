# encuesta_calidad.py
import pandas as pd
import streamlit as st
import altair as alt
import gspread
import textwrap

# ============================================================
# Etiquetas de secciones
# ============================================================
SECTION_LABELS = {
    "DIR": "Director / Coordinación",
    "SER": "Servicios administrativos y generales",
    "ADM": "Acceso a soporte administrativo",
    "ACD": "Servicios académicos",
    "APR": "Aprendizaje",
    "EVA": "Evaluación del conocimiento",
    "SEAC": "Plataforma SEAC",
    "PLAT": "Plataforma SEAC",
    "SAT": "Plataforma SEAC",  # Prepa
    "MAT": "Materiales en la plataforma",
    "UDL": "Comunicación con la Universidad",
    "COM": "Comunicación con compañeros",
    "INS": "Instalaciones y equipo tecnológico",
    "AMB": "Ambiente escolar",
    "REC": "Recomendación y satisfacción",
    "OTR": "Otros",
}

# ============================================================
# 🔴 PARCHE CLAVE: columnas Sí / No reales (0–1)
# ============================================================
YESNO_COLS = {
    # Virtual / Mixto
    "ADM_ContactoExiste_num",
    "UDL_ViasComunicacion_num",
    "REC_Volveria_num",
    "REC_Recomendaria_num",

    # Escolarizado / Ejecutivas
    "AMB_ProblemaCompanero_num",
}

MAX_VERTICAL_QUESTIONS = 7
MAX_VERTICAL_SECTIONS = 7

# ============================================================
# Helpers
# ============================================================
def _section_from_numcol(col: str) -> str:
    return col.split("_", 1)[0] if "_" in col else "OTR"


def _is_yesno_col(col: str) -> bool:
    return col in YESNO_COLS


def _to_datetime_safe(s):
    return pd.to_datetime(s, errors="coerce", dayfirst=True)


def _wrap_text(s: str, width: int = 18, max_lines: int = 3) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s).strip()
    if not s:
        return ""
    lines = textwrap.wrap(s, width=width)
    if len(lines) <= max_lines:
        return "\n".join(lines)
    kept = lines[:max_lines]
    kept[-1] = kept[-1][:-1] + "…" if kept[-1] else "…"
    return "\n".join(kept)


def _mean_numeric(series: pd.Series):
    return pd.to_numeric(series, errors="coerce").mean()


def _bar_chart_auto(
    df_in: pd.DataFrame,
    category_col: str,
    value_col: str,
    value_domain: list,
    value_title: str,
    tooltip_cols: list,
    max_vertical: int,
    wrap_width_vertical: int = 18,
    wrap_width_horizontal: int = 30,
    height_per_row: int = 28,
    base_height: int = 260,
):
    if df_in is None or df_in.empty:
        return None

    df = df_in.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col])
    if df.empty:
        return None

    n = len(df)

    if n <= max_vertical:
        df["_cat"] = df[category_col].apply(lambda x: _wrap_text(x, wrap_width_vertical))
        return (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("_cat:N", sort="-y", title=None),
                y=alt.Y(f"{value_col}:Q", scale=alt.Scale(domain=value_domain), title=value_title),
                tooltip=tooltip_cols,
            )
            .properties(height=base_height)
        )

    df["_cat"] = df[category_col].apply(lambda x: _wrap_text(x, wrap_width_horizontal))
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y("_cat:N", sort="-x", title=None),
            x=alt.X(f"{value_col}:Q", scale=alt.Scale(domain=value_domain), title=value_title),
            tooltip=tooltip_cols,
        )
        .properties(height=max(base_height, n * height_per_row))
    )

# ============================================================
# Google Sheets
# ============================================================
@st.cache_data(ttl=300, show_spinner=False)
def _load_from_gsheets_by_url(url: str):
    sa = dict(st.secrets["gcp_service_account_json"])
    gc = gspread.service_account_from_dict(sa)
    sh = gc.open_by_url(url)

    def ws_to_df(ws):
        values = ws.get_all_values()
        if not values:
            return pd.DataFrame()
        return pd.DataFrame(values[1:], columns=values[0]).replace("", pd.NA)

    df = ws_to_df(sh.worksheet("PROCESADO"))
    mapa = ws_to_df(sh.worksheet("Mapa_Preguntas"))
    return df, mapa


def _get_url_for_modalidad(modalidad: str) -> str:
    key = {
        "Virtual / Mixto": "EC_VIRTUAL_URL",
        "Escolarizado / Ejecutivas": "EC_ESCOLAR_URL",
        "Preparatoria": "EC_PREPA_URL",
    }[modalidad]
    return st.secrets[key]


# ============================================================
# Render principal
# ============================================================
def render_encuesta_calidad(vista: str = "Dirección General", carrera: str | None = None):
    st.subheader("Encuesta de calidad")

    modalidad = (
        st.selectbox("Modalidad", ["Virtual / Mixto", "Escolarizado / Ejecutivas", "Preparatoria"])
        if vista == "Dirección General"
        else "Preparatoria" if (carrera or "").lower() == "preparatoria"
        else "Escolarizado / Ejecutivas"
    )

    df, mapa = _load_from_gsheets_by_url(_get_url_for_modalidad(modalidad))

    if df.empty:
        st.warning("PROCESADO vacío.")
        return

    if "Marca temporal" in df.columns:
        df["Marca temporal"] = _to_datetime_safe(df["Marca temporal"])

    mapa["section_code"] = mapa["header_num"].apply(_section_from_numcol)
    mapa["section_name"] = mapa["section_code"].map(SECTION_LABELS).fillna(mapa["section_code"])
    mapa = mapa[mapa["header_num"].isin(df.columns)]

    num_cols = [c for c in df.columns if c.endswith("_num")]
    likert_cols = [c for c in num_cols if c not in YESNO_COLS]
    yesno_cols = [c for c in num_cols if c in YESNO_COLS]

    # ---------------------------
    # Resumen
    # ---------------------------
    st.divider()
    c1, c2, c3 = st.columns(3)

    c1.metric("Respuestas", len(df))

    if likert_cols:
        c2.metric(
            "Promedio global (Likert)",
            f"{pd.to_numeric(df[likert_cols].stack(), errors='coerce').mean():.2f}",
        )
    else:
        c2.metric("Promedio global (Likert)", "—")

    if yesno_cols:
        c3.metric(
            "% Sí",
            f"{pd.to_numeric(df[yesno_cols].stack(), errors='coerce').mean() * 100:.1f}%",
        )
    else:
        c3.metric("% Sí", "—")

    st.success("Encuesta de calidad cargada correctamente.")
