import pandas as pd
import streamlit as st
import altair as alt
import gspread
import re

# =====================================================
# CONFIG
# =====================================================
SHEET_NAME_DEFAULT = "REPROBACION"
UMBRAL_REPROBACION_DEFAULT = 70


# =====================================================
# NORMALIZACIÓN
# =====================================================
def normalizar_texto(valor):
    if pd.isna(valor):
        return ""
    s = str(valor).lower()
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return s.strip()


def _norm_cols(df):
    df.columns = [str(c).strip().upper() for c in df.columns]
    return df


def _to_num(s):
    return pd.to_numeric(s, errors="coerce")


# =====================================================
# CARGA DE DATOS
# =====================================================
@st.cache_data(ttl=300)
def cargar_datos(url):
    sa = st.secrets["gcp_service_account_json"]
    gc = gspread.service_account_from_dict(dict(sa))
    sh = gc.open_by_url(url)

    ws = sh.sheet1
    values = ws.get_all_values()

    df = pd.DataFrame(values[1:], columns=values[0])
    df = _norm_cols(df)

    return df


# =====================================================
# LIMPIEZA
# =====================================================
def limpiar(df):
    df["CALIF_FINAL"] = _to_num(df.get("CALIF_FINAL"))
    df["AREA"] = df.get("AREA", "").astype(str)
    df["MATERIA"] = df.get("MATERIA", "").astype(str)
    df["MATRICULA"] = df.get("MATRICULA", "").astype(str)
    df["CICLO"] = df.get("CICLO", "").astype(str)

    df["MATRICULA_norm"] = df["MATRICULA"].str.lower().str.strip()

    df = df[df["CALIF_FINAL"] < 70]

    return df


# =====================================================
# KPIs
# =====================================================
def calcular_kpis(df):
    alumnos = df["MATRICULA_norm"].nunique()
    registros = len(df)
    materias = df["MATERIA"].nunique()
    promedio = df["CALIF_FINAL"].mean()

    ratio = registros / alumnos if alumnos else 0

    return alumnos, registros, materias, promedio, ratio


# =====================================================
# FUNCIONES ANALÍTICAS
# =====================================================
def top_materias(df):
    g = df.groupby("MATERIA")

    tabla = pd.DataFrame({
        "Materia": g.size().index,
        "Alumnos únicos": g["MATRICULA_norm"].nunique().values,
        "Total reprobaciones": g.size().values
    })

    return tabla.sort_values("Alumnos únicos", ascending=False).head(10)


def resumen_carreras(df):
    g = df.groupby("AREA")

    tabla = pd.DataFrame({
        "Carrera": g.size().index,
        "Alumnos únicos": g["MATRICULA_norm"].nunique().values,
        "Total reprobaciones": g.size().values
    })

    return tabla.sort_values("Alumnos únicos", ascending=False)


def historico(df):
    g = df.groupby("CICLO")

    tabla = pd.DataFrame({
        "Ciclo": g.size().index,
        "Alumnos únicos": g["MATRICULA_norm"].nunique().values
    })

    return tabla


# =====================================================
# RENDER PRINCIPAL
# =====================================================
def render_indice_reprobacion(vista=None, carrera=None):

    st.title("Índice de reprobación")

    url = st.secrets.get("IR_URL")

    if not url:
        st.error("Falta URL")
        return

    df = cargar_datos(url)
    df = limpiar(df)

    if df.empty:
        st.warning("No hay datos")
        return

    # =========================
    # FILTROS
    # =========================
    col1, col2 = st.columns(2)

    with col1:
        area = st.selectbox("Carrera", ["Todas"] + sorted(df["AREA"].unique()))

    with col2:
        ciclo = st.selectbox("Ciclo", ["Todos"] + sorted(df["CICLO"].unique()))

    if area != "Todas":
        df = df[df["AREA"] == area]

    if ciclo != "Todos":
        df = df[df["CICLO"] == ciclo]

    # =========================
    # KPIs
    # =========================
    alumnos, registros, materias, promedio, ratio = calcular_kpis(df)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Alumnos con reprobación", alumnos)
    c2.metric("Materias reprobadas", registros)
    c3.metric("Materias distintas", materias)
    c4.metric("Promedio reprobatorio", round(promedio, 2))

    st.info(
        f"En este filtro hay {alumnos} alumnos que acumulan {registros} reprobaciones. "
        f"Cada alumno reprueba en promedio {round(ratio,1)} materias."
    )

    st.divider()

    # =========================
    # RESUMEN
    # =========================
    st.subheader("1. Materias críticas")

    tm = top_materias(df)

    st.bar_chart(tm.set_index("Materia")["Alumnos únicos"])
    st.dataframe(tm)

    st.divider()

    st.subheader("2. Comparativo por carrera")

    rc = resumen_carreras(df)

    if len(rc) > 1:
        st.bar_chart(rc.set_index("Carrera")["Alumnos únicos"])

    st.dataframe(rc)

    st.divider()

    st.subheader("3. Tendencia por ciclo")

    hist = historico(df)

    st.line_chart(hist.set_index("Ciclo"))
    st.dataframe(hist)
