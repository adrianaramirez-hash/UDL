import streamlit as st
import pandas as pd
import plotly.express as px

# =====================================================
# CONFIG
# =====================================================
URL_SEGUIMIENTO = "https://docs.google.com/spreadsheets/d/1Dgu3_UMAYecX-KCxLhHUe_EYiXCF68rvE2XpYwOx9lM/export?format=csv&gid=519739604"


# =====================================================
# CARGA DE DATOS
# =====================================================
@st.cache_data(ttl=300)
def cargar_datos():
    df = pd.read_csv(URL_SEGUIMIENTO)

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    return df


# =====================================================
# LIMPIEZA
# =====================================================
def limpiar(df):
    df = df.copy()

    if "area_de_adscripcion" not in df.columns:
        df["area_de_adscripcion"] = "SIN ÁREA"

    if "curso" not in df.columns:
        df["curso"] = "SIN CURSO"

    if "estatus_final" not in df.columns:
        df["estatus_final"] = "SIN ESTATUS"

    if "nombre_normalizado" not in df.columns:
        df["nombre_normalizado"] = "SIN NOMBRE"

    df["estatus_final"] = df["estatus_final"].astype(str).str.upper()

    return df


# =====================================================
# KPIs
# =====================================================
def calcular_kpis(df):
    total_inscripciones = len(df)
    docentes = df["nombre_normalizado"].nunique()
    cursos = df["curso"].nunique()

    finalizados = len(df[df["estatus_final"] == "FINALIZADO"])
    proceso = len(df[df["estatus_final"] == "EN_PROCESO"])
    no_seac = len(df[df["estatus_final"] == "NO_APARECE_EN_SEAC"])
    sin_tareas = len(df[df["estatus_final"] == "EN_SEAC_SIN_TAREAS"])

    return {
        "docentes": docentes,
        "inscripciones": total_inscripciones,
        "cursos": cursos,
        "finalizados": finalizados,
        "proceso": proceso,
        "no_seac": no_seac,
        "sin_tareas": sin_tareas
    }


# =====================================================
# FUNCIÓN PRINCIPAL (LA QUE LLAMA APP.PY)
# =====================================================
def render_capacitacion_docente(vista=None, carrera=None):

    st.title("📚 Capacitación Docente")

    df = cargar_datos()
    df = limpiar(df)

    kpis = calcular_kpis(df)

    # =====================================================
    # KPIS
    # =====================================================
    st.subheader("Resumen general")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Docentes", kpis["docentes"])
    c2.metric("Inscripciones", kpis["inscripciones"])
    c3.metric("Cursos", kpis["cursos"])
    c4.metric("Finalizados", kpis["finalizados"])

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("En proceso", kpis["proceso"])
    c6.metric("No en SEAC", kpis["no_seac"])
    c7.metric("Sin tareas", kpis["sin_tareas"])
    c8.metric("Avance %", round((kpis["finalizados"]/kpis["inscripciones"])*100,1) if kpis["inscripciones"] else 0)

    st.divider()

    # =====================================================
    # PESTAÑAS
    # =====================================================
    tab1, tab2, tab3 = st.tabs([
        "📊 General",
        "🏫 Por área",
        "📘 Por curso"
    ])

    # =====================================================
    # GENERAL
    # =====================================================
    with tab1:
        st.subheader("Distribución general")

        fig = px.histogram(df, x="estatus_final")
        st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # POR ÁREA
    # =====================================================
    with tab2:
        st.subheader("Por área de adscripción")

        resumen = df.groupby("area_de_adscripcion").agg(
            docentes=("nombre_normalizado", "nunique"),
            inscripciones=("curso", "count"),
            finalizados=("estatus_final", lambda x: (x == "FINALIZADO").sum())
        ).reset_index()

        fig = px.bar(resumen, x="area_de_adscripcion", y="docentes")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(resumen, use_container_width=True)

    # =====================================================
    # POR CURSO
    # =====================================================
    with tab3:
        cursos = df["curso"].dropna().unique()
        curso_sel = st.selectbox("Selecciona curso", cursos)

        df_curso = df[df["curso"] == curso_sel]

        st.write(f"Total inscritos: {len(df_curso)}")

        fig = px.pie(df_curso, names="estatus_final")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df_curso, use_container_width=True)
