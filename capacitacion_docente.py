import streamlit as st
import pandas as pd
import plotly.express as px


# =====================================================
# CONFIGURACIÓN
# =====================================================

URL_SEGUIMIENTO = "https://docs.google.com/spreadsheets/d/1Dgu3_UMAYecX-KCxLhHUe_EYiXCF68rvE2XpYwOx9lM/export?format=csv&gid=519739604"


# =====================================================
# CARGA DE DATOS
# =====================================================

@st.cache_data(ttl=300)
def cargar_datos_capacitacion():
    df = pd.read_csv(URL_SEGUIMIENTO)

    # Normalizar nombres de columnas
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("á", "a")
        .str.replace("é", "e")
        .str.replace("í", "i")
        .str.replace("ó", "o")
        .str.replace("ú", "u")
    )

    return df


# =====================================================
# LIMPIEZA Y VALIDACIÓN
# =====================================================

def preparar_datos(df):
    df = df.copy()

    # Ajuste por si tu columna viene como area_de_adscripcion
    posibles_area = [
        "area_de_adscripcion",
        "area_adscripcion",
        "adscripcion",
        "area"
    ]

    columna_area = None
    for col in posibles_area:
        if col in df.columns:
            columna_area = col
            break

    if columna_area is None:
        df["area_de_adscripcion"] = "SIN ÁREA"
    else:
        df["area_de_adscripcion"] = df[columna_area].fillna("SIN ÁREA")

    # Columnas mínimas esperadas
    if "curso" not in df.columns:
        df["curso"] = "SIN CURSO"

    if "estatus_final" not in df.columns:
        df["estatus_final"] = "SIN ESTATUS"

    if "avance_pct" not in df.columns:
        df["avance_pct"] = 0

    if "nombre_normalizado" not in df.columns:
        if "nombre_forms" in df.columns:
            df["nombre_normalizado"] = df["nombre_forms"]
        elif "nombre_seac" in df.columns:
            df["nombre_normalizado"] = df["nombre_seac"]
        else:
            df["nombre_normalizado"] = "SIN NOMBRE"

    if "correo_docente" not in df.columns:
        df["correo_docente"] = ""

    df["estatus_final"] = df["estatus_final"].fillna("SIN ESTATUS").astype(str).str.upper()
    df["curso"] = df["curso"].fillna("SIN CURSO")
    df["area_de_adscripcion"] = df["area_de_adscripcion"].fillna("SIN ÁREA")
    df["nombre_normalizado"] = df["nombre_normalizado"].fillna("SIN NOMBRE")

    return df


# =====================================================
# MÉTRICAS GENERALES
# =====================================================

def calcular_kpis(df):
    total_inscripciones = len(df)
    total_docentes = df["nombre_normalizado"].nunique()
    total_cursos = df["curso"].nunique()

    finalizados = len(df[df["estatus_final"] == "FINALIZADO"])
    en_proceso = len(df[df["estatus_final"] == "EN_PROCESO"])
    no_seac = len(df[df["estatus_final"] == "NO_APARECE_EN_SEAC"])
    sin_tareas = len(df[df["estatus_final"] == "EN_SEAC_SIN_TAREAS"])

    tasa_finalizacion = (finalizados / total_inscripciones * 100) if total_inscripciones > 0 else 0
    tasa_sin_avance = ((no_seac + sin_tareas) / total_inscripciones * 100) if total_inscripciones > 0 else 0

    return {
        "total_inscripciones": total_inscripciones,
        "total_docentes": total_docentes,
        "total_cursos": total_cursos,
        "finalizados": finalizados,
        "en_proceso": en_proceso,
        "no_seac": no_seac,
        "sin_tareas": sin_tareas,
        "tasa_finalizacion": tasa_finalizacion,
        "tasa_sin_avance": tasa_sin_avance,
    }


# =====================================================
# VISTA PRINCIPAL
# =====================================================

def mostrar_modulo_capacitacion_docente():
    st.title("📚 Capacitación Docente")

    st.caption(
        "Seguimiento operativo de docentes inscritos en capacitaciones. "
        "Por ahora, el análisis se realiza con base en los docentes registrados en el archivo de seguimiento."
    )

    df = cargar_datos_capacitacion()
    df = preparar_datos(df)

    kpis = calcular_kpis(df)

    # =====================================================
    # KPIS GENERALES
    # =====================================================

    st.subheader("Resumen general")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Docentes inscritos", kpis["total_docentes"])
    col2.metric("Inscripciones", kpis["total_inscripciones"])
    col3.metric("Cursos activos", kpis["total_cursos"])
    col4.metric("Finalización", f"{kpis['tasa_finalizacion']:.1f}%")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Finalizados", kpis["finalizados"])
    col6.metric("En proceso", kpis["en_proceso"])
    col7.metric("No aparecen en SEAC", kpis["no_seac"])
    col8.metric("Sin tareas", kpis["sin_tareas"])

    st.divider()

    # =====================================================
    # PESTAÑAS
    # =====================================================

    tab_dg, tab_dc, tab_curso, tab_ranking, tab_detalle = st.tabs([
        "🏛️ Vista DG",
        "🎓 Vista Director de Carrera",
        "📘 Vista por curso",
        "🏆 Rankings",
        "📋 Detalle general"
    ])

    # =====================================================
    # VISTA DG
    # =====================================================

    with tab_dg:
        st.subheader("Panorama institucional")

        resumen_area = df.groupby("area_de_adscripcion").agg(
            docentes_inscritos=("nombre_normalizado", "nunique"),
            inscripciones=("curso", "count"),
            finalizados=("estatus_final", lambda x: (x == "FINALIZADO").sum()),
            en_proceso=("estatus_final", lambda x: (x == "EN_PROCESO").sum()),
            no_aparece_seac=("estatus_final", lambda x: (x == "NO_APARECE_EN_SEAC").sum()),
            sin_tareas=("estatus_final", lambda x: (x == "EN_SEAC_SIN_TAREAS").sum()),
        ).reset_index()

        resumen_area["tasa_finalizacion"] = (
            resumen_area["finalizados"] / resumen_area["inscripciones"] * 100
        ).round(1)

        fig_area = px.bar(
            resumen_area.sort_values("docentes_inscritos", ascending=False),
            x="area_de_adscripcion",
            y="docentes_inscritos",
            title="Docentes inscritos por área de adscripción",
            text="docentes_inscritos"
        )
        fig_area.update_layout(xaxis_title="Área de adscripción", yaxis_title="Docentes inscritos")
        st.plotly_chart(fig_area, use_container_width=True)

        fig_finalizacion = px.bar(
            resumen_area.sort_values("tasa_finalizacion", ascending=False),
            x="area_de_adscripcion",
            y="tasa_finalizacion",
            title="Tasa de finalización por área",
            text="tasa_finalizacion"
        )
        fig_finalizacion.update_layout(xaxis_title="Área de adscripción", yaxis_title="% finalización")
        st.plotly_chart(fig_finalizacion, use_container_width=True)

        st.dataframe(resumen_area, use_container_width=True)

    # =====================================================
    # VISTA DIRECTOR DE CARRERA
    # =====================================================

    with tab_dc:
        st.subheader("Vista por Director de Carrera")

        areas = sorted(df["area_de_adscripcion"].dropna().unique())
        area_sel = st.selectbox("Selecciona área de adscripción / carrera", areas)

        df_area = df[df["area_de_adscripcion"] == area_sel]

        kpis_area = calcular_kpis(df_area)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Docentes inscritos", kpis_area["total_docentes"])
        col2.metric("Inscripciones", kpis_area["total_inscripciones"])
        col3.metric("Cursos", kpis_area["total_cursos"])
        col4.metric("Finalización", f"{kpis_area['tasa_finalizacion']:.1f}%")

        resumen_docente = df_area.groupby("nombre_normalizado").agg(
            correo=("correo_docente", "first"),
            cursos_inscritos=("curso", "count"),
            cursos_finalizados=("estatus_final", lambda x: (x == "FINALIZADO").sum()),
            cursos_en_proceso=("estatus_final", lambda x: (x == "EN_PROCESO").sum()),
            no_aparece_seac=("estatus_final", lambda x: (x == "NO_APARECE_EN_SEAC").sum()),
            sin_tareas=("estatus_final", lambda x: (x == "EN_SEAC_SIN_TAREAS").sum()),
        ).reset_index()

        resumen_docente["avance_general"] = (
            resumen_docente["cursos_finalizados"] / resumen_docente["cursos_inscritos"] * 100
        ).round(1)

        st.dataframe(
            resumen_docente.sort_values("cursos_inscritos", ascending=False),
            use_container_width=True
        )

    # =====================================================
    # VISTA POR CURSO
    # =====================================================

    with tab_curso:
        st.subheader("Vista por curso")

        cursos = sorted(df["curso"].dropna().unique())
        curso_sel = st.selectbox("Selecciona curso", cursos)

        df_curso = df[df["curso"] == curso_sel]
        kpis_curso = calcular_kpis(df_curso)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Inscripciones", kpis_curso["total_inscripciones"])
        col2.metric("Docentes", kpis_curso["total_docentes"])
        col3.metric("Finalizados", kpis_curso["finalizados"])
        col4.metric("Finalización", f"{kpis_curso['tasa_finalizacion']:.1f}%")

        resumen_estatus = df_curso["estatus_final"].value_counts().reset_index()
        resumen_estatus.columns = ["estatus_final", "total"]

        fig_estatus = px.pie(
            resumen_estatus,
            names="estatus_final",
            values="total",
            title="Distribución de estatus del curso"
        )
        st.plotly_chart(fig_estatus, use_container_width=True)

        columnas_mostrar = [
            "nombre_normalizado",
            "correo_docente",
            "area_de_adscripcion",
            "curso",
            "estatus_final",
            "avance_pct"
        ]

        columnas_existentes = [c for c in columnas_mostrar if c in df_curso.columns]

        st.dataframe(
            df_curso[columnas_existentes],
            use_container_width=True
        )

    # =====================================================
    # RANKINGS
    # =====================================================

    with tab_ranking:
        st.subheader("Rankings de capacitación")

        ranking_areas = df.groupby("area_de_adscripcion").agg(
            docentes_inscritos=("nombre_normalizado", "nunique"),
            inscripciones=("curso", "count")
        ).reset_index().sort_values("docentes_inscritos", ascending=False)

        ranking_cursos = df.groupby("curso").agg(
            inscritos=("nombre_normalizado", "count"),
            finalizados=("estatus_final", lambda x: (x == "FINALIZADO").sum())
        ).reset_index()

        ranking_cursos["tasa_finalizacion"] = (
            ranking_cursos["finalizados"] / ranking_cursos["inscritos"] * 100
        ).round(1)

        ranking_docentes = df.groupby("nombre_normalizado").agg(
            cursos_inscritos=("curso", "count"),
            cursos_finalizados=("estatus_final", lambda x: (x == "FINALIZADO").sum())
        ).reset_index().sort_values("cursos_inscritos", ascending=False)

        st.markdown("### Áreas con más docentes inscritos")
        st.dataframe(ranking_areas, use_container_width=True)

        st.markdown("### Cursos con mayor demanda")
        st.dataframe(
            ranking_cursos.sort_values("inscritos", ascending=False),
            use_container_width=True
        )

        st.markdown("### Docentes con más cursos inscritos")
        st.dataframe(
            ranking_docentes,
            use_container_width=True
        )

    # =====================================================
    # DETALLE GENERAL
    # =====================================================

    with tab_detalle:
        st.subheader("Base general de seguimiento")

        st.dataframe(df, use_container_width=True)
