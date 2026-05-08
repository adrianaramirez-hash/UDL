import streamlit as st
import pandas as pd
import plotly.express as px
import unicodedata
import re

SHEET_ID = "1Cl0QQxh0Ls5EqCwzowVVV2bCscok9kXR0m_HdyoEiRw"
URL_SEGUIMIENTO = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet=SEGUIMIENTO_ACTUAL"

COLOR_AZUL = "#2F80ED"
COLOR_VERDE = "#10B981"
COLOR_GRIS = "#374151"

COLOR_MAP = {
    "FINALIZADO": COLOR_VERDE,
    "EN_PROCESO": COLOR_AZUL,
}


def normalizar_texto(texto):
    if not isinstance(texto, str):
        return ""
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    texto = texto.lower().strip()
    texto = re.sub(r"\s+", "_", texto)
    texto = re.sub(r"[^\w]", "", texto)
    return texto


def normalizar_columnas(df):
    df = df.copy()
    df.columns = [normalizar_texto(c) for c in df.columns]
    return df


@st.cache_data(ttl=60)
def cargar_datos():
    df = pd.read_csv(URL_SEGUIMIENTO)
    df = normalizar_columnas(df)
    return df


def preparar_datos(df):
    df = df.copy()

    columnas_necesarias = {
        "curso": "SIN CURSO",
        "edicion_curso": "",
        "fuente_seac": "",
        "fecha_ultimo_corte": "",
        "numero_registro": "",
        "matricula": "",
        "nombre_forms": "",
        "nombre_seac": "",
        "correo_docente": "",
        "tareas_entregadas": 0,
        "tareas_totales": 0,
        "avance_pct": 0,
        "asistencia": "",
        "estatus_final": "EN_PROCESO",
        "requiere_correo": "",
        "tipo_correo": "",
        "ultima_fecha_envio": "",
        "observaciones": "",
    }

    for col, default in columnas_necesarias.items():
        if col not in df.columns:
            df[col] = default

    df["curso"] = df["curso"].astype(str).str.strip().replace("", "SIN CURSO")
    df["nombre_seac"] = df["nombre_seac"].astype(str).str.strip()
    df["nombre_forms"] = df["nombre_forms"].astype(str).str.strip()
    df["correo_docente"] = df["correo_docente"].astype(str).str.strip().str.lower()
    df["matricula"] = df["matricula"].astype(str).str.strip()

    df["nombre_docente"] = df["nombre_seac"]
    df.loc[df["nombre_docente"].isin(["", "nan", "None"]), "nombre_docente"] = df["nombre_forms"]

    df["tareas_entregadas"] = pd.to_numeric(df["tareas_entregadas"], errors="coerce").fillna(0)
    df["tareas_totales"] = pd.to_numeric(df["tareas_totales"], errors="coerce").fillna(0)
    df["avance_pct"] = pd.to_numeric(df["avance_pct"], errors="coerce").fillna(0).clip(0, 100)

    df["estatus_norm"] = df["estatus_final"].apply(normalizar_texto)

    df["estatus_final_limpio"] = df["estatus_norm"].apply(
        lambda x: "FINALIZADO" if "finaliz" in x else "EN_PROCESO"
    )

    df["finalizado"] = df["estatus_final_limpio"] == "FINALIZADO"

    if "area_de_adscripcion" not in df.columns:
        df["area_de_adscripcion"] = "SIN ÁREA"

    df["area_de_adscripcion"] = (
        df["area_de_adscripcion"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace("", "SIN ÁREA")
        .replace("NAN", "SIN ÁREA")
    )

    return df


def filtrar_por_carrera(df, vista, carrera):
    if vista == "Director de carrera" and carrera:
        carrera_norm = normalizar_texto(carrera)
        return df[df["area_de_adscripcion"].apply(normalizar_texto) == carrera_norm]
    return df


def calcular_kpis(df):
    total_inscripciones = len(df)
    docentes_unicos = df["nombre_docente"].nunique()
    cursos = df["curso"].nunique()
    finalizados = int(df["finalizado"].sum())
    en_proceso = total_inscripciones - finalizados

    return {
        "docentes": docentes_unicos,
        "inscripciones": total_inscripciones,
        "cursos": cursos,
        "finalizados": finalizados,
        "en_proceso": en_proceso,
        "pct_finalizados": round(finalizados / total_inscripciones * 100, 1) if total_inscripciones else 0,
        "pct_en_proceso": round(en_proceso / total_inscripciones * 100, 1) if total_inscripciones else 0,
    }


def resumen_por_curso(df):
    tabla = (
        df.groupby("curso")
        .agg(
            docentes_inscritos=("nombre_docente", "nunique"),
            inscripciones=("curso", "count"),
            finalizados=("finalizado", "sum"),
        )
        .reset_index()
    )

    tabla["finalizados"] = tabla["finalizados"].astype(int)
    tabla["en_proceso"] = tabla["inscripciones"] - tabla["finalizados"]
    tabla["pct_finalizados"] = (tabla["finalizados"] / tabla["inscripciones"] * 100).round(1)
    tabla["pct_en_proceso"] = (tabla["en_proceso"] / tabla["inscripciones"] * 100).round(1)

    tabla = tabla.rename(columns={
        "curso": "Curso",
        "docentes_inscritos": "Docentes inscritos",
        "inscripciones": "Inscripciones",
        "finalizados": "Finalizados",
        "en_proceso": "En proceso",
        "pct_finalizados": "% Finalizados",
        "pct_en_proceso": "% En proceso",
    })

    return tabla.sort_values("Inscripciones", ascending=False)


def resumen_docentes(df):
    tabla = (
        df.groupby("nombre_docente")
        .agg(
            correo=("correo_docente", "first"),
            area=("area_de_adscripcion", "first"),
            cursos_inscritos=("curso", "nunique"),
            inscripciones=("curso", "count"),
            cursos_finalizados=("finalizado", "sum"),
        )
        .reset_index()
    )

    tabla["cursos_finalizados"] = tabla["cursos_finalizados"].astype(int)
    tabla["cursos_en_proceso"] = tabla["inscripciones"] - tabla["cursos_finalizados"]
    tabla["pct_finalizacion"] = (tabla["cursos_finalizados"] / tabla["inscripciones"] * 100).round(1)

    tabla = tabla.rename(columns={
        "nombre_docente": "Docente",
        "correo": "Correo",
        "area": "Área",
        "cursos_inscritos": "Cursos inscritos",
        "inscripciones": "Inscripciones",
        "cursos_finalizados": "Cursos finalizados",
        "cursos_en_proceso": "Cursos en proceso",
        "pct_finalizacion": "% Finalización",
    })

    return tabla.sort_values(["Cursos finalizados", "Cursos inscritos"], ascending=False)


def kpi_card(titulo, valor, subtitulo, color):
    st.markdown(
        f"""
        <div style="
            background:#F7F8FB;
            border-radius:12px;
            padding:18px;
            border-left:5px solid {color};
            box-shadow:0 1px 4px rgba(0,0,0,.06);
            min-height:115px;">
            <div style="font-size:.78rem;font-weight:700;color:#6B7280;text-transform:uppercase;">
                {titulo}
            </div>
            <div style="font-size:1.9rem;font-weight:800;color:{color};margin-top:6px;">
                {valor}
            </div>
            <div style="font-size:.82rem;color:#6B7280;">
                {subtitulo}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def mostrar_kpis(kpis):
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi_card("Docentes inscritos", kpis["docentes"], "Docentes únicos", COLOR_AZUL)

    with c2:
        kpi_card("Inscripciones", kpis["inscripciones"], "Total docente-curso", COLOR_AZUL)

    with c3:
        kpi_card("Finalizados", kpis["finalizados"], f'{kpis["pct_finalizados"]}% del total', COLOR_VERDE)

    with c4:
        kpi_card("En proceso", kpis["en_proceso"], f'{kpis["pct_en_proceso"]}% del total', COLOR_GRIS)


def grafica_estatus_curso(tabla):
    datos = tabla.melt(
        id_vars="Curso",
        value_vars=["Finalizados", "En proceso"],
        var_name="Estatus",
        value_name="Cantidad",
    )

    fig = px.bar(
        datos,
        x="Curso",
        y="Cantidad",
        color="Estatus",
        text="Cantidad",
        barmode="group",
        color_discrete_map={
            "Finalizados": COLOR_VERDE,
            "En proceso": COLOR_AZUL,
        },
    )

    fig.update_layout(
        height=420,
        xaxis_tickangle=-25,
        yaxis_title="Docentes / inscripciones",
        margin=dict(t=20, b=100, l=10, r=10),
    )

    return fig


def grafica_pct_finalizacion(tabla):
    fig = px.bar(
        tabla,
        x="Curso",
        y="% Finalizados",
        text="% Finalizados",
        color="% Finalizados",
        color_continuous_scale=["#DBEAFE", COLOR_VERDE],
        range_color=[0, 100],
    )

    fig.update_traces(texttemplate="%{text:.1f}%")
    fig.update_layout(
        height=420,
        coloraxis_showscale=False,
        xaxis_tickangle=-25,
        yaxis_title="% Finalizados",
        margin=dict(t=20, b=100, l=10, r=10),
    )

    return fig


def grafica_inscritos_area(df):
    tabla = (
        df.groupby("area_de_adscripcion")["nombre_docente"]
        .nunique()
        .reset_index(name="Docentes inscritos")
        .rename(columns={"area_de_adscripcion": "Área"})
        .sort_values("Docentes inscritos", ascending=False)
    )

    fig = px.bar(
        tabla,
        x="Área",
        y="Docentes inscritos",
        text="Docentes inscritos",
        color="Docentes inscritos",
        color_continuous_scale=["#DBEAFE", COLOR_AZUL],
    )

    fig.update_layout(
        height=420,
        coloraxis_showscale=False,
        xaxis_tickangle=-35,
        margin=dict(t=20, b=120, l=10, r=10),
    )

    return fig


def render_capacitacion_docente(vista=None, carrera=None):
    st.title("Capacitación Docente")
    st.caption("Seguimiento de inscritos, finalizados y docentes en proceso por curso.")

    df_raw = cargar_datos()
    df = preparar_datos(df_raw)
    df = filtrar_por_carrera(df, vista, carrera)

    if df.empty:
        st.warning("No hay registros disponibles para esta vista.")
        return

    tabla_cursos = resumen_por_curso(df)
    kpis = calcular_kpis(df)

    mostrar_kpis(kpis)

    st.divider()

    st.sidebar.markdown("### Capacitación Docente")
    vista_modulo = st.sidebar.radio(
        "Vista del módulo",
        ["Resumen", "Por curso", "Director de Carrera", "Rankings", "Detalle general"],
        key="nav_capacitacion_docente",
    )

    if vista_modulo == "Resumen":
        st.subheader("Resumen general por curso")

        st.dataframe(
            tabla_cursos,
            use_container_width=True,
            hide_index=True,
            column_config={
                "% Finalizados": st.column_config.ProgressColumn(
                    "% Finalizados", min_value=0, max_value=100, format="%.1f%%"
                ),
                "% En proceso": st.column_config.ProgressColumn(
                    "% En proceso", min_value=0, max_value=100, format="%.1f%%"
                ),
            },
        )

        st.markdown("### Avance por curso")

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(grafica_estatus_curso(tabla_cursos), use_container_width=True)

        with col2:
            st.plotly_chart(grafica_pct_finalizacion(tabla_cursos), use_container_width=True)

        st.markdown("### Docentes inscritos por área")
        st.plotly_chart(grafica_inscritos_area(df), use_container_width=True)

    elif vista_modulo == "Por curso":
        cursos = sorted(df["curso"].dropna().unique())

        curso_sel = st.selectbox("Selecciona curso", cursos)

        df_curso = df[df["curso"] == curso_sel]
        tabla_curso = resumen_por_curso(df_curso)
        kpis_curso = calcular_kpis(df_curso)

        st.subheader(curso_sel)
        mostrar_kpis(kpis_curso)

        st.markdown("### Estado del curso")

        st.dataframe(
            tabla_curso,
            use_container_width=True,
            hide_index=True,
            column_config={
                "% Finalizados": st.column_config.ProgressColumn(
                    "% Finalizados", min_value=0, max_value=100, format="%.1f%%"
                ),
                "% En proceso": st.column_config.ProgressColumn(
                    "% En proceso", min_value=0, max_value=100, format="%.1f%%"
                ),
            },
        )

        st.markdown("### Docentes inscritos en el curso")

        tabla_docentes = df_curso[[
            "nombre_docente",
            "matricula",
            "correo_docente",
            "area_de_adscripcion",
            "tareas_entregadas",
            "tareas_totales",
            "avance_pct",
            "estatus_final",
        ]].rename(columns={
            "nombre_docente": "Docente",
            "matricula": "Matrícula",
            "correo_docente": "Correo",
            "area_de_adscripcion": "Área",
            "tareas_entregadas": "Tareas entregadas",
            "tareas_totales": "Tareas totales",
            "avance_pct": "Avance %",
            "estatus_final": "Estatus",
        })

        st.dataframe(
            tabla_docentes,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Avance %": st.column_config.ProgressColumn(
                    "Avance %", min_value=0, max_value=100, format="%.1f%%"
                )
            },
        )

    elif vista_modulo == "Director de Carrera":
        areas = sorted(df["area_de_adscripcion"].dropna().unique())

        if vista == "Director de carrera" and carrera:
            area_sel = carrera
        else:
            area_sel = st.selectbox("Selecciona área / carrera", areas)

        df_area = df[df["area_de_adscripcion"].apply(normalizar_texto) == normalizar_texto(area_sel)]

        st.subheader(f"Seguimiento por carrera / área: {area_sel}")

        mostrar_kpis(calcular_kpis(df_area))

        tabla_area = resumen_por_curso(df_area)

        st.dataframe(
            tabla_area,
            use_container_width=True,
            hide_index=True,
            column_config={
                "% Finalizados": st.column_config.ProgressColumn(
                    "% Finalizados", min_value=0, max_value=100, format="%.1f%%"
                ),
                "% En proceso": st.column_config.ProgressColumn(
                    "% En proceso", min_value=0, max_value=100, format="%.1f%%"
                ),
            },
        )

        st.plotly_chart(grafica_pct_finalizacion(tabla_area), use_container_width=True)

        st.markdown("### Docentes de la carrera / área")
        st.dataframe(
            resumen_docentes(df_area),
            use_container_width=True,
            hide_index=True,
            column_config={
                "% Finalización": st.column_config.ProgressColumn(
                    "% Finalización", min_value=0, max_value=100, format="%.1f%%"
                )
            },
        )

    elif vista_modulo == "Rankings":
        st.subheader("Rankings")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Cursos con mayor demanda")
            st.dataframe(
                tabla_cursos[["Curso", "Docentes inscritos", "Inscripciones", "Finalizados", "% Finalizados"]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "% Finalizados": st.column_config.ProgressColumn(
                        "% Finalizados", min_value=0, max_value=100, format="%.1f%%"
                    )
                },
            )

        with col2:
            st.markdown("### Áreas con más inscritos")
            tabla_areas = (
                df.groupby("area_de_adscripcion")["nombre_docente"]
                .nunique()
                .reset_index(name="Docentes inscritos")
                .rename(columns={"area_de_adscripcion": "Área"})
                .sort_values("Docentes inscritos", ascending=False)
            )
            st.dataframe(tabla_areas, use_container_width=True, hide_index=True)

        st.markdown("### Docentes con más cursos finalizados")
        st.dataframe(
            resumen_docentes(df),
            use_container_width=True,
            hide_index=True,
            column_config={
                "% Finalización": st.column_config.ProgressColumn(
                    "% Finalización", min_value=0, max_value=100, format="%.1f%%"
                )
            },
        )

    elif vista_modulo == "Detalle general":
        st.subheader("Detalle general")

        col1, col2, col3 = st.columns(3)

        areas = ["Todas"] + sorted(df["area_de_adscripcion"].dropna().unique())
        cursos = ["Todos"] + sorted(df["curso"].dropna().unique())
        estatus = ["Todos"] + sorted(df["estatus_final"].dropna().unique())

        area_sel = col1.selectbox("Área", areas)
        curso_sel = col2.selectbox("Curso", cursos)
        estatus_sel = col3.selectbox("Estatus", estatus)

        df_detalle = df.copy()

        if area_sel != "Todas":
            df_detalle = df_detalle[df_detalle["area_de_adscripcion"] == area_sel]

        if curso_sel != "Todos":
            df_detalle = df_detalle[df_detalle["curso"] == curso_sel]

        if estatus_sel != "Todos":
            df_detalle = df_detalle[df_detalle["estatus_final"] == estatus_sel]

        tabla_detalle = df_detalle[[
            "curso",
            "matricula",
            "nombre_docente",
            "correo_docente",
            "area_de_adscripcion",
            "tareas_entregadas",
            "tareas_totales",
            "avance_pct",
            "estatus_final",
            "requiere_correo",
            "tipo_correo",
            "observaciones",
        ]].rename(columns={
            "curso": "Curso",
            "matricula": "Matrícula",
            "nombre_docente": "Docente",
            "correo_docente": "Correo",
            "area_de_adscripcion": "Área",
            "tareas_entregadas": "Tareas entregadas",
            "tareas_totales": "Tareas totales",
            "avance_pct": "Avance %",
            "estatus_final": "Estatus",
            "requiere_correo": "Requiere correo",
            "tipo_correo": "Tipo correo",
            "observaciones": "Observaciones",
        })

        st.caption(f"{len(tabla_detalle)} registros encontrados")

        st.dataframe(
            tabla_detalle,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Avance %": st.column_config.ProgressColumn(
                    "Avance %", min_value=0, max_value=100, format="%.1f%%"
                )
            },
        )
