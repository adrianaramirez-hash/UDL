import streamlit as st
import pandas as pd
import plotly.express as px
import unicodedata
import re

# =====================================================
# CONFIG
# =====================================================
URL_SEGUIMIENTO = "https://docs.google.com/spreadsheets/d/1Dgu3_UMAYecX-KCxLhHUe_EYiXCF68rvE2XpYwOx9lM/export?format=csv&gid=519739604"

COLOR_AZUL = "#2F80ED"
COLOR_VERDE = "#10B981"
COLOR_ROJO = "#D7263D"
COLOR_NARANJA = "#F97316"
COLOR_AMARILLO = "#EAB308"
COLOR_GRIS = "#374151"
COLOR_MORADO = "#6C63FF"

COLOR_MAP = {
    "Finalizado": COLOR_VERDE,
    "En proceso": COLOR_AZUL,
    "No aparece en SEAC": COLOR_ROJO,
    "En SEAC sin tareas": COLOR_NARANJA,
    "Otro": COLOR_GRIS,
}


# =====================================================
# UTILIDADES
# =====================================================
def normalizar_texto(texto):
    if not isinstance(texto, str):
        return ""
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    texto = texto.lower().strip()
    texto = re.sub(r"\s+", "_", texto)
    texto = re.sub(r"[^\w]", "", texto)
    return texto


@st.cache_data(ttl=300)
def cargar_datos():
    df = pd.read_csv(URL_SEGUIMIENTO)
    df.columns = [normalizar_texto(c) for c in df.columns]
    return df


ALIAS_AREA = [
    "area_de_adscripcion",
    "area_adscripcion",
    "adscripcion",
    "area",
    "carrera",
    "departamento",
]

ALIAS_NOMBRE = [
    "nombre_normalizado",
    "nombre_forms",
    "nombre_seac",
    "nombre",
    "docente",
    "nombre_docente",
]


def detectar_columna(df, aliases):
    for alias in aliases:
        cand = normalizar_texto(alias)
        for col in df.columns:
            if normalizar_texto(col) == cand:
                return col

    for alias in aliases:
        for col in df.columns:
            if normalizar_texto(alias) in normalizar_texto(col):
                return col

    return None


def limpiar(df):
    df = df.copy()

    col_area = detectar_columna(df, ALIAS_AREA)
    col_nombre = detectar_columna(df, ALIAS_NOMBRE)

    if col_area and col_area != "area_de_adscripcion":
        df = df.rename(columns={col_area: "area_de_adscripcion"})

    if col_nombre and col_nombre != "nombre_normalizado":
        df = df.rename(columns={col_nombre: "nombre_normalizado"})

    defaults = {
        "area_de_adscripcion": "SIN ÁREA",
        "curso": "SIN CURSO",
        "estatus_final": "SIN ESTATUS",
        "nombre_normalizado": "SIN NOMBRE",
        "correo_docente": "",
        "avance_pct": 0,
        "tareas_entregadas": 0,
        "tareas_totales": 0,
        "matricula": "",
        "requiere_correo": "",
        "tipo_correo": "",
        "fecha_ultimo_corte": "",
        "observaciones": "",
    }

    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    df["area_de_adscripcion"] = (
        df["area_de_adscripcion"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace("NAN", "SIN ÁREA")
        .replace("", "SIN ÁREA")
    )

    df["curso"] = (
        df["curso"]
        .astype(str)
        .str.strip()
        .replace("nan", "SIN CURSO")
        .replace("", "SIN CURSO")
    )

    df["nombre_normalizado"] = (
        df["nombre_normalizado"]
        .astype(str)
        .str.strip()
        .replace("nan", "SIN NOMBRE")
        .replace("", "SIN NOMBRE")
    )

    df["estatus_final"] = (
        df["estatus_final"]
        .astype(str)
        .str.strip()
        .replace("nan", "SIN ESTATUS")
        .replace("", "SIN ESTATUS")
    )

    df["estatus_norm"] = df["estatus_final"].apply(normalizar_texto)

    df["avance_pct"] = pd.to_numeric(df["avance_pct"], errors="coerce").fillna(0)
    df["avance_pct"] = df["avance_pct"].clip(lower=0, upper=100)

    return df


# =====================================================
# CLASIFICACIÓN
# =====================================================
def es_finalizado(serie):
    return serie.str.contains("finaliz", case=False, na=False)


def es_proceso(serie):
    return serie.str.contains("proceso|en_proceso", case=False, na=False)


def es_no_seac(serie):
    return serie.str.contains("no_aparece|no_ingres|sin_ingreso", case=False, na=False)


def es_sin_tareas(serie):
    return serie.str.contains("sin_tarea|seac_sin", case=False, na=False)


def categoria_estatus(valor):
    v = normalizar_texto(str(valor))

    if "finaliz" in v:
        return "Finalizado"
    if "proceso" in v:
        return "En proceso"
    if "sin_tarea" in v or "seac_sin" in v:
        return "En SEAC sin tareas"
    if "no_aparece" in v or "no_ingres" in v or "sin_ingreso" in v:
        return "No aparece en SEAC"

    return "Otro"


# =====================================================
# CÁLCULOS
# =====================================================
def calcular_kpis(df):
    n = len(df)

    fin = es_finalizado(df["estatus_norm"]).sum()
    proc = es_proceso(df["estatus_norm"]).sum()
    no_seac = es_no_seac(df["estatus_norm"]).sum()
    sin_tar = es_sin_tareas(df["estatus_norm"]).sum()

    return {
        "docentes": df["nombre_normalizado"].nunique(),
        "inscripciones": n,
        "cursos": df["curso"].nunique(),
        "finalizados": int(fin),
        "proceso": int(proc),
        "no_seac": int(no_seac),
        "sin_tareas": int(sin_tar),
        "pct_fin": round(fin / n * 100, 1) if n else 0,
        "pct_sin_avance": round((no_seac + sin_tar) / n * 100, 1) if n else 0,
    }


def resumen_por_curso(df):
    registros = []

    for curso, grp in df.groupby("curso"):
        n = len(grp)
        fin = es_finalizado(grp["estatus_norm"]).sum()
        proc = es_proceso(grp["estatus_norm"]).sum()
        no_seac = es_no_seac(grp["estatus_norm"]).sum()
        sin_tareas = es_sin_tareas(grp["estatus_norm"]).sum()

        registros.append({
            "Curso": curso,
            "Docentes inscritos": grp["nombre_normalizado"].nunique(),
            "Inscripciones": n,
            "Finalizados": int(fin),
            "En proceso": int(proc),
            "No en SEAC": int(no_seac),
            "Sin tareas": int(sin_tareas),
            "% Finalización": round(fin / n * 100, 1) if n else 0,
            "% Sin avance": round((no_seac + sin_tareas) / n * 100, 1) if n else 0,
        })

    return pd.DataFrame(registros).sort_values("Inscripciones", ascending=False)


def tabla_resumen_docentes(df):
    grp = df.groupby("nombre_normalizado")

    resumen = grp.agg(
        correo=("correo_docente", "first"),
        area=("area_de_adscripcion", "first"),
        inscritos=("curso", "count"),
    ).reset_index()

    resumen["finalizados"] = grp.apply(lambda x: es_finalizado(x["estatus_norm"]).sum()).values
    resumen["en_proceso"] = grp.apply(lambda x: es_proceso(x["estatus_norm"]).sum()).values
    resumen["no_seac"] = grp.apply(lambda x: es_no_seac(x["estatus_norm"]).sum()).values
    resumen["sin_tareas"] = grp.apply(lambda x: es_sin_tareas(x["estatus_norm"]).sum()).values

    resumen["pct_fin"] = (
        resumen["finalizados"] / resumen["inscritos"] * 100
    ).round(1).where(resumen["inscritos"] > 0, 0)

    resumen = resumen.rename(columns={
        "nombre_normalizado": "Docente",
        "correo": "Correo",
        "area": "Área",
        "inscritos": "Inscritos",
        "finalizados": "Finalizados",
        "en_proceso": "En proceso",
        "no_seac": "No en SEAC",
        "sin_tareas": "Sin tareas",
        "pct_fin": "% Fin.",
    })

    return resumen.sort_values("% Fin.", ascending=False).reset_index(drop=True)


def filtrar_por_carrera_si_aplica(df, vista, carrera):
    if vista != "Director de carrera" or not carrera:
        return df

    carrera_norm = normalizar_texto(carrera)

    return df[
        df["area_de_adscripcion"].apply(normalizar_texto) == carrera_norm
    ]


# =====================================================
# VISUAL
# =====================================================
def aplicar_estilos():
    st.markdown(
        """
        <style>
        .kpi-card {
            background: #F7F8FB;
            border-radius: 12px;
            padding: 18px 18px 14px 18px;
            min-height: 118px;
            border-left: 5px solid #2F80ED;
            box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        }
        .kpi-title {
            font-size: 0.78rem;
            font-weight: 700;
            color: #6B7280;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .kpi-value {
            font-size: 1.85rem;
            font-weight: 800;
            color: #111827;
            margin-top: 6px;
        }
        .kpi-caption {
            font-size: 0.82rem;
            color: #6B7280;
            margin-top: 2px;
        }
        .course-pill {
            display: inline-block;
            padding: 7px 12px;
            margin: 4px 6px 4px 0;
            background: #EEF2FF;
            color: #3730A3;
            border-radius: 999px;
            font-size: 0.86rem;
            border: 1px solid #C7D2FE;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label, value, caption="", color=COLOR_AZUL):
    st.markdown(
        f"""
        <div class="kpi-card" style="border-left-color:{color};">
            <div class="kpi-title">{label}</div>
            <div class="kpi-value" style="color:{color};">{value}</div>
            <div class="kpi-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def mostrar_kpis(kpis):
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi_card("Docentes inscritos", kpis["docentes"], "Docentes únicos", COLOR_AZUL)
    with c2:
        kpi_card("Inscripciones", kpis["inscripciones"], "Registros de capacitación", COLOR_VERDE)
    with c3:
        kpi_card("Cursos activos", kpis["cursos"], "Cursos en seguimiento", COLOR_GRIS)
    with c4:
        kpi_card("% Finalización", f"{kpis['pct_fin']}%", "Cursos concluidos", COLOR_VERDE)

    c5, c6, c7, c8 = st.columns(4)

    with c5:
        kpi_card("Finalizados", kpis["finalizados"], "Concluyeron actividades", COLOR_VERDE)
    with c6:
        kpi_card("En proceso", kpis["proceso"], "Avance parcial", COLOR_AZUL)
    with c7:
        kpi_card("No aparecen en SEAC", kpis["no_seac"], "Alerta de ingreso", COLOR_ROJO)
    with c8:
        kpi_card("En SEAC sin tareas", kpis["sin_tareas"], "Requiere seguimiento", COLOR_NARANJA)


def mostrar_cursos_incluidos(df):
    cursos = sorted([c for c in df["curso"].dropna().unique().tolist() if c and c != "SIN CURSO"])

    st.markdown("### Cursos incluidos en el seguimiento actual")

    if not cursos:
        st.info("No se detectaron cursos en la base.")
        return

    pills = "".join([f'<span class="course-pill">{i+1}. {curso}</span>' for i, curso in enumerate(cursos)])
    st.markdown(pills, unsafe_allow_html=True)


# =====================================================
# GRÁFICAS
# =====================================================
def grafica_estatus(df):
    df2 = df.copy()
    df2["Estatus"] = df2["estatus_norm"].apply(categoria_estatus)

    cont = df2["Estatus"].value_counts().reset_index()
    cont.columns = ["Estatus", "Total"]

    fig = px.pie(
        cont,
        names="Estatus",
        values="Total",
        color="Estatus",
        color_discrete_map=COLOR_MAP,
        hole=0.45,
    )

    fig.update_traces(textposition="outside", textinfo="percent+label")
    fig.update_layout(
        showlegend=True,
        margin=dict(t=10, b=10, l=10, r=10),
        height=320,
    )

    return fig


def grafica_area(df):
    df2 = df.copy()
    df2["Estatus"] = df2["estatus_norm"].apply(categoria_estatus)

    resumen = (
        df2.groupby(["area_de_adscripcion", "Estatus"])
        .size()
        .reset_index(name="Total")
    )

    fig = px.bar(
        resumen,
        x="area_de_adscripcion",
        y="Total",
        color="Estatus",
        color_discrete_map=COLOR_MAP,
        barmode="stack",
        labels={
            "area_de_adscripcion": "Área de adscripción",
            "Total": "Inscripciones",
        },
    )

    fig.update_layout(
        xaxis_tickangle=-35,
        height=380,
        margin=dict(t=10, b=90, l=10, r=10),
        legend_title_text="Estatus",
    )

    return fig


def grafica_curso(df):
    resumen = resumen_por_curso(df)

    fig = px.bar(
        resumen,
        x="Inscripciones",
        y="Curso",
        orientation="h",
        color="% Finalización",
        color_continuous_scale=["#FEE2E2", "#BFDBFE", "#10B981"],
        range_color=[0, 100],
        text="Inscripciones",
    )

    fig.update_layout(
        height=max(320, len(resumen) * 55),
        coloraxis_colorbar_title="% Finalización",
        yaxis_title="",
        xaxis_title="Inscripciones",
        margin=dict(t=10, b=10, l=10, r=10),
    )

    return fig


# =====================================================
# FUNCIÓN PRINCIPAL
# =====================================================
def render_capacitacion_docente(vista=None, carrera=None):
    aplicar_estilos()

    st.title("Capacitación Docente")
    st.caption("Seguimiento de participación, avance y finalización de capacitaciones docentes.")

    with st.spinner("Cargando datos de capacitación..."):
        df_raw = cargar_datos()

    df = limpiar(df_raw)
    df_permitido = filtrar_por_carrera_si_aplica(df, vista, carrera)

    if df_permitido.empty:
        st.warning("No hay registros de capacitación para la carrera/servicio asignado.")
        st.caption(f"Vista: {vista} | Carrera/servicio: {carrera}")
        return

    kpis = calcular_kpis(df_permitido)

    mostrar_kpis(kpis)
    st.divider()

    mostrar_cursos_incluidos(df_permitido)
    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Resumen",
        "👥 Director de Carrera",
        "📘 Por curso",
        "⭐ Rankings",
        "📋 Detalle general",
    ])

    # ==================================================
    # TAB 1 — RESUMEN
    # ==================================================
    with tab1:
        st.subheader("Resumen por curso de capacitación")

        tabla_cursos = resumen_por_curso(df_permitido)

        st.dataframe(
            tabla_cursos,
            use_container_width=True,
            hide_index=True,
            key="tabla_resumen_cursos",
            column_config={
                "% Finalización": st.column_config.ProgressColumn(
                    "% Finalización", min_value=0, max_value=100, format="%.1f%%"
                ),
                "% Sin avance": st.column_config.ProgressColumn(
                    "% Sin avance", min_value=0, max_value=100, format="%.1f%%"
                ),
            },
        )

        st.markdown("### Visualización general")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("**Distribución general por estatus**")
            st.plotly_chart(
                grafica_estatus(df_permitido),
                use_container_width=True,
                key="grafica_estatus_resumen",
            )

        with col2:
            st.markdown("**Avance por curso de capacitación**")
            st.plotly_chart(
                grafica_curso(df_permitido),
                use_container_width=True,
                key="grafica_curso_resumen",
            )

        st.markdown("**Inscripciones por área de adscripción y estatus**")
        st.plotly_chart(
            grafica_area(df_permitido),
            use_container_width=True,
            key="grafica_area_resumen",
        )

    # ==================================================
    # TAB 2 — DIRECTOR DE CARRERA
    # ==================================================
    with tab2:
        if vista == "Director de carrera" and carrera:
            area_sel = carrera
            st.info(f"Vista filtrada para: **{area_sel}**")
            df_area = df_permitido.copy()
        else:
            areas = sorted(df_permitido["area_de_adscripcion"].dropna().unique())
            area_sel = st.selectbox("Selecciona área / carrera", areas, key="sel_area_dc")
            df_area = df_permitido[df_permitido["area_de_adscripcion"] == area_sel]

        if df_area.empty:
            st.warning("No hay datos para esta área.")
        else:
            st.subheader(f"Seguimiento de capacitación — {area_sel}")

            mostrar_kpis(calcular_kpis(df_area))
            st.divider()

            tabla_doc = tabla_resumen_docentes(df_area).drop(columns=["Área"], errors="ignore")

            st.dataframe(
                tabla_doc,
                use_container_width=True,
                hide_index=True,
                key="tabla_docentes_area",
                column_config={
                    "% Fin.": st.column_config.ProgressColumn(
                        "% Fin.", min_value=0, max_value=100, format="%.1f%%"
                    ),
                },
            )

            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("**Distribución por estatus del área**")
                st.plotly_chart(
                    grafica_estatus(df_area),
                    use_container_width=True,
                    key="grafica_estatus_area_dc",
                )

            with col2:
                st.markdown("**Cursos tomados por docente**")
                cursos_doc = (
                    df_area.groupby("nombre_normalizado")
                    .size()
                    .reset_index(name="Cursos inscritos")
                    .sort_values("Cursos inscritos", ascending=True)
                )

                fig_doc = px.bar(
                    cursos_doc,
                    x="Cursos inscritos",
                    y="nombre_normalizado",
                    orientation="h",
                    color="Cursos inscritos",
                    color_continuous_scale=["#DBEAFE", COLOR_AZUL],
                )

                fig_doc.update_layout(
                    height=max(300, len(cursos_doc) * 30),
                    coloraxis_showscale=False,
                    yaxis_title="",
                    xaxis_title="Cursos inscritos",
                    margin=dict(t=10, b=10, l=10, r=10),
                )

                st.plotly_chart(
                    fig_doc,
                    use_container_width=True,
                    key="grafica_cursos_docente_dc",
                )

    # ==================================================
    # TAB 3 — POR CURSO
    # ==================================================
    with tab3:
        st.subheader("Análisis individual por curso")

        cursos = sorted(df_permitido["curso"].dropna().unique())
        curso_sel = st.selectbox("Selecciona curso", cursos, key="sel_curso_capacitacion")

        df_curso = df_permitido[df_permitido["curso"] == curso_sel]

        if df_curso.empty:
            st.warning("No hay datos para este curso.")
        else:
            st.markdown(f"### {curso_sel}")

            mostrar_kpis(calcular_kpis(df_curso))
            st.divider()

            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("**Estatus del curso**")
                st.plotly_chart(
                    grafica_estatus(df_curso),
                    use_container_width=True,
                    key="grafica_estatus_curso",
                )

            with col2:
                st.markdown("**Docentes inscritos en el curso**")

                cols = [
                    "nombre_normalizado",
                    "area_de_adscripcion",
                    "correo_docente",
                    "avance_pct",
                    "estatus_final",
                ]

                tabla = df_curso[cols].copy()
                tabla = tabla.rename(columns={
                    "nombre_normalizado": "Docente",
                    "area_de_adscripcion": "Área",
                    "correo_docente": "Correo",
                    "avance_pct": "Avance %",
                    "estatus_final": "Estatus",
                }).sort_values("Avance %", ascending=False).reset_index(drop=True)

                st.dataframe(
                    tabla,
                    use_container_width=True,
                    hide_index=True,
                    key="tabla_docentes_curso",
                    column_config={
                        "Avance %": st.column_config.ProgressColumn(
                            "Avance %", min_value=0, max_value=100, format="%.1f%%"
                        ),
                    },
                )

    # ==================================================
    # TAB 4 — RANKINGS
    # ==================================================
    with tab4:
        st.subheader("Rankings y análisis comparativo")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Áreas con más docentes inscritos**")

            r1 = (
                df_permitido.groupby("area_de_adscripcion")["nombre_normalizado"]
                .nunique()
                .reset_index(name="Docentes")
                .sort_values("Docentes", ascending=False)
            )

            st.dataframe(r1, use_container_width=True, hide_index=True, key="rank_area_docentes")

        with col2:
            st.markdown("**Cursos con mayor demanda**")

            r2 = (
                df_permitido.groupby("curso")
                .size()
                .reset_index(name="Inscripciones")
                .sort_values("Inscripciones", ascending=False)
            )

            st.dataframe(r2, use_container_width=True, hide_index=True, key="rank_cursos_demanda")

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("**Docentes con más cursos inscritos**")

            r3 = (
                df_permitido.groupby("nombre_normalizado")
                .size()
                .reset_index(name="Cursos inscritos")
                .sort_values("Cursos inscritos", ascending=False)
                .head(15)
            )

            st.dataframe(r3, use_container_width=True, hide_index=True, key="rank_docentes_inscritos")

        with col4:
            st.markdown("**Docentes con más cursos finalizados**")

            r4 = (
                df_permitido[es_finalizado(df_permitido["estatus_norm"])]
                .groupby("nombre_normalizado")
                .size()
                .reset_index(name="Cursos finalizados")
                .sort_values("Cursos finalizados", ascending=False)
                .head(15)
            )

            st.dataframe(r4, use_container_width=True, hide_index=True, key="rank_docentes_finalizados")

    # ==================================================
    # TAB 5 — DETALLE
    # ==================================================
    with tab5:
        st.subheader("Tabla completa con filtros")

        f1, f2, f3 = st.columns(3)

        areas_todas = ["Todas"] + sorted(df_permitido["area_de_adscripcion"].dropna().unique().tolist())
        cursos_todos = ["Todos"] + sorted(df_permitido["curso"].dropna().unique().tolist())
        estatus_todos = ["Todos"] + sorted(df_permitido["estatus_final"].dropna().unique().tolist())

        area_f = f1.selectbox("Área", areas_todas, key="f_area_detalle")
        curso_f = f2.selectbox("Curso", cursos_todos, key="f_curso_detalle")
        estatus_f = f3.selectbox("Estatus", estatus_todos, key="f_estatus_detalle")

        df_det = df_permitido.copy()

        if area_f != "Todas":
            df_det = df_det[df_det["area_de_adscripcion"] == area_f]
        if curso_f != "Todos":
            df_det = df_det[df_det["curso"] == curso_f]
        if estatus_f != "Todos":
            df_det = df_det[df_det["estatus_final"] == estatus_f]

        cols_mostrar = [
            c for c in [
                "nombre_normalizado",
                "matricula",
                "correo_docente",
                "area_de_adscripcion",
                "curso",
                "avance_pct",
                "tareas_entregadas",
                "tareas_totales",
                "estatus_final",
                "fecha_ultimo_corte",
                "observaciones",
            ]
            if c in df_det.columns
        ]

        tabla_det = df_det[cols_mostrar].rename(columns={
            "nombre_normalizado": "Docente",
            "matricula": "Matrícula",
            "correo_docente": "Correo",
            "area_de_adscripcion": "Área",
            "curso": "Curso",
            "avance_pct": "Avance %",
            "tareas_entregadas": "T. Entregadas",
            "tareas_totales": "T. Totales",
            "estatus_final": "Estatus",
            "fecha_ultimo_corte": "Último corte",
            "observaciones": "Observaciones",
        }).reset_index(drop=True)

        st.caption(f"{len(tabla_det)} registros encontrados")

        st.dataframe(
            tabla_det,
            use_container_width=True,
            hide_index=True,
            key="tabla_detalle_capacitacion",
            column_config={
                "Avance %": st.column_config.ProgressColumn(
                    "Avance %", min_value=0, max_value=100, format="%.1f%%"
                ),
            },
        )
