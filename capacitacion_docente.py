import streamlit as st
import pandas as pd
import plotly.express as px
import unicodedata
import re

# =====================================================
# CONFIG
# =====================================================
URL_SEGUIMIENTO = "https://docs.google.com/spreadsheets/d/1Dgu3_UMAYecX-KCxLhHUe_EYiXCF68rvE2XpYwOx9lM/export?format=csv&gid=519739604"

COLORES = {
    "finalizado": "#1D9E75",
    "en_proceso": "#378ADD",
    "no_seac": "#E24B4A",
    "sin_tareas": "#EF9F27",
    "otro": "#888780",
}


# =====================================================
# NORMALIZACIÓN
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


# =====================================================
# DETECCIÓN FLEXIBLE DE COLUMNAS
# =====================================================
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

    df["curso"] = df["curso"].astype(str).str.strip()
    df["nombre_normalizado"] = df["nombre_normalizado"].astype(str).str.strip()

    df["estatus_norm"] = (
        df["estatus_final"]
        .astype(str)
        .str.strip()
        .apply(normalizar_texto)
    )

    df["avance_pct"] = pd.to_numeric(df["avance_pct"], errors="coerce").fillna(0)

    return df


# =====================================================
# CLASIFICACIÓN DE ESTATUS
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
# KPIs
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
        "pct_abandono": round((no_seac + sin_tar) / n * 100, 1) if n else 0,
    }


# =====================================================
# GRÁFICAS
# =====================================================
def grafica_estatus(df):
    df2 = df.copy()
    df2["Estatus"] = df2["estatus_norm"].apply(categoria_estatus)

    cont = df2["Estatus"].value_counts().reset_index()
    cont.columns = ["Estatus", "Total"]

    color_map = {
        "Finalizado": COLORES["finalizado"],
        "En proceso": COLORES["en_proceso"],
        "No aparece en SEAC": COLORES["no_seac"],
        "En SEAC sin tareas": COLORES["sin_tareas"],
        "Otro": COLORES["otro"],
    }

    fig = px.pie(
        cont,
        names="Estatus",
        values="Total",
        color="Estatus",
        color_discrete_map=color_map,
        hole=0.45,
    )

    fig.update_traces(textposition="outside", textinfo="percent+label")
    fig.update_layout(
        showlegend=True,
        margin=dict(t=10, b=10, l=10, r=10),
        height=320,
    )

    return fig


def grafica_barras_area(df):
    df2 = df.copy()
    df2["Estatus"] = df2["estatus_norm"].apply(categoria_estatus)

    resumen = (
        df2.groupby(["area_de_adscripcion", "Estatus"])
        .size()
        .reset_index(name="Total")
    )

    color_map = {
        "Finalizado": COLORES["finalizado"],
        "En proceso": COLORES["en_proceso"],
        "No aparece en SEAC": COLORES["no_seac"],
        "En SEAC sin tareas": COLORES["sin_tareas"],
        "Otro": COLORES["otro"],
    }

    fig = px.bar(
        resumen,
        x="area_de_adscripcion",
        y="Total",
        color="Estatus",
        color_discrete_map=color_map,
        barmode="stack",
        labels={"area_de_adscripcion": "Área", "Total": "Inscripciones"},
    )

    fig.update_layout(
        xaxis_tickangle=-35,
        margin=dict(t=10, b=80, l=10, r=10),
        height=350,
        legend_title_text="Estatus",
    )

    return fig


def grafica_barras_curso(df):
    df2 = df.copy()
    df2["Estatus"] = df2["estatus_norm"].apply(categoria_estatus)

    resumen = (
        df2.groupby(["curso", "Estatus"])
        .size()
        .reset_index(name="Total")
    )

    color_map = {
        "Finalizado": COLORES["finalizado"],
        "En proceso": COLORES["en_proceso"],
        "No aparece en SEAC": COLORES["no_seac"],
        "En SEAC sin tareas": COLORES["sin_tareas"],
        "Otro": COLORES["otro"],
    }

    fig = px.bar(
        resumen,
        x="curso",
        y="Total",
        color="Estatus",
        color_discrete_map=color_map,
        barmode="stack",
        labels={"curso": "Curso", "Total": "Inscripciones"},
    )

    fig.update_layout(
        xaxis_tickangle=-35,
        margin=dict(t=10, b=80, l=10, r=10),
        height=350,
        legend_title_text="Estatus",
    )

    return fig


# =====================================================
# COMPONENTES
# =====================================================
def mostrar_kpis(kpis):
    filas = [
        [
            ("Docentes inscritos", kpis["docentes"]),
            ("Total inscripciones", kpis["inscripciones"]),
            ("Cursos activos", kpis["cursos"]),
            ("Finalizados", kpis["finalizados"]),
        ],
        [
            ("En proceso", kpis["proceso"]),
            ("No aparecen en SEAC", kpis["no_seac"]),
            ("En SEAC sin tareas", kpis["sin_tareas"]),
            ("% Finalización", f"{kpis['pct_fin']}%"),
        ],
    ]

    for fila in filas:
        cols = st.columns(4)
        for i, (label, valor) in enumerate(fila):
            cols[i].metric(label, valor)


def tabla_resumen_docentes(df):
    grp = df.groupby("nombre_normalizado")

    resumen = grp.agg(
        correo=("correo_docente", "first"),
        area=("area_de_adscripcion", "first"),
        inscritos=("curso", "count"),
    ).reset_index()

    resumen["finalizados"] = grp.apply(
        lambda x: es_finalizado(x["estatus_norm"]).sum()
    ).values

    resumen["en_proceso"] = grp.apply(
        lambda x: es_proceso(x["estatus_norm"]).sum()
    ).values

    resumen["no_seac"] = grp.apply(
        lambda x: es_no_seac(x["estatus_norm"]).sum()
    ).values

    resumen["sin_tareas"] = grp.apply(
        lambda x: es_sin_tareas(x["estatus_norm"]).sum()
    ).values

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

    df_filtrado = df[
        df["area_de_adscripcion"].apply(normalizar_texto) == carrera_norm
    ]

    return df_filtrado


# =====================================================
# FUNCIÓN PRINCIPAL
# =====================================================
def render_capacitacion_docente(vista=None, carrera=None):
    st.title("Capacitación Docente")
    st.caption("Seguimiento de participación, avance y finalización de capacitaciones")

    with st.spinner("Cargando datos..."):
        df_raw = cargar_datos()

    df = limpiar(df_raw)

    # Si el usuario es DC, se limita la base a su carrera
    df_permitido = filtrar_por_carrera_si_aplica(df, vista, carrera)

    if df_permitido.empty:
        st.warning(
            "No hay registros de capacitación para la carrera/servicio asignado."
        )
        st.caption(f"Vista: {vista} | Carrera/servicio: {carrera}")
        return

    kpis = calcular_kpis(df_permitido)

    st.subheader("Resumen general")
    mostrar_kpis(kpis)
    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "General",
        "Director de Carrera",
        "Por curso",
        "Rankings",
        "Detalle general",
    ])

    # ==================================================
    # TAB 1 — GENERAL
    # ==================================================
    with tab1:
        st.subheader("Distribución general")

        col_pie, col_area = st.columns([1, 1])

        with col_pie:
            st.markdown("**Inscripciones por estatus**")
            st.plotly_chart(grafica_estatus(df_permitido), use_container_width=True)

        with col_area:
            st.markdown("**Inscripciones por área**")
            st.plotly_chart(grafica_barras_area(df_permitido), use_container_width=True)

        st.markdown("**Inscripciones por curso**")
        st.plotly_chart(grafica_barras_curso(df_permitido), use_container_width=True)

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
            area_sel = st.selectbox("Selecciona área / carrera", areas, key="sel_area")
            df_area = df_permitido[df_permitido["area_de_adscripcion"] == area_sel]

        if df_area.empty:
            st.warning("No hay datos para esta área.")
        else:
            k = calcular_kpis(df_area)
            mostrar_kpis(k)
            st.divider()

            col_g1, col_g2 = st.columns([1, 1])

            with col_g1:
                st.markdown("**Estatus del área**")
                st.plotly_chart(grafica_estatus(df_area), use_container_width=True)

            with col_g2:
                st.markdown("**Avance % por docente**")
                avg_avance = (
                    df_area.groupby("nombre_normalizado")["avance_pct"]
                    .mean()
                    .reset_index()
                    .sort_values("avance_pct", ascending=True)
                )

                fig_av = px.bar(
                    avg_avance,
                    x="avance_pct",
                    y="nombre_normalizado",
                    orientation="h",
                    labels={
                        "avance_pct": "Avance promedio %",
                        "nombre_normalizado": "",
                    },
                    color="avance_pct",
                    color_continuous_scale=["#E24B4A", "#EF9F27", "#1D9E75"],
                    range_color=[0, 100],
                )

                fig_av.update_layout(
                    showlegend=False,
                    coloraxis_showscale=False,
                    margin=dict(t=10, b=10, l=10, r=10),
                    height=max(250, len(avg_avance) * 28),
                )

                st.plotly_chart(fig_av, use_container_width=True)

            st.subheader(f"Docentes — {area_sel}")
            tabla = tabla_resumen_docentes(df_area).drop(columns=["Área"], errors="ignore")

            st.dataframe(
                tabla,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "% Fin.": st.column_config.ProgressColumn(
                        "% Fin.",
                        min_value=0,
                        max_value=100,
                        format="%.1f%%",
                    ),
                },
            )

    # ==================================================
    # TAB 3 — POR CURSO
    # ==================================================
    with tab3:
        cursos = sorted(df_permitido["curso"].dropna().unique())
        curso_sel = st.selectbox("Selecciona curso", cursos, key="sel_curso")

        df_curso = df_permitido[df_permitido["curso"] == curso_sel]

        if df_curso.empty:
            st.warning("No hay datos para este curso.")
        else:
            k = calcular_kpis(df_curso)

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Inscritos", k["inscripciones"])
            c2.metric("Finalizados", k["finalizados"])
            c3.metric("En proceso", k["proceso"])
            c4.metric("No en SEAC", k["no_seac"])
            c5.metric("% Finalización", f"{k['pct_fin']}%")

            st.divider()

            col_p, col_t = st.columns([1, 2])

            with col_p:
                st.markdown("**Distribución por estatus**")
                st.plotly_chart(grafica_estatus(df_curso), use_container_width=True)

            with col_t:
                st.markdown("**Docentes inscritos**")

                tabla_c = df_curso[[
                    "nombre_normalizado",
                    "area_de_adscripcion",
                    "correo_docente",
                    "avance_pct",
                    "estatus_final",
                ]].copy()

                tabla_c = tabla_c.rename(columns={
                    "nombre_normalizado": "Docente",
                    "area_de_adscripcion": "Área",
                    "correo_docente": "Correo",
                    "avance_pct": "Avance %",
                    "estatus_final": "Estatus",
                }).sort_values("Avance %", ascending=False).reset_index(drop=True)

                st.dataframe(
                    tabla_c,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Avance %": st.column_config.ProgressColumn(
                            "Avance %",
                            min_value=0,
                            max_value=100,
                            format="%.1f%%",
                        ),
                    },
                )

    # ==================================================
    # TAB 4 — RANKINGS
    # ==================================================
    with tab4:
        st.subheader("Rankings y análisis comparativo")

        r_col1, r_col2 = st.columns(2)

        with r_col1:
            st.markdown("**Áreas con más docentes inscritos**")
            rank_area_doc = (
                df_permitido.groupby("area_de_adscripcion")["nombre_normalizado"]
                .nunique()
                .reset_index(name="Docentes")
                .sort_values("Docentes", ascending=False)
            )

            fig_r1 = px.bar(
                rank_area_doc,
                x="Docentes",
                y="area_de_adscripcion",
                orientation="h",
                color="Docentes",
                color_continuous_scale=["#B5D4F4", "#534AB7"],
            )

            fig_r1.update_layout(
                coloraxis_showscale=False,
                yaxis_title="",
                xaxis_title="Docentes únicos",
                margin=dict(t=10, b=10, l=10, r=10),
                height=300,
            )

            st.plotly_chart(fig_r1, use_container_width=True)

        with r_col2:
            st.markdown("**Áreas con mayor % de finalización**")

            rank_area_pct = (
                df_permitido.groupby("area_de_adscripcion")
                .apply(
                    lambda grp: round(
                        es_finalizado(grp["estatus_norm"]).sum() / len(grp) * 100,
                        1,
                    ) if len(grp) else 0
                )
                .reset_index(name="% Finalización")
                .sort_values("% Finalización", ascending=False)
            )

            fig_r2 = px.bar(
                rank_area_pct,
                x="% Finalización",
                y="area_de_adscripcion",
                orientation="h",
                color="% Finalización",
                color_continuous_scale=["#9FE1CB", "#0F6E56"],
                range_color=[0, 100],
            )

            fig_r2.update_layout(
                coloraxis_showscale=False,
                yaxis_title="",
                xaxis_title="% Finalización",
                margin=dict(t=10, b=10, l=10, r=10),
                height=300,
            )

            st.plotly_chart(fig_r2, use_container_width=True)

        r_col3, r_col4 = st.columns(2)

        with r_col3:
            st.markdown("**Cursos con mayor demanda**")

            rank_cursos_dem = (
                df_permitido.groupby("curso")
                .size()
                .reset_index(name="Inscritos")
                .sort_values("Inscritos", ascending=False)
                .head(10)
            )

            fig_r3 = px.bar(
                rank_cursos_dem,
                x="Inscritos",
                y="curso",
                orientation="h",
                color="Inscritos",
                color_continuous_scale=["#FAC775", "#854F0B"],
            )

            fig_r3.update_layout(
                coloraxis_showscale=False,
                yaxis_title="",
                xaxis_title="Inscritos",
                margin=dict(t=10, b=10, l=10, r=10),
                height=320,
            )

            st.plotly_chart(fig_r3, use_container_width=True)

        with r_col4:
            st.markdown("**Cursos con mayor abandono / sin avance**")

            rank_cursos_aban = (
                df_permitido.groupby("curso")
                .apply(
                    lambda grp: round(
                        (
                            es_no_seac(grp["estatus_norm"])
                            | es_sin_tareas(grp["estatus_norm"])
                        ).sum() / len(grp) * 100,
                        1,
                    ) if len(grp) else 0
                )
                .reset_index(name="% Abandono")
                .sort_values("% Abandono", ascending=False)
                .head(10)
            )

            fig_r4 = px.bar(
                rank_cursos_aban,
                x="% Abandono",
                y="curso",
                orientation="h",
                color="% Abandono",
                color_continuous_scale=["#F7C1C1", "#A32D2D"],
                range_color=[0, 100],
            )

            fig_r4.update_layout(
                coloraxis_showscale=False,
                yaxis_title="",
                xaxis_title="% Abandono",
                margin=dict(t=10, b=10, l=10, r=10),
                height=320,
            )

            st.plotly_chart(fig_r4, use_container_width=True)

        st.divider()

        r_col5, r_col6 = st.columns(2)

        with r_col5:
            st.markdown("**Docentes con más cursos inscritos**")
            rank_doc_ins = (
                df_permitido.groupby("nombre_normalizado")
                .size()
                .reset_index(name="Cursos inscritos")
                .sort_values("Cursos inscritos", ascending=False)
                .head(15)
            )

            st.dataframe(
                rank_doc_ins,
                use_container_width=True,
                hide_index=True,
            )

        with r_col6:
            st.markdown("**Docentes con más cursos finalizados**")
            rank_doc_fin = (
                df_permitido[es_finalizado(df_permitido["estatus_norm"])]
                .groupby("nombre_normalizado")
                .size()
                .reset_index(name="Cursos finalizados")
                .sort_values("Cursos finalizados", ascending=False)
                .head(15)
            )

            st.dataframe(
                rank_doc_fin,
                use_container_width=True,
                hide_index=True,
            )

    # ==================================================
    # TAB 5 — DETALLE GENERAL
    # ==================================================
    with tab5:
        st.subheader("Tabla completa con filtros")

        f1, f2, f3 = st.columns(3)

        areas_todas = ["Todas"] + sorted(
            df_permitido["area_de_adscripcion"].dropna().unique().tolist()
        )
        cursos_todos = ["Todos"] + sorted(
            df_permitido["curso"].dropna().unique().tolist()
        )
        estatus_todos = ["Todos"] + sorted(
            df_permitido["estatus_final"].dropna().unique().tolist()
        )

        area_f = f1.selectbox("Filtrar por área", areas_todas, key="f_area")
        curso_f = f2.selectbox("Filtrar por curso", cursos_todos, key="f_curso")
        estat_f = f3.selectbox("Filtrar por estatus", estatus_todos, key="f_estat")

        df_det = df_permitido.copy()

        if area_f != "Todas":
            df_det = df_det[df_det["area_de_adscripcion"] == area_f]

        if curso_f != "Todos":
            df_det = df_det[df_det["curso"] == curso_f]

        if estat_f != "Todos":
            df_det = df_det[df_det["estatus_final"] == estat_f]

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
            column_config={
                "Avance %": st.column_config.ProgressColumn(
                    "Avance %",
                    min_value=0,
                    max_value=100,
                    format="%.1f%%",
                ),
            },
        )
