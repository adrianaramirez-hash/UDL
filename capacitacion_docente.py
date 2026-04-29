import streamlit as st
import pandas as pd
import plotly.express as px
import unicodedata
import re

URL_SEGUIMIENTO = "https://docs.google.com/spreadsheets/d/1Dgu3_UMAYecX-KCxLhHUe_EYiXCF68rvE2XpYwOx9lM/export?format=csv&gid=519739604"

COLOR_AZUL = "#2F80ED"
COLOR_VERDE = "#10B981"
COLOR_ROJO = "#D7263D"
COLOR_NARANJA = "#F97316"
COLOR_GRIS = "#374151"

COLOR_MAP = {
    "Finalizado": COLOR_VERDE,
    "En proceso": COLOR_AZUL,
    "No aparece en SEAC": COLOR_ROJO,
    "En SEAC sin tareas": COLOR_NARANJA,
    "Otro": COLOR_GRIS,
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


def normalizar_valor(texto):
    if pd.isna(texto):
        return ""
    return str(texto).strip()


@st.cache_data(ttl=300)
def cargar_datos():
    df_raw = pd.read_csv(URL_SEGUIMIENTO)
    columnas_originales = list(df_raw.columns)
    columnas_norm = [normalizar_texto(c) for c in columnas_originales]

    mapa_original = dict(zip(columnas_norm, columnas_originales))

    df_raw.columns = columnas_norm
    df_raw.attrs["mapa_original"] = mapa_original

    return df_raw


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
    "nombre_completo",
]

ALIAS_CORREO = [
    "correo_docente",
    "correo",
    "correo_electronico",
    "direccion_de_correo_electronico",
    "correo_seac",
]

ALIAS_ESTATUS = [
    "estatus_final",
    "estatus",
    "estado",
    "estatus_seac",
]

ALIAS_AVANCE = [
    "avance_pct",
    "avance",
    "porcentaje_avance",
]

ALIAS_TAREAS = [
    "tareas",
    "actividades",
]

ALIAS_CURSO = [
    "curso",
    "nombre_curso",
    "capacitacion",
    "taller",
]


def detectar_columna(df, aliases):
    for alias in aliases:
        cand = normalizar_texto(alias)
        for col in df.columns:
            if normalizar_texto(col) == cand:
                return col

    for alias in aliases:
        cand = normalizar_texto(alias)
        for col in df.columns:
            if cand in normalizar_texto(col):
                return col

    return None


def extraer_curso_desde_columna(nombre_columna_original):
    texto = str(nombre_columna_original)

    m = re.search(r"\[(.*?)\]", texto)
    if m:
        return m.group(1).strip()

    texto = texto.replace("Selecciona el taller al que deseas inscribirte", "")
    texto = texto.replace("en caso de no no desees inscribirte a algún curso selecciona NO INSCRIBIRME", "")
    texto = texto.replace("en caso de no desees inscribirte a algún curso selecciona NO INSCRIBIRME", "")
    texto = texto.replace(",", "")
    texto = texto.strip()

    return texto if texto else "SIN CURSO"


def construir_curso_si_no_existe(df):
    """
    Si no existe una columna curso clara, intenta construirla desde columnas de Forms.
    Ejemplo:
    - columnas que dicen 'Selecciona el taller... [Nombre del curso]'
    - valores tipo INSCRIBIRME / NO INSCRIBIRME
    """
    df = df.copy()

    if "curso" in df.columns and df["curso"].notna().any():
        df["curso"] = df["curso"].astype(str).str.strip()
        return df

    mapa_original = df.attrs.get("mapa_original", {})

    posibles_cols_curso = []
    for col in df.columns:
        col_norm = normalizar_texto(col)
        col_original = mapa_original.get(col, col)

        if (
            "taller" in col_norm
            or "curso" in col_norm
            or "capacitacion" in col_norm
            or "inscribirte" in normalizar_texto(str(col_original))
        ):
            posibles_cols_curso.append(col)

    posibles_cols_curso = [
        c for c in posibles_cols_curso
        if c not in ["curso", "nombre_curso", "estatus_final", "tipo_correo"]
    ]

    if not posibles_cols_curso:
        df["curso"] = "SIN CURSO"
        return df

    registros = []

    columnas_base = [
        c for c in df.columns
        if c not in posibles_cols_curso
    ]

    for _, row in df.iterrows():
        for col in posibles_cols_curso:
            valor = normalizar_valor(row.get(col, ""))
            valor_norm = normalizar_texto(valor)

            if not valor_norm:
                continue

            if "no_inscribirme" in valor_norm or "no_me_inscribo" in valor_norm:
                continue

            if valor_norm in ["nan", "sin_dato", "no"]:
                continue

            nuevo = row[columnas_base].to_dict()

            col_original = mapa_original.get(col, col)
            curso_desde_columna = extraer_curso_desde_columna(col_original)

            if "inscrib" in valor_norm or valor_norm in ["si", "sí"]:
                nuevo["curso"] = curso_desde_columna
            else:
                nuevo["curso"] = valor

            registros.append(nuevo)

    if registros:
        return pd.DataFrame(registros)

    df["curso"] = "SIN CURSO"
    return df


def limpiar(df):
    df = df.copy()

    col_area = detectar_columna(df, ALIAS_AREA)
    col_nombre = detectar_columna(df, ALIAS_NOMBRE)
    col_correo = detectar_columna(df, ALIAS_CORREO)
    col_estatus = detectar_columna(df, ALIAS_ESTATUS)
    col_avance = detectar_columna(df, ALIAS_AVANCE)

    if col_area and col_area != "area_de_adscripcion":
        df = df.rename(columns={col_area: "area_de_adscripcion"})

    if col_nombre and col_nombre != "nombre_normalizado":
        df = df.rename(columns={col_nombre: "nombre_normalizado"})

    if col_correo and col_correo != "correo_docente":
        df = df.rename(columns={col_correo: "correo_docente"})

    if col_estatus and col_estatus != "estatus_final":
        df = df.rename(columns={col_estatus: "estatus_final"})

    if col_avance and col_avance != "avance_pct":
        df = df.rename(columns={col_avance: "avance_pct"})

    df = construir_curso_si_no_existe(df)

    defaults = {
        "area_de_adscripcion": "SIN ÁREA",
        "curso": "SIN CURSO",
        "estatus_final": "",
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

    if "tareas" in df.columns:
        tareas_split = df["tareas"].astype(str).str.extract(r"(\d+)\s*/\s*(\d+)")
        if tareas_split.notna().any().any():
            df["tareas_entregadas"] = pd.to_numeric(tareas_split[0], errors="coerce").fillna(df["tareas_entregadas"])
            df["tareas_totales"] = pd.to_numeric(tareas_split[1], errors="coerce").fillna(df["tareas_totales"])

    df["tareas_entregadas"] = pd.to_numeric(df["tareas_entregadas"], errors="coerce").fillna(0)
    df["tareas_totales"] = pd.to_numeric(df["tareas_totales"], errors="coerce").fillna(0)

    df["avance_pct"] = pd.to_numeric(df["avance_pct"], errors="coerce").fillna(0)

    # Si avance_pct viene en decimal 0-1, convertir a 0-100
    if df["avance_pct"].max() <= 1 and df["avance_pct"].max() > 0:
        df["avance_pct"] = df["avance_pct"] * 100

    # Si avance viene vacío pero hay tareas, calcular avance
    mask_avance_cero = (df["avance_pct"] == 0) & (df["tareas_totales"] > 0)
    df.loc[mask_avance_cero, "avance_pct"] = (
        df.loc[mask_avance_cero, "tareas_entregadas"] /
        df.loc[mask_avance_cero, "tareas_totales"] * 100
    )

    df["avance_pct"] = df["avance_pct"].clip(lower=0, upper=100)

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

    df["correo_docente"] = (
        df["correo_docente"]
        .astype(str)
        .str.strip()
        .replace("nan", "")
    )

    # Si no hay estatus, inferir con avance/tareas
    df["estatus_final"] = df["estatus_final"].astype(str).str.strip().replace("nan", "")

    def inferir_estatus(row):
        est = normalizar_texto(row.get("estatus_final", ""))
        if est:
            return row.get("estatus_final", "")

        avance = row.get("avance_pct", 0)
        tareas_totales = row.get("tareas_totales", 0)

        if avance >= 100:
            return "FINALIZADO"
        if avance > 0:
            return "EN_PROCESO"
        if tareas_totales > 0:
            return "EN_SEAC_SIN_TAREAS"
        return "NO_APARECE_EN_SEAC"

    df["estatus_final"] = df.apply(inferir_estatus, axis=1)
    df["estatus_norm"] = df["estatus_final"].apply(normalizar_texto)

    return df


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


def calcular_kpis(df):
    n = len(df)
    docentes_unicos = df["nombre_normalizado"].nunique()
    cursos_unicos = df["curso"].replace("SIN CURSO", pd.NA).dropna().nunique()

    fin = es_finalizado(df["estatus_norm"]).sum()
    proc = es_proceso(df["estatus_norm"]).sum()
    no_seac = es_no_seac(df["estatus_norm"]).sum()
    sin_tar = es_sin_tareas(df["estatus_norm"]).sum()

    return {
        "docentes": docentes_unicos,
        "inscripciones": n,
        "cursos": cursos_unicos,
        "finalizados": int(fin),
        "proceso": int(proc),
        "no_seac": int(no_seac),
        "sin_tareas": int(sin_tar),
        "pct_fin": round(fin / n * 100, 1) if n else 0,
        "pct_sin_avance": round((no_seac + sin_tar) / n * 100, 1) if n else 0,
    }


def resumen_por_curso(df):
    registros = []

    df = df[df["curso"] != "SIN CURSO"].copy()

    for curso, grp in df.groupby("curso"):
        n = len(grp)
        fin = es_finalizado(grp["estatus_norm"]).sum()
        proc = es_proceso(grp["estatus_norm"]).sum()
        no_seac = es_no_seac(grp["estatus_norm"]).sum()
        sin_tareas = es_sin_tareas(grp["estatus_norm"]).sum()

        registros.append({
            "Curso": curso,
            "Docentes únicos": grp["nombre_normalizado"].nunique(),
            "Inscripciones": n,
            "Finalizados": int(fin),
            "En proceso": int(proc),
            "No en SEAC": int(no_seac),
            "Sin tareas": int(sin_tareas),
            "% Finalización": round(fin / n * 100, 1) if n else 0,
            "% Sin avance": round((no_seac + sin_tareas) / n * 100, 1) if n else 0,
        })

    if not registros:
        return pd.DataFrame(columns=[
            "Curso", "Docentes únicos", "Inscripciones", "Finalizados",
            "En proceso", "No en SEAC", "Sin tareas",
            "% Finalización", "% Sin avance"
        ])

    return pd.DataFrame(registros).sort_values("Inscripciones", ascending=False)


def tabla_resumen_docentes(df):
    grp = df.groupby("nombre_normalizado")

    resumen = grp.agg(
        correo=("correo_docente", "first"),
        area=("area_de_adscripcion", "first"),
        cursos_inscritos=("curso", lambda x: x[x != "SIN CURSO"].nunique()),
        inscripciones=("curso", "count"),
    ).reset_index()

    resumen["finalizados"] = grp.apply(lambda x: es_finalizado(x["estatus_norm"]).sum()).values
    resumen["en_proceso"] = grp.apply(lambda x: es_proceso(x["estatus_norm"]).sum()).values
    resumen["no_seac"] = grp.apply(lambda x: es_no_seac(x["estatus_norm"]).sum()).values
    resumen["sin_tareas"] = grp.apply(lambda x: es_sin_tareas(x["estatus_norm"]).sum()).values

    resumen["pct_fin"] = (
        resumen["finalizados"] / resumen["inscripciones"] * 100
    ).round(1).where(resumen["inscripciones"] > 0, 0)

    resumen = resumen.rename(columns={
        "nombre_normalizado": "Docente",
        "correo": "Correo",
        "area": "Área",
        "cursos_inscritos": "Cursos inscritos",
        "inscripciones": "Inscripciones",
        "finalizados": "Finalizados",
        "en_proceso": "En proceso",
        "no_seac": "No en SEAC",
        "sin_tareas": "Sin tareas",
        "pct_fin": "% Fin.",
    })

    return resumen.sort_values(["Cursos inscritos", "% Fin."], ascending=False).reset_index(drop=True)


def filtrar_por_carrera_si_aplica(df, vista, carrera):
    if vista != "Director de carrera" or not carrera:
        return df

    carrera_norm = normalizar_texto(carrera)

    return df[
        df["area_de_adscripcion"].apply(normalizar_texto) == carrera_norm
    ]


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
        kpi_card("Docentes inscritos", kpis["docentes"], "Docentes únicos sin duplicar", COLOR_AZUL)
    with c2:
        kpi_card("Inscripciones", kpis["inscripciones"], "Total docente-curso", COLOR_VERDE)
    with c3:
        kpi_card("Cursos activos", kpis["cursos"], "Cursos detectados", COLOR_GRIS)
    with c4:
        kpi_card("% Finalización", f"{kpis['pct_fin']}%", "Inscripciones concluidas", COLOR_VERDE)

    c5, c6, c7, c8 = st.columns(4)

    with c5:
        kpi_card("Finalizados", kpis["finalizados"], "Curso concluido", COLOR_VERDE)
    with c6:
        kpi_card("En proceso", kpis["proceso"], "Avance parcial", COLOR_AZUL)
    with c7:
        kpi_card("No aparecen en SEAC", kpis["no_seac"], "Alerta de ingreso", "#D7263D")
    with c8:
        kpi_card("En SEAC sin tareas", kpis["sin_tareas"], "Requiere seguimiento", "#F97316")


def mostrar_cursos_incluidos(df):
    cursos = sorted([c for c in df["curso"].dropna().unique().tolist() if c and c != "SIN CURSO"])

    st.markdown("### Cursos incluidos en el seguimiento actual")

    if not cursos:
        st.warning("No se detectaron cursos. Revisa que el GID corresponda a la hoja final de seguimiento o que exista una columna de curso/taller.")
        return

    pills = "".join([f'<span class="course-pill">{i+1}. {curso}</span>' for i, curso in enumerate(cursos)])
    st.markdown(pills, unsafe_allow_html=True)


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
    fig.update_layout(showlegend=True, margin=dict(t=10, b=10, l=10, r=10), height=320)

    return fig


COLOR_MAP = {
    "Finalizado": "#10B981",
    "En proceso": "#2F80ED",
    "No aparece en SEAC": "#D7263D",
    "En SEAC sin tareas": "#F97316",
    "Otro": "#374151",
}


def grafica_area(df):
    df2 = df.copy()
    df2["Estatus"] = df2["estatus_norm"].apply(categoria_estatus)

    resumen = df2.groupby(["area_de_adscripcion", "Estatus"]).size().reset_index(name="Total")

    fig = px.bar(
        resumen,
        x="area_de_adscripcion",
        y="Total",
        color="Estatus",
        color_discrete_map=COLOR_MAP,
        barmode="stack",
        labels={"area_de_adscripcion": "Área de adscripción", "Total": "Inscripciones"},
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

    if resumen.empty:
        return px.bar(title="No hay cursos detectados")

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
            st.plotly_chart(grafica_estatus(df_permitido), use_container_width=True, key="grafica_estatus_resumen")

        with col2:
            st.markdown("**Avance por curso de capacitación**")
            st.plotly_chart(grafica_curso(df_permitido), use_container_width=True, key="grafica_curso_resumen")

        st.markdown("**Inscripciones por área de adscripción y estatus**")
        st.plotly_chart(grafica_area(df_permitido), use_container_width=True, key="grafica_area_resumen")

    with tab2:
        if vista == "Director de carrera" and carrera:
            area_sel = carrera
            st.info(f"Vista filtrada para: **{area_sel}**")
            df_area = df_permitido.copy()
        else:
            areas = sorted(df_permitido["area_de_adscripcion"].dropna().unique())
            area_sel = st.selectbox("Selecciona área / carrera", areas, key="sel_area_dc")
            df_area = df_permitido[df_permitido["area_de_adscripcion"] == area_sel]

        st.subheader(f"Seguimiento de capacitación — {area_sel}")

        mostrar_kpis(calcular_kpis(df_area))
        st.divider()

        st.dataframe(
            tabla_resumen_docentes(df_area).drop(columns=["Área"], errors="ignore"),
            use_container_width=True,
            hide_index=True,
            key="tabla_docentes_area",
            column_config={
                "% Fin.": st.column_config.ProgressColumn(
                    "% Fin.", min_value=0, max_value=100, format="%.1f%%"
                ),
            },
        )

    with tab3:
        st.subheader("Análisis individual por curso")

        cursos = sorted([c for c in df_permitido["curso"].dropna().unique().tolist() if c != "SIN CURSO"])

        if not cursos:
            st.warning("No hay cursos detectados.")
        else:
            curso_sel = st.selectbox("Selecciona curso", cursos, key="sel_curso_capacitacion")
            df_curso = df_permitido[df_permitido["curso"] == curso_sel]

            st.markdown(f"### {curso_sel}")

            mostrar_kpis(calcular_kpis(df_curso))
            st.divider()

            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("**Estatus del curso**")
                st.plotly_chart(grafica_estatus(df_curso), use_container_width=True, key="grafica_estatus_curso")

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
            r2 = resumen_por_curso(df_permitido)[["Curso", "Inscripciones", "Docentes únicos", "% Finalización"]]
            st.dataframe(r2, use_container_width=True, hide_index=True, key="rank_cursos_demanda")

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("**Docentes con más cursos inscritos**")
            r3 = (
                df_permitido[df_permitido["curso"] != "SIN CURSO"]
                .groupby("nombre_normalizado")["curso"]
                .nunique()
                .reset_index(name="Cursos inscritos")
                .sort_values("Cursos inscritos", ascending=False)
                .head(15)
            )
            st.dataframe(r3, use_container_width=True, hide_index=True, key="rank_docentes_inscritos")

        with col4:
            st.markdown("**Docentes con más cursos finalizados**")
            r4 = (
                df_permitido[es_finalizado(df_permitido["estatus_norm"])]
                .groupby("nombre_normalizado")["curso"]
                .nunique()
                .reset_index(name="Cursos finalizados")
                .sort_values("Cursos finalizados", ascending=False)
                .head(15)
            )
            st.dataframe(r4, use_container_width=True, hide_index=True, key="rank_docentes_finalizados")

    with tab5:
        st.subheader("Tabla completa con filtros")

        f1, f2, f3 = st.columns(3)

        areas_todas = ["Todas"] + sorted(df_permitido["area_de_adscripcion"].dropna().unique().tolist())
        cursos_todos = ["Todos"] + sorted([c for c in df_permitido["curso"].dropna().unique().tolist() if c != "SIN CURSO"])
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
