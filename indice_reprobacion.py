import pandas as pd
import streamlit as st
import altair as alt
import gspread
import re

try:
    import bajas_retencion
except Exception:
    bajas_retencion = None


# =====================================================
# CONFIG
# =====================================================
SHEET_NAME_DEFAULT = "REPROBACION"
UMBRAL_REPROBACION_DEFAULT = 70

COLOR_AZUL = "#2F80ED"
COLOR_VERDE = "#10B981"
COLOR_ROJO = "#D7263D"
COLOR_NARANJA = "#F97316"
COLOR_GRIS = "#374151"
COLOR_MORADO = "#7C3AED"


# =====================================================
# ESTILOS
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
        .context-box {
            background: #EFF6FF;
            border-left: 5px solid #2F80ED;
            border-radius: 10px;
            padding: 14px 16px;
            margin-bottom: 16px;
            color: #1F2937;
            font-size: 0.95rem;
        }
        .note-box {
            background: #FFF7ED;
            border-left: 5px solid #F97316;
            border-radius: 10px;
            padding: 12px 14px;
            margin-top: 8px;
            margin-bottom: 16px;
            color: #374151;
            font-size: 0.9rem;
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


def contexto_box(carrera_txt, ciclo_txt):
    st.markdown(
        f"""
        <div class="context-box">
            <b>Vista actual:</b> {carrera_txt} &nbsp; | &nbsp;
            <b>Ciclo:</b> {ciclo_txt}
        </div>
        """,
        unsafe_allow_html=True,
    )


def nota_promedio():
    st.markdown(
        """
        <div class="note-box">
            <b>Nota:</b> El promedio mostrado corresponde únicamente a registros de reprobación.
            No representa el promedio general de todos los alumnos inscritos.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =====================================================
# NORMALIZACIÓN
# =====================================================
def normalizar_texto(valor) -> str:
    if pd.isna(valor):
        return ""
    s = str(valor).lower()
    s = s.replace("\u00A0", " ")
    s = (
        s.replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        .replace("ü", "u")
        .replace("ñ", "n")
    )
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().upper() for c in out.columns]
    return out


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def _to_num(s):
    return pd.to_numeric(s, errors="coerce")


def _ciclo_to_int(x) -> int | None:
    if x is None:
        return None
    s = str(x).strip()
    if not s or s in ["(Todos)", "(Todas)"]:
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def _ciclo_sort_key(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.strip(), errors="coerce")


def _user_can_see_bajas() -> bool:
    if bool(st.session_state.get("user_allow_all", False)):
        return True
    mods = st.session_state.get("user_modulos", set())
    try:
        return "bajas_retencion" in set(mods)
    except Exception:
        return False


# =====================================================
# CARGA DE DATOS
# =====================================================
@st.cache_data(show_spinner=False, ttl=300)
def _load_reprobacion_from_gsheets(url: str, sheet_name: str | None = None) -> pd.DataFrame:
    sa = st.secrets["gcp_service_account_json"]
    sa_dict = dict(sa) if isinstance(sa, dict) else dict(sa)

    gc = gspread.service_account_from_dict(sa_dict)
    sh = gc.open_by_url(url)

    ws = sh.worksheet(sheet_name) if sheet_name else sh.sheet1
    values = ws.get_all_values()

    if not values:
        return pd.DataFrame()

    df = pd.DataFrame(values[1:], columns=[h.strip() for h in values[0]]).replace("", pd.NA)
    df = _norm_cols(df)

    renames = {
        _pick_col(df, ["CICLO", "CICLO_ESCOLAR", "PERIODO"]): "CICLO",
        _pick_col(df, ["ESCUELA", "PLANTEL"]): "ESCUELA",
        _pick_col(df, ["NIVEL"]): "NIVEL",
        _pick_col(df, ["AREA", "ÁREA", "CARRERA", "SERVICIO", "PROGRAMA"]): "AREA",
        _pick_col(df, ["MATRICULA", "MATRÍCULA", "MATRICULA ALUMNO"]): "MATRICULA",
        _pick_col(df, ["ALUMNO", "NOMBRE ALUMNO", "NOMBRE_COMPLETO"]): "ALUMNO",
        _pick_col(df, ["MATERIA", "ASIGNATURA", "UNIDAD DE APRENDIZAJE"]): "MATERIA",
        _pick_col(df, ["DOCENTE", "PROFESOR", "MAESTRO", "CATEDRATICO", "CATEDRÁTICO"]): "DOCENTE",
        _pick_col(df, ["CALIF FINAL", "CALIF_FINAL", "CALIFICACION FINAL", "CALIFICACIÓN FINAL"]): "CALIF_FINAL",
    }

    for k, v in renames.items():
        if k and k != v:
            df = df.rename(columns={k: v})

    return df


# =====================================================
# LIMPIEZA
# =====================================================
def preparar_reprobacion(df: pd.DataFrame, umbral_reprobacion: float = 70) -> pd.DataFrame:
    df = df.copy()

    defaults = {
        "CICLO": "SIN CICLO",
        "ESCUELA": "SIN ESCUELA",
        "NIVEL": "SIN NIVEL",
        "AREA": "SIN ÁREA",
        "MATRICULA": "",
        "ALUMNO": "SIN ALUMNO",
        "MATERIA": "SIN MATERIA",
        "DOCENTE": "SIN DOCENTE",
        "CALIF_FINAL": pd.NA,
    }

    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    for c in ["CICLO", "ESCUELA", "NIVEL", "AREA", "MATRICULA", "ALUMNO", "MATERIA", "DOCENTE"]:
        df[c] = df[c].astype(str).str.strip()
        df[c] = df[c].replace(["nan", "None", ""], defaults.get(c, ""))

    df["CALIF_FINAL"] = _to_num(df["CALIF_FINAL"])

    df["AREA_norm"] = df["AREA"].apply(normalizar_texto)
    df["MATERIA_norm"] = df["MATERIA"].apply(normalizar_texto)
    df["DOCENTE_norm"] = df["DOCENTE"].apply(normalizar_texto)
    df["MATRICULA_norm"] = df["MATRICULA"].astype(str).str.strip().str.lower()

    # Si la hoja ya es de REPROBACIÓN, normalmente todos los registros cuentan.
    # Si accidentalmente vienen calificaciones aprobatorias, se excluyen.
    if df["CALIF_FINAL"].notna().any() and (df["CALIF_FINAL"] >= umbral_reprobacion).any():
        df = df[(df["CALIF_FINAL"].isna()) | (df["CALIF_FINAL"] < umbral_reprobacion)].copy()

    return df


# =====================================================
# CÁLCULOS
# =====================================================
def calcular_kpis(df: pd.DataFrame) -> dict:
    registros = len(df)
    alumnos_unicos = df["MATRICULA_norm"].replace("", pd.NA).dropna().nunique()
    materias_distintas = df["MATERIA"].replace("SIN MATERIA", pd.NA).dropna().nunique()
    docentes_distintos = df["DOCENTE"].replace("SIN DOCENTE", pd.NA).dropna().nunique()
    promedio = df["CALIF_FINAL"].mean() if "CALIF_FINAL" in df.columns else pd.NA

    promedio_repr_por_alumno = registros / alumnos_unicos if alumnos_unicos else 0

    return {
        "registros": int(registros),
        "alumnos_unicos": int(alumnos_unicos),
        "materias_distintas": int(materias_distintas),
        "docentes_distintos": int(docentes_distintos),
        "promedio": promedio,
        "promedio_repr_por_alumno": promedio_repr_por_alumno,
    }


def mostrar_kpis(kpis: dict):
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi_card(
            "Alumnos con reprobación",
            f"{kpis['alumnos_unicos']:,}",
            "Alumnos únicos",
            COLOR_ROJO,
        )

    with c2:
        kpi_card(
            "Materias reprobadas",
            f"{kpis['registros']:,}",
            "Registros alumno-materia",
            COLOR_NARANJA,
        )

    with c3:
        prom = "—" if pd.isna(kpis["promedio"]) else f"{kpis['promedio']:.2f}"
        kpi_card(
            "Promedio reprobatorio",
            prom,
            "Solo registros reprobados",
            COLOR_MORADO,
        )

    with c4:
        kpi_card(
            "Materias distintas",
            f"{kpis['materias_distintas']:,}",
            "Asignaturas con reprobación",
            COLOR_AZUL,
        )


def resumen_por_carrera(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("AREA", dropna=False)

    resumen = pd.DataFrame({
        "Carrera": g.size().index.astype(str),
        "Materias reprobadas": g.size().values,
        "Alumnos únicos con reprobación": g["MATRICULA_norm"].nunique().values,
        "Materias distintas": g["MATERIA"].nunique().values,
        "Docentes distintos": g["DOCENTE"].nunique().values,
        "Promedio reprobatorio": g["CALIF_FINAL"].mean().values,
    })

    resumen["Reprobaciones por alumno"] = (
        resumen["Materias reprobadas"] / resumen["Alumnos únicos con reprobación"]
    ).replace([float("inf"), -float("inf")], 0).fillna(0)

    return resumen.sort_values(
        ["Alumnos únicos con reprobación", "Materias reprobadas"],
        ascending=False
    ).reset_index(drop=True)


def top_materias(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    g = df.groupby(["MATERIA", "DOCENTE"], dropna=False)

    tabla = pd.DataFrame({
        "Materia": [idx[0] for idx in g.size().index],
        "Docente": [idx[1] for idx in g.size().index],
        "Materias reprobadas": g.size().values,
        "Alumnos únicos con reprobación": g["MATRICULA_norm"].nunique().values,
        "Promedio reprobatorio": g["CALIF_FINAL"].mean().values,
    })

    tabla["Promedio reprobatorio"] = tabla["Promedio reprobatorio"].round(2)

    return tabla.sort_values(
        ["Alumnos únicos con reprobación", "Materias reprobadas"],
        ascending=False
    ).head(n).reset_index(drop=True)


def top_docentes(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    g = df.groupby("DOCENTE", dropna=False)

    tabla = pd.DataFrame({
        "Docente": g.size().index.astype(str),
        "Materias reprobadas": g.size().values,
        "Alumnos únicos con reprobación": g["MATRICULA_norm"].nunique().values,
        "Materias distintas": g["MATERIA"].nunique().values,
        "Promedio reprobatorio": g["CALIF_FINAL"].mean().values,
    })

    tabla["Promedio reprobatorio"] = tabla["Promedio reprobatorio"].round(2)

    tabla = tabla[tabla["Docente"] != "SIN DOCENTE"]

    return tabla.sort_values(
        ["Alumnos únicos con reprobación", "Materias reprobadas"],
        ascending=False
    ).head(n).reset_index(drop=True)


def historico_por_ciclo(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("CICLO", dropna=False)

    hist = pd.DataFrame({
        "Ciclo": g.size().index.astype(str),
        "Materias reprobadas": g.size().values,
        "Alumnos únicos con reprobación": g["MATRICULA_norm"].nunique().values,
    })

    hist["CICLO_NUM"] = _ciclo_sort_key(hist["Ciclo"])
    hist = hist.sort_values(["CICLO_NUM", "Ciclo"]).drop(columns=["CICLO_NUM"])

    return hist.reset_index(drop=True)


# =====================================================
# GRÁFICAS
# =====================================================
def grafica_carreras(resumen: pd.DataFrame):
    if resumen.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_bar()

    base = resumen.head(15).copy()

    chart = (
        alt.Chart(base)
        .mark_bar()
        .encode(
            y=alt.Y("Carrera:N", sort="-x", title=None),
            x=alt.X("Alumnos únicos con reprobación:Q", title="Alumnos únicos con reprobación"),
            tooltip=[
                alt.Tooltip("Carrera:N", title="Carrera"),
                alt.Tooltip("Alumnos únicos con reprobación:Q", title="Alumnos únicos"),
                alt.Tooltip("Materias reprobadas:Q", title="Materias reprobadas"),
                alt.Tooltip("Reprobaciones por alumno:Q", title="Rep. por alumno", format=".2f"),
                alt.Tooltip("Promedio reprobatorio:Q", title="Promedio", format=".2f"),
            ],
        )
        .properties(height=max(320, min(700, 28 * len(base))))
    )

    labels = (
        alt.Chart(base)
        .mark_text(align="left", dx=5)
        .encode(
            y=alt.Y("Carrera:N", sort="-x"),
            x=alt.X("Alumnos únicos con reprobación:Q"),
            text=alt.Text("Alumnos únicos con reprobación:Q"),
        )
    )

    return chart + labels


def grafica_top_materias(tabla: pd.DataFrame):
    if tabla.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_bar()

    base = tabla.copy()
    base["Materia - Docente"] = base["Materia"] + " | " + base["Docente"]

    chart = (
        alt.Chart(base)
        .mark_bar()
        .encode(
            y=alt.Y("Materia - Docente:N", sort="-x", title=None),
            x=alt.X("Alumnos únicos con reprobación:Q", title="Alumnos únicos con reprobación"),
            tooltip=[
                alt.Tooltip("Materia:N", title="Materia"),
                alt.Tooltip("Docente:N", title="Docente"),
                alt.Tooltip("Alumnos únicos con reprobación:Q", title="Alumnos únicos"),
                alt.Tooltip("Materias reprobadas:Q", title="Materias reprobadas"),
                alt.Tooltip("Promedio reprobatorio:Q", title="Promedio", format=".2f"),
            ],
        )
        .properties(height=max(320, min(650, 34 * len(base))))
    )

    labels = (
        alt.Chart(base)
        .mark_text(align="left", dx=5)
        .encode(
            y=alt.Y("Materia - Docente:N", sort="-x"),
            x=alt.X("Alumnos únicos con reprobación:Q"),
            text=alt.Text("Alumnos únicos con reprobación:Q"),
        )
    )

    return chart + labels


def grafica_historico(hist: pd.DataFrame, titulo: str):
    if hist.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_line()

    base = hist.copy()

    line = (
        alt.Chart(base)
        .mark_line(point=True)
        .encode(
            x=alt.X("Ciclo:N", title="Ciclo", sort=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Alumnos únicos con reprobación:Q", title="Alumnos únicos con reprobación"),
            tooltip=[
                alt.Tooltip("Ciclo:N", title="Ciclo"),
                alt.Tooltip("Alumnos únicos con reprobación:Q", title="Alumnos únicos"),
                alt.Tooltip("Materias reprobadas:Q", title="Materias reprobadas"),
            ],
        )
        .properties(height=360, title=titulo)
    )

    labels = (
        alt.Chart(base)
        .mark_text(dy=-10)
        .encode(
            x=alt.X("Ciclo:N", sort=None),
            y=alt.Y("Alumnos únicos con reprobación:Q"),
            text=alt.Text("Alumnos únicos con reprobación:Q"),
        )
    )

    return line + labels


def grafica_pie_bajas(top: pd.DataFrame):
    if top is None or top.empty:
        return None

    df = top.copy()

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    text_cols = [c for c in df.columns if c not in numeric_cols]

    if not numeric_cols or not text_cols:
        return None

    col_valor = numeric_cols[0]
    col_categoria = text_cols[0]

    df[col_categoria] = df[col_categoria].astype(str)

    chart = (
        alt.Chart(df)
        .mark_arc()
        .encode(
            theta=alt.Theta(f"{col_valor}:Q"),
            color=alt.Color(f"{col_categoria}:N", title="Tipo de baja"),
            tooltip=[
                alt.Tooltip(f"{col_categoria}:N", title="Tipo"),
                alt.Tooltip(f"{col_valor}:Q", title="Casos"),
            ],
        )
        .properties(height=330)
    )

    return chart


# =====================================================
# BAJAS COMO CONTEXTO
# =====================================================
def mostrar_contexto_bajas(ciclo_sel, area_ctx):
    if bajas_retencion is None:
        return

    if not _user_can_see_bajas():
        return

    with st.expander("Contexto adicional: bajas y retención", expanded=False):
        ciclo_int = _ciclo_to_int(ciclo_sel)

        st.caption(
            f"Resumen de bajas para: "
            f"{area_ctx if area_ctx else 'todas las carreras'} | "
            f"{ciclo_sel if ciclo_sel != '(Todos)' else 'todos los ciclos'}"
        )

        try:
            res = bajas_retencion.resumen_bajas_por_filtros(
                ciclo=ciclo_int,
                area=area_ctx,
            )

            st.metric("Bajas", f"{res.get('n', 0):,}")

            top = res.get("top_motivos")

            if top is not None and not top.empty:
                c1, c2 = st.columns([1, 1])

                with c1:
                    st.dataframe(top, use_container_width=True, hide_index=True)

                with c2:
                    chart = grafica_pie_bajas(top)
                    if chart is not None:
                        st.altair_chart(chart, use_container_width=True)
                    else:
                        st.caption("No fue posible construir la gráfica circular con la estructura actual.")
            else:
                st.caption("Sin detalle de motivos para este filtro.")

        except Exception as e:
            st.error("No pude calcular el resumen de bajas.")
            st.exception(e)


# =====================================================
# RENDER PRINCIPAL
# =====================================================
def render_indice_reprobacion(vista: str | None = None, carrera: str | None = None):
    aplicar_estilos()

    st.title("Índice de reprobación")
    st.caption("Análisis de alumnos con reprobación, materias críticas, docentes asociados y comparación por carrera.")

    if not vista:
        vista = "Dirección General"

    url = (st.secrets.get("IR_URL", "") or "").strip()
    sheet_name = (st.secrets.get("IR_SHEET_NAME", SHEET_NAME_DEFAULT) or "").strip() or None
    umbral = float(st.secrets.get("IR_UMBRAL_REPROBACION", UMBRAL_REPROBACION_DEFAULT))

    if not url:
        st.error("Falta configurar IR_URL en Secrets.")
        return

    try:
        with st.spinner("Cargando datos de reprobación..."):
            df_raw = _load_reprobacion_from_gsheets(url, sheet_name)
    except Exception as e:
        st.error("No se pudo cargar el Google Sheet de reprobación.")
        st.exception(e)
        return

    if df_raw.empty:
        st.warning("La hoja de reprobación está vacía.")
        return

    df = preparar_reprobacion(df_raw, umbral)

    for req in ["AREA", "CICLO", "MATERIA", "MATRICULA"]:
        if req not in df.columns:
            st.error(f"Falta columna requerida: {req}")
            st.caption(f"Columnas detectadas: {', '.join(df.columns)}")
            return

    if df.empty:
        st.warning("No hay registros de reprobación después de aplicar la limpieza.")
        return

    # =====================================================
    # FILTROS
    # =====================================================
    f = df.copy()
    area_ctx = None

    if vista == "Director de carrera" and carrera:
        carrera_norm = normalizar_texto(carrera)
        f = f[f["AREA_norm"] == carrera_norm]

        st.text_input("Carrera", value=carrera, disabled=True)

        ciclos = ["(Todos)"] + sorted(f["CICLO"].dropna().unique().tolist())
        ciclo_sel = st.selectbox("Ciclo", ciclos, key="ir_ciclo_director")

        if ciclo_sel != "(Todos)":
            f = f[f["CICLO"] == ciclo_sel]

        area_ctx = carrera
        carrera_txt = carrera

    else:
        c1, c2 = st.columns(2)

        with c1:
            area_sel = st.selectbox(
                "Carrera",
                ["(Todas)"] + sorted(f["AREA"].dropna().unique().tolist()),
                key="ir_area_dg",
            )

        with c2:
            ciclo_sel = st.selectbox(
                "Ciclo",
                ["(Todos)"] + sorted(f["CICLO"].dropna().unique().tolist()),
                key="ir_ciclo_dg",
            )

        if area_sel != "(Todas)":
            f = f[f["AREA"] == area_sel]
            area_ctx = area_sel
            carrera_txt = area_sel
        else:
            carrera_txt = "Todas las carreras"

        if ciclo_sel != "(Todos)":
            f = f[f["CICLO"] == ciclo_sel]

    ciclo_txt = ciclo_sel if ciclo_sel != "(Todos)" else "Todos los ciclos"
    contexto_box(carrera_txt, ciclo_txt)

    if f.empty:
        st.warning("No hay registros con los filtros seleccionados.")
        return

    # =====================================================
    # NAVEGACIÓN INTERNA
    # =====================================================
    st.sidebar.markdown("### Índice de reprobación")

    opciones = [
        "Resumen ejecutivo",
        "Comparativo por carrera",
        "Top materias críticas",
        "Docentes asociados",
        "Detalle general",
    ]

    vista_modulo = st.sidebar.radio(
        "Vista del módulo",
        opciones,
        key="nav_indice_reprobacion",
    )

    # =====================================================
    # KPIs GENERALES
    # =====================================================
    kpis = calcular_kpis(f)
    mostrar_kpis(kpis)
    nota_promedio()

    mostrar_contexto_bajas(ciclo_sel, area_ctx)

    st.divider()

    # =====================================================
    # RESUMEN EJECUTIVO
    # =====================================================
    if vista_modulo == "Resumen ejecutivo":
        st.subheader("Resumen ejecutivo de reprobación")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Tendencia por ciclo")
            hist = historico_por_ciclo(f)
            titulo = f"Alumnos únicos con reprobación — {carrera_txt}"
            st.altair_chart(grafica_historico(hist, titulo), use_container_width=True)
            st.dataframe(hist, use_container_width=True, hide_index=True)

        with col2:
            st.markdown("### Top 10 materias críticas")
            tm = top_materias(f, 10)
            st.altair_chart(grafica_top_materias(tm), use_container_width=True)

        st.markdown("### Tabla de materias críticas")
        st.dataframe(
            top_materias(f, 10),
            use_container_width=True,
            hide_index=True,
        )

    # =====================================================
    # COMPARATIVO POR CARRERA
    # =====================================================
    elif vista_modulo == "Comparativo por carrera":
        st.subheader("Comparativo institucional por carrera")

        resumen = resumen_por_carrera(f)

        if len(resumen) < 2:
            st.info("Con el filtro actual solo hay una carrera. Para comparar, selecciona '(Todas)' en Carrera.")
        else:
            st.markdown("### Carreras con mayor número de alumnos únicos con reprobación")
            st.altair_chart(grafica_carreras(resumen), use_container_width=True)

        st.markdown("### Tabla comparativa")
        st.dataframe(
            resumen,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Promedio reprobatorio": st.column_config.NumberColumn(
                    "Promedio reprobatorio", format="%.2f"
                ),
                "Reprobaciones por alumno": st.column_config.NumberColumn(
                    "Reprobaciones por alumno", format="%.2f"
                ),
            },
        )

        st.caption(
            "Interpretación: 'Materias reprobadas' son registros alumno-materia. "
            "'Alumnos únicos con reprobación' evita duplicar alumnos que reprobaron varias materias."
        )

    # =====================================================
    # TOP MATERIAS CRÍTICAS
    # =====================================================
    elif vista_modulo == "Top materias críticas":
        st.subheader("Top materias con mayor reprobación")

        n_top = st.slider("Número de materias a mostrar", 5, 30, 10)

        tabla = top_materias(f, n_top)

        if tabla.empty:
            st.warning("No hay materias para mostrar.")
            return

        st.markdown("### Gráfica de materias críticas")
        st.altair_chart(grafica_top_materias(tabla), use_container_width=True)

        st.markdown("### Tabla de materias críticas")
        st.dataframe(
            tabla,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Promedio reprobatorio": st.column_config.NumberColumn(
                    "Promedio reprobatorio", format="%.2f"
                ),
            },
        )

        st.caption(
            "Esta es la vista más útil para directores: permite identificar materias donde conviene proponer "
            "regularización, tutorías, revisión de evaluación o acompañamiento docente."
        )

    # =====================================================
    # DOCENTES ASOCIADOS
    # =====================================================
    elif vista_modulo == "Docentes asociados":
        st.subheader("Docentes asociados a mayor número de reprobaciones")

        if "DOCENTE" not in f.columns or f["DOCENTE"].replace("SIN DOCENTE", pd.NA).dropna().empty:
            st.warning("No se detectó información de docente en la hoja.")
            st.caption("Para habilitar esta vista correctamente, la base debe incluir una columna DOCENTE o PROFESOR.")
            return

        n_top = st.slider("Número de docentes a mostrar", 5, 30, 10)

        tabla = top_docentes(f, n_top)

        if tabla.empty:
            st.warning("No hay docentes para mostrar.")
            return

        chart = (
            alt.Chart(tabla)
            .mark_bar()
            .encode(
                y=alt.Y("Docente:N", sort="-x", title=None),
                x=alt.X("Alumnos únicos con reprobación:Q", title="Alumnos únicos con reprobación"),
                tooltip=[
                    alt.Tooltip("Docente:N", title="Docente"),
                    alt.Tooltip("Alumnos únicos con reprobación:Q", title="Alumnos únicos"),
                    alt.Tooltip("Materias reprobadas:Q", title="Materias reprobadas"),
                    alt.Tooltip("Materias distintas:Q", title="Materias distintas"),
                    alt.Tooltip("Promedio reprobatorio:Q", title="Promedio", format=".2f"),
                ],
            )
            .properties(height=max(320, min(700, 32 * len(tabla))))
        )

        labels = (
            alt.Chart(tabla)
            .mark_text(align="left", dx=5)
            .encode(
                y=alt.Y("Docente:N", sort="-x"),
                x=alt.X("Alumnos únicos con reprobación:Q"),
                text=alt.Text("Alumnos únicos con reprobación:Q"),
            )
        )

        st.altair_chart(chart + labels, use_container_width=True)

        st.dataframe(
            tabla,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Promedio reprobatorio": st.column_config.NumberColumn(
                    "Promedio reprobatorio", format="%.2f"
                ),
            },
        )

        st.caption(
            "Importante: esta vista no debe usarse para señalar culpables de forma aislada. "
            "Debe cruzarse con materia, grupo, ciclo, tamaño del grupo y contexto académico."
        )

    # =====================================================
    # DETALLE GENERAL
    # =====================================================
    elif vista_modulo == "Detalle general":
        st.subheader("Detalle general con filtros")

        d1, d2, d3 = st.columns(3)

        materias = ["(Todas)"] + sorted(f["MATERIA"].dropna().unique().tolist())
        docentes = ["(Todos)"] + sorted(f["DOCENTE"].dropna().unique().tolist())
        niveles = ["(Todos)"] + sorted(f["NIVEL"].dropna().unique().tolist())

        materia_f = d1.selectbox("Materia", materias, key="ir_f_materia")
        docente_f = d2.selectbox("Docente", docentes, key="ir_f_docente")
        nivel_f = d3.selectbox("Nivel", niveles, key="ir_f_nivel")

        df_det = f.copy()

        if materia_f != "(Todas)":
            df_det = df_det[df_det["MATERIA"] == materia_f]

        if docente_f != "(Todos)":
            df_det = df_det[df_det["DOCENTE"] == docente_f]

        if nivel_f != "(Todos)":
            df_det = df_det[df_det["NIVEL"] == nivel_f]

        st.caption(f"{len(df_det):,} registros encontrados")

        cols = [
            "CICLO",
            "ESCUELA",
            "NIVEL",
            "AREA",
            "MATRICULA",
            "ALUMNO",
            "MATERIA",
            "DOCENTE",
            "CALIF_FINAL",
        ]

        cols = [c for c in cols if c in df_det.columns]

        tabla = df_det[cols].rename(columns={
            "CICLO": "Ciclo",
            "ESCUELA": "Escuela",
            "NIVEL": "Nivel",
            "AREA": "Carrera",
            "MATRICULA": "Matrícula",
            "ALUMNO": "Alumno",
            "MATERIA": "Materia",
            "DOCENTE": "Docente",
            "CALIF_FINAL": "Calificación final",
        })

        st.dataframe(
            tabla,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Calificación final": st.column_config.NumberColumn(
                    "Calificación final", format="%.2f"
                ),
            },
        )
