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
        .insight-box {
            background: #F9FAFB;
            border-left: 5px solid #374151;
            border-radius: 10px;
            padding: 14px 16px;
            margin-top: 12px;
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
            <b>Reporte filtrado por:</b><br>
            Programa académico: <b>{carrera_txt}</b><br>
            Ciclo: <b>{ciclo_txt}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )


def nota_promedio():
    st.markdown(
        """
        <div class="note-box">
            <b>Aclaración:</b> El promedio reprobatorio se calcula solo con calificaciones menores a 70.
            No representa el promedio general de todos los alumnos.
        </div>
        """,
        unsafe_allow_html=True,
    )


def bloque_lectura_simple(kpis):
    alumnos = kpis["alumnos_unicos"]
    registros = kpis["registros"]
    ratio = kpis["promedio_repr_por_alumno"]

    st.markdown(
        f"""
        <div class="insight-box">
            <b>Cómo leer este reporte:</b><br>
            En este filtro hay <b>{alumnos:,} alumnos</b> con al menos una materia reprobada.
            En conjunto acumulan <b>{registros:,} materias reprobadas</b>.
            Esto equivale a un promedio de <b>{ratio:.1f} materias reprobadas por alumno afectado</b>.
            <br><br>
            La prioridad operativa es revisar primero las <b>materias críticas</b>, porque ahí se pueden planear
            asesorías, regularización o acompañamiento académico.
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
        "CALIF_FINAL": pd.NA,
    }

    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    for c in ["CICLO", "ESCUELA", "NIVEL", "AREA", "MATRICULA", "ALUMNO", "MATERIA"]:
        df[c] = df[c].astype(str).str.strip()
        df[c] = df[c].replace(["nan", "None", ""], defaults.get(c, ""))

    df["CALIF_FINAL"] = _to_num(df["CALIF_FINAL"])

    df["AREA_norm"] = df["AREA"].apply(normalizar_texto)
    df["MATERIA_norm"] = df["MATERIA"].apply(normalizar_texto)
    df["MATRICULA_norm"] = df["MATRICULA"].astype(str).str.strip().str.lower()

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
    promedio = df["CALIF_FINAL"].mean() if "CALIF_FINAL" in df.columns else pd.NA
    promedio_repr_por_alumno = registros / alumnos_unicos if alumnos_unicos else 0

    return {
        "registros": int(registros),
        "alumnos_unicos": int(alumnos_unicos),
        "materias_distintas": int(materias_distintas),
        "promedio": promedio,
        "promedio_repr_por_alumno": promedio_repr_por_alumno,
    }


def mostrar_kpis(kpis: dict):
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi_card(
            "Alumnos con reprobación",
            f"{kpis['alumnos_unicos']:,}",
            "Personas distintas",
            COLOR_ROJO,
        )

    with c2:
        kpi_card(
            "Materias reprobadas",
            f"{kpis['registros']:,}",
            "Total alumno-materia",
            COLOR_NARANJA,
        )

    with c3:
        prom = "—" if pd.isna(kpis["promedio"]) else f"{kpis['promedio']:.2f}"
        kpi_card(
            "Promedio reprobatorio",
            prom,
            "Solo calificaciones < 70",
            COLOR_MORADO,
        )

    with c4:
        kpi_card(
            "Materias distintas",
            f"{kpis['materias_distintas']:,}",
            "Asignaturas afectadas",
            COLOR_AZUL,
        )


def resumen_por_carrera(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("AREA", dropna=False)

    resumen = pd.DataFrame({
        "Carrera": g.size().index.astype(str),
        "Total de materias reprobadas": g.size().values,
        "Alumnos únicos con reprobación": g["MATRICULA_norm"].nunique().values,
        "Materias distintas con reprobación": g["MATERIA"].nunique().values,
        "Promedio reprobatorio": g["CALIF_FINAL"].mean().values,
    })

    resumen["Materias reprobadas por alumno"] = (
        resumen["Total de materias reprobadas"] / resumen["Alumnos únicos con reprobación"]
    ).replace([float("inf"), -float("inf")], 0).fillna(0)

    resumen["Promedio reprobatorio"] = resumen["Promedio reprobatorio"].round(2)
    resumen["Materias reprobadas por alumno"] = resumen["Materias reprobadas por alumno"].round(2)

    return resumen.sort_values(
        ["Alumnos únicos con reprobación", "Total de materias reprobadas"],
        ascending=False
    ).reset_index(drop=True)


def top_materias(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    g = df.groupby("MATERIA", dropna=False)

    tabla = pd.DataFrame({
        "Materia": g.size().index.astype(str),
        "Alumnos únicos con reprobación": g["MATRICULA_norm"].nunique().values,
        "Total de materias reprobadas": g.size().values,
        "Promedio reprobatorio": g["CALIF_FINAL"].mean().values,
    })

    tabla["Promedio reprobatorio"] = tabla["Promedio reprobatorio"].round(2)

    return tabla.sort_values(
        ["Alumnos únicos con reprobación", "Total de materias reprobadas"],
        ascending=False
    ).head(n).reset_index(drop=True)


def historico_por_ciclo(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("CICLO", dropna=False)

    hist = pd.DataFrame({
        "Ciclo": g.size().index.astype(str),
        "Alumnos únicos con reprobación": g["MATRICULA_norm"].nunique().values,
        "Total de materias reprobadas": g.size().values,
    })

    hist["CICLO_NUM"] = _ciclo_sort_key(hist["Ciclo"])
    hist = hist.sort_values(["CICLO_NUM", "Ciclo"]).drop(columns=["CICLO_NUM"])

    return hist.reset_index(drop=True)


def distribucion_calificaciones(df: pd.DataFrame) -> pd.DataFrame:
    base = df.copy()
    base = base[base["CALIF_FINAL"].notna()].copy()

    if base.empty:
        return pd.DataFrame()

    bins = [-1, 30, 50, 60, 70]
    labels = ["0 a 30", "31 a 50", "51 a 60", "61 a 69"]

    base["Rango"] = pd.cut(base["CALIF_FINAL"], bins=bins, labels=labels)

    dist = (
        base.groupby("Rango", observed=False)
        .size()
        .reset_index(name="Total de materias reprobadas")
    )

    total = dist["Total de materias reprobadas"].sum()
    dist["Porcentaje"] = (dist["Total de materias reprobadas"] / total * 100).round(1) if total else 0

    return dist


# =====================================================
# GRÁFICAS
# =====================================================
def grafica_carreras(resumen: pd.DataFrame):
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
                alt.Tooltip("Total de materias reprobadas:Q", title="Total materias reprobadas"),
                alt.Tooltip("Materias reprobadas por alumno:Q", title="Materias por alumno", format=".2f"),
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
    base = tabla.copy()

    chart = (
        alt.Chart(base)
        .mark_bar()
        .encode(
            y=alt.Y("Materia:N", sort="-x", title=None),
            x=alt.X("Alumnos únicos con reprobación:Q", title="Alumnos únicos con reprobación"),
            tooltip=[
                alt.Tooltip("Materia:N", title="Materia"),
                alt.Tooltip("Alumnos únicos con reprobación:Q", title="Alumnos únicos"),
                alt.Tooltip("Total de materias reprobadas:Q", title="Total materias reprobadas"),
                alt.Tooltip("Promedio reprobatorio:Q", title="Promedio", format=".2f"),
            ],
        )
        .properties(height=max(320, min(650, 34 * len(base))))
    )

    labels = (
        alt.Chart(base)
        .mark_text(align="left", dx=5)
        .encode(
            y=alt.Y("Materia:N", sort="-x"),
            x=alt.X("Alumnos únicos con reprobación:Q"),
            text=alt.Text("Alumnos únicos con reprobación:Q"),
        )
    )

    return chart + labels


def grafica_historico(hist: pd.DataFrame, titulo: str):
    line = (
        alt.Chart(hist)
        .mark_line(point=True)
        .encode(
            x=alt.X("Ciclo:N", title="Ciclo", sort=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Alumnos únicos con reprobación:Q", title="Alumnos únicos con reprobación"),
            tooltip=[
                alt.Tooltip("Ciclo:N", title="Ciclo"),
                alt.Tooltip("Alumnos únicos con reprobación:Q", title="Alumnos únicos"),
                alt.Tooltip("Total de materias reprobadas:Q", title="Total materias reprobadas"),
            ],
        )
        .properties(height=360, title=titulo)
    )

    labels = (
        alt.Chart(hist)
        .mark_text(dy=-10)
        .encode(
            x=alt.X("Ciclo:N", sort=None),
            y=alt.Y("Alumnos únicos con reprobación:Q"),
            text=alt.Text("Alumnos únicos con reprobación:Q"),
        )
    )

    return line + labels


def grafica_distribucion(dist: pd.DataFrame):
    return (
        alt.Chart(dist)
        .mark_bar()
        .encode(
            x=alt.X("Rango:N", title="Rango de calificación"),
            y=alt.Y("Total de materias reprobadas:Q", title="Total"),
            tooltip=[
                alt.Tooltip("Rango:N", title="Rango"),
                alt.Tooltip("Total de materias reprobadas:Q", title="Total"),
                alt.Tooltip("Porcentaje:Q", title="Porcentaje", format=".1f"),
            ],
        )
        .properties(height=320)
    )


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
        .mark_arc(innerRadius=50)
        .encode(
            theta=alt.Theta(f"{col_valor}:Q"),
            color=alt.Color(f"{col_categoria}:N", title="Motivo de baja"),
            tooltip=[
                alt.Tooltip(f"{col_categoria}:N", title="Motivo"),
                alt.Tooltip(f"{col_valor}:Q", title="Casos"),
            ],
        )
        .properties(height=330, title="Distribución de bajas")
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
            f"Este bloque corresponde a: "
            f"{area_ctx if area_ctx else 'todas las carreras'} | "
            f"{ciclo_sel if ciclo_sel != '(Todos)' else 'todos los ciclos'}"
        )

        try:
            res = bajas_retencion.resumen_bajas_por_filtros(
                ciclo=ciclo_int,
                area=area_ctx,
            )

            st.metric("Total de bajas", f"{res.get('n', 0):,}")

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
                        st.caption("No fue posible construir la gráfica con la estructura actual.")
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
    st.caption("Identificación de alumnos afectados, materias críticas y comparación por carrera.")

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
        st.warning("No hay registros de reprobación con calificaciones menores a 70.")
        return

    # =====================================================
    # FILTROS
    # =====================================================
    f = df.copy()
    area_ctx = None

    if vista == "Director de carrera" and carrera:
        carrera_norm = normalizar_texto(carrera)
        f = f[f["AREA_norm"] == carrera_norm]

        st.text_input("Programa académico", value=carrera, disabled=True)

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
                "Programa académico",
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
    # MENÚ
    # =====================================================
    st.sidebar.markdown("### Índice de reprobación")

    opciones = [
        "Resumen ejecutivo",
        "Comparativo por carrera",
        "Materias críticas",
        "Detalle de alumnos",
    ]

    vista_modulo = st.sidebar.radio(
        "Vista del módulo",
        opciones,
        key="nav_indice_reprobacion",
    )

    # =====================================================
    # KPIs
    # =====================================================
    kpis = calcular_kpis(f)
    mostrar_kpis(kpis)
    nota_promedio()
    bloque_lectura_simple(kpis)

    mostrar_contexto_bajas(ciclo_sel, area_ctx)

    st.divider()

    # =====================================================
    # RESUMEN EJECUTIVO
    # =====================================================
    if vista_modulo == "Resumen ejecutivo":
        st.subheader("Resumen ejecutivo")

        st.markdown("### 1. Materias que requieren atención prioritaria")
        tm = top_materias(f, 10)

        if tm.empty:
            st.info("No hay materias suficientes para mostrar.")
        else:
            st.altair_chart(grafica_top_materias(tm), use_container_width=True)
            st.dataframe(tm, use_container_width=True, hide_index=True)

        st.divider()

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### 2. Tendencia por ciclo")
            hist = historico_por_ciclo(f)
            st.altair_chart(
                grafica_historico(hist, f"Alumnos con reprobación — {carrera_txt}"),
                use_container_width=True,
            )
            st.dataframe(hist, use_container_width=True, hide_index=True)

        with col2:
            st.markdown("### 3. Distribución de calificaciones reprobatorias")
            dist = distribucion_calificaciones(f)
            if dist.empty:
                st.info("No hay calificaciones numéricas para construir la distribución.")
            else:
                st.altair_chart(grafica_distribucion(dist), use_container_width=True)
                st.dataframe(dist, use_container_width=True, hide_index=True)

    # =====================================================
    # COMPARATIVO POR CARRERA
    # =====================================================
    elif vista_modulo == "Comparativo por carrera":
        st.subheader("Comparativo por carrera")

        resumen = resumen_por_carrera(f)

        if len(resumen) < 2:
            st.info("Con el filtro actual solo hay una carrera. Para comparar, selecciona '(Todas)' en Programa académico.")
        else:
            st.markdown("### Carreras con más alumnos afectados")
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
                "Materias reprobadas por alumno": st.column_config.NumberColumn(
                    "Materias reprobadas por alumno", format="%.2f"
                ),
            },
        )

        st.caption(
            "Alumnos únicos evita duplicar personas. Total de materias reprobadas cuenta cada materia reprobada por cada alumno."
        )

    # =====================================================
    # MATERIAS CRÍTICAS
    # =====================================================
    elif vista_modulo == "Materias críticas":
        st.subheader("Materias críticas")

        n_top = st.slider("Número de materias a mostrar", 5, 30, 10)

        tabla = top_materias(f, n_top)

        if tabla.empty:
            st.warning("No se encontraron materias críticas.")
            return

        st.markdown("### Materias con más alumnos afectados")
        st.altair_chart(grafica_top_materias(tabla), use_container_width=True)

        st.markdown("### Tabla para planeación de regularización")
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

        st.info(
            "Uso sugerido: tomar las primeras materias del ranking para planear cursos de regularización, asesorías o seguimiento académico."
        )

    # =====================================================
    # DETALLE DE ALUMNOS
    # =====================================================
    elif vista_modulo == "Detalle de alumnos":
        st.subheader("Detalle de alumnos con reprobación")

        d1, d2, d3 = st.columns(3)

        materias = ["(Todas)"] + sorted(f["MATERIA"].dropna().unique().tolist())
        niveles = ["(Todos)"] + sorted(f["NIVEL"].dropna().unique().tolist())
        ciclos_det = ["(Todos)"] + sorted(f["CICLO"].dropna().unique().tolist())

        materia_f = d1.selectbox("Materia", materias, key="ir_f_materia")
        nivel_f = d2.selectbox("Nivel", niveles, key="ir_f_nivel")
        ciclo_f = d3.selectbox("Ciclo", ciclos_det, key="ir_f_ciclo_det")

        df_det = f.copy()

        if materia_f != "(Todas)":
            df_det = df_det[df_det["MATERIA"] == materia_f]

        if nivel_f != "(Todos)":
            df_det = df_det[df_det["NIVEL"] == nivel_f]

        if ciclo_f != "(Todos)":
            df_det = df_det[df_det["CICLO"] == ciclo_f]

        st.caption(f"Mostrando {len(df_det):,} registros de reprobación")

        cols = [
            "CICLO",
            "ESCUELA",
            "NIVEL",
            "AREA",
            "MATRICULA",
            "ALUMNO",
            "MATERIA",
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
