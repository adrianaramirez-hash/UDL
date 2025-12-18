# encuesta_calidad.py
import re
import textwrap
from typing import Dict, List, Tuple

import altair as alt
import gspread
import pandas as pd
import streamlit as st


# =========================
# Utilidades
# =========================
def _norm(s: str) -> str:
    return str(s).strip().lower().replace(" ", "").replace("_", "")


def _make_unique_headers(headers: List[str]) -> List[str]:
    """
    Google Forms suele repetir headers como '¿Por qué?'.
    gspread.get_all_records() falla con duplicados, por eso aquí los hacemos únicos:
    '¿Por qué?' -> '¿Por qué?__2', '¿Por qué?__3', etc.
    """
    counts: Dict[str, int] = {}
    out = []
    for h in headers:
        h0 = str(h).strip()
        if h0 == "":
            h0 = "col"
        counts[h0] = counts.get(h0, 0) + 1
        if counts[h0] == 1:
            out.append(h0)
        else:
            out.append(f"{h0}__{counts[h0]}")
    return out


def _worksheet_to_df(ws) -> pd.DataFrame:
    values = ws.get_all_values()
    if not values or len(values) < 2:
        return pd.DataFrame()
    headers = _make_unique_headers(values[0])
    rows = values[1:]
    df = pd.DataFrame(rows, columns=headers)
    # normaliza strings vacíos a NaN
    df = df.replace({"": pd.NA})
    return df


def _safe_float(x):
    try:
        if pd.isna(x):
            return pd.NA
        return float(x)
    except Exception:
        return pd.NA


def _coerce_year(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    return dt.dt.year


def _is_nonempty_text(x) -> bool:
    return (x is not None) and (not pd.isna(x)) and (str(x).strip() != "")


def _percent_match(series: pd.Series, target: str) -> float:
    """
    % de target dentro de respuestas NO vacías (denominador = respondieron algo).
    """
    if series is None or series.empty:
        return 0.0
    s = series.dropna().astype(str).str.strip()
    denom = len(s)
    if denom == 0:
        return 0.0
    num = (s.str.lower() == target.lower()).sum()
    return (num / denom) * 100.0


# =========================
# Carga desde Google Sheets
# =========================
@st.cache_data(show_spinner=False)
def _load_from_gsheets(sheet_id: str):
    sa = dict(st.secrets["gcp_service_account"])
    gc = gspread.service_account_from_dict(sa)
    sh = gc.open_by_key(sheet_id)

    # Resolver títulos reales (tolerante)
    wanted = [
        "Respuestas",
        "Mapa_Preguntas",
        "Catalogo_Servicio",
        "Respuestas_EE",
        "Mapa_Preguntas_EE",
        "Diccionario_Escalas",
    ]
    all_ws = sh.worksheets()
    titles = [w.title for w in all_ws]
    titles_norm = {_norm(t): t for t in titles}

    resolved = {}
    for w in wanted:
        wn = _norm(w)
        if wn in titles_norm:
            resolved[w] = titles_norm[wn]

    # DataFrames (si no existe, df vacío)
    dfs = {}
    for key in ["Respuestas", "Respuestas_EE"]:
        if key in resolved:
            ws = sh.worksheet(resolved[key])
            dfs[key] = _worksheet_to_df(ws)
        else:
            dfs[key] = pd.DataFrame()

    maps = {}
    for key in ["Mapa_Preguntas", "Mapa_Preguntas_EE"]:
        if key in resolved:
            ws = sh.worksheet(resolved[key])
            maps[key] = _worksheet_to_df(ws)
        else:
            maps[key] = pd.DataFrame()

    catalogo = pd.DataFrame()
    if "Catalogo_Servicio" in resolved:
        catalogo = _worksheet_to_df(sh.worksheet(resolved["Catalogo_Servicio"]))

    dicc = pd.DataFrame()
    if "Diccionario_Escalas" in resolved:
        dicc = _worksheet_to_df(sh.worksheet(resolved["Diccionario_Escalas"]))

    return dfs, maps, catalogo, dicc


def _merge_catalogo_virtual(df: pd.DataFrame, catalogo: pd.DataFrame) -> pd.DataFrame:
    """
    Une catálogo a Virtual/Mixto si existe.
    Espera que df tenga 'Selecciona el programa académico que estudias'
    y catalogo tenga una columna tipo 'Programa' y columnas de salida 'Servicio' / 'Carrera_Catalogo' o similares.
    """
    if df.empty or catalogo.empty:
        return df

    # Buscar col de programa en df
    prog_col_df = None
    for c in df.columns:
        if c.strip() == "Selecciona el programa académico que estudias":
            prog_col_df = c
            break
    if not prog_col_df:
        return df

    # Buscar col programa en catálogo
    prog_col_cat = None
    for c in catalogo.columns:
        if c.strip().lower() in ["programa", "programa_academico", "programaacademico"]:
            prog_col_cat = c
            break
    if not prog_col_cat:
        # Si el catálogo ya trae el mismo nombre
        if "Selecciona el programa académico que estudias" in catalogo.columns:
            prog_col_cat = "Selecciona el programa académico que estudias"
        else:
            return df

    # Buscar columnas de salida
    # Servicio
    svc_col = None
    for c in catalogo.columns:
        if c.strip().lower() in ["servicio", "modalidad_servicio"]:
            svc_col = c
            break

    # Carrera_Catalogo
    car_col = None
    for c in catalogo.columns:
        if c.strip().lower() in ["carrera_catalogo", "carrera", "programa_catalogo"]:
            car_col = c
            break

    # Merge
    cat = catalogo.copy()
    cat = cat.rename(columns={prog_col_cat: "__programa__"})
    df2 = df.copy()
    df2 = df2.rename(columns={prog_col_df: "__programa__"})

    dfm = df2.merge(cat, on="__programa__", how="left", suffixes=("", "_cat"))
    dfm = dfm.rename(columns={"__programa__": prog_col_df})

    # Normaliza columnas estándar si no existen
    if "Servicio" not in dfm.columns and svc_col:
        dfm["Servicio"] = dfm[svc_col]
    if "Carrera_Catalogo" not in dfm.columns and car_col:
        dfm["Carrera_Catalogo"] = dfm[car_col]

    return dfm


def _ensure_standard_cols(df: pd.DataFrame, modalidad_default: str) -> pd.DataFrame:
    df = df.copy()
    if "Modalidad" not in df.columns:
        df["Modalidad"] = modalidad_default

    # Servicio/Carrera_Catalogo: si no están, intenta usar el programa como fallback
    if "Servicio" not in df.columns:
        # fallback: si existe "Servicio de procedencia" en EE
        if "Servicio de procedencia" in df.columns:
            df["Servicio"] = df["Servicio de procedencia"]
        else:
            df["Servicio"] = modalidad_default

    if "Carrera_Catalogo" not in df.columns:
        if "Servicio de procedencia" in df.columns:
            df["Carrera_Catalogo"] = df["Servicio de procedencia"]
        elif "Selecciona el programa académico que estudias" in df.columns:
            df["Carrera_Catalogo"] = df["Selecciona el programa académico que estudias"]
        else:
            df["Carrera_Catalogo"] = pd.NA

    # Año
    if "Anio" not in df.columns:
        if "Marca temporal" in df.columns:
            df["Anio"] = _coerce_year(df["Marca temporal"])
        else:
            df["Anio"] = pd.NA

    return df


# =========================
# Mapeo secciones por prefijo
# =========================
SECTION_NAMES_VIRTUAL = {
    "DIR": "Director/ Coordinador",
    "APR": "Aprendizaje",
    "MAT": "Materiales en la plataforma",
    "EVA": "Evaluación del conocimiento",
    "SEAC": "Soporte académico / SEAC",
    "ADM": "Acceso a soporte administrativo",
    "COM": "Comunicación con compañeros",
    "REC": "Recomendación",
    "PLAT": "Plataforma SEAC",
    "UDL": "Comunicación con la Universidad",
}

SECTION_NAMES_EE = {
    "DIR_ESC": "Director/ Coordinador",
    "SER_ESC": "Servicios",
    "ACD_ESC": "Servicios académicos",
    "INS_ESC": "Instalaciones y equipo tecnológico",
    "AMB_ESC": "Ambiente escolar",
    "REC_ESC": "Recomendación",
}


def _infer_section_from_header_num(header_num: str) -> str:
    if not header_num or pd.isna(header_num):
        return "Sin sección"
    h = str(header_num).strip()

    # EE primero (más específico)
    for pref, name in SECTION_NAMES_EE.items():
        if h.startswith(pref + "_"):
            return name

    # Virtual
    for pref, name in SECTION_NAMES_VIRTUAL.items():
        if h.startswith(pref + "_"):
            return name

    return "Sin sección"


def _prepare_map(mapa: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza el mapa a:
    header_exacto | scale_code | header_num | seccion
    """
    if mapa.empty:
        return mapa

    # normaliza nombres de columnas
    cols = {c: c.strip() for c in mapa.columns}
    mapa = mapa.rename(columns=cols)

    required = ["header_exacto", "scale_code", "header_num"]
    for r in required:
        if r not in mapa.columns:
            raise ValueError(f"Mapa_Preguntas no tiene la columna requerida: {r}")

    mapa2 = mapa.copy()
    mapa2["header_exacto"] = mapa2["header_exacto"].astype(str).str.strip()
    mapa2["scale_code"] = mapa2["scale_code"].astype(str).str.strip()
    mapa2["header_num"] = mapa2["header_num"].astype(str).str.strip()
    mapa2["seccion"] = mapa2["header_num"].apply(_infer_section_from_header_num)

    # orden (por sufijo num si existe)
    def _ord(hnum: str):
        m = re.search(r"_(\d+)_num$", str(hnum))
        return int(m.group(1)) if m else 9999

    mapa2["orden"] = mapa2["header_num"].apply(_ord)
    return mapa2.sort_values(["seccion", "orden", "header_num"]).reset_index(drop=True)


def _list_comment_columns(df: pd.DataFrame) -> List[str]:
    if df.empty:
        return []
    patterns = [
        "comentario",
        "sugerencia",
        "descríb",
        "describ",
        "¿por qué",
        "por qué",
        "porque",
        "en caso afirmativo",
    ]
    cols = []
    for c in df.columns:
        cl = str(c).strip().lower()
        if any(p in cl for p in patterns):
            cols.append(c)
    # quita duplicados preservando orden
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _wrap_lines(s: str, width: int = 60) -> str:
    s = str(s).strip()
    return "\n".join(textwrap.wrap(s, width=width))


# =========================
# Cálculos
# =========================
def _compute_section_stats(
    df: pd.DataFrame,
    mapa: pd.DataFrame,
) -> Tuple[pd.DataFrame, float, float]:
    """
    Devuelve:
    - tabla por sección: seccion | promedio | preguntas
    - promedio_global_likert (excluye YESNO)
    - %sí global (solo YESNO)
    """
    if df.empty or mapa.empty:
        return pd.DataFrame(columns=["Sección", "Promedio", "Preguntas"]), float("nan"), float("nan")

    # columnas por tipo
    likert_nums = mapa.loc[mapa["scale_code"].str.upper() != "YESNO", "header_num"].tolist()
    yesno_nums = mapa.loc[mapa["scale_code"].str.upper() == "YESNO", "header_num"].tolist()

    # limpieza a num
    for c in set(likert_nums + yesno_nums):
        if c in df.columns:
            df[c] = df[c].apply(_safe_float)

    # global
    global_likert = float(pd.Series(df[likert_nums].stack()).mean()) if likert_nums else float("nan")
    global_yes = float(pd.Series(df[yesno_nums].stack()).mean()) * 100.0 if yesno_nums else float("nan")

    # por sección
    rows = []
    for seccion, g in mapa.groupby("seccion"):
        cols = g["header_num"].tolist()
        cols = [c for c in cols if c in df.columns]
        if not cols:
            continue
        prom = float(pd.Series(df[cols].stack()).mean())
        rows.append({"Sección": seccion, "Promedio": prom, "Preguntas": len(cols)})

    t = pd.DataFrame(rows)
    if not t.empty:
        t = t.sort_values("Promedio", ascending=False).reset_index(drop=True)
    return t, global_likert, global_yes


def _question_table_for_section(df: pd.DataFrame, mapa_sec: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve tabla por pregunta:
    Pregunta | Promedio | n_válidas | %No lo utilizo / %No sé (si aplica)
    """
    rows = []
    for _, r in mapa_sec.iterrows():
        header_txt = r["header_exacto"]
        header_num = r["header_num"]
        scale = str(r["scale_code"]).strip().upper()

        if header_num not in df.columns:
            continue

        s_num = pd.to_numeric(df[header_num], errors="coerce")
        mean = float(s_num.mean()) if s_num.notna().any() else float("nan")
        n_valid = int(s_num.notna().sum())

        pct_na = None
        pct_label = ""
        if header_txt in df.columns:
            if scale == "GRID6_NOUSO":
                pct_na = _percent_match(df[header_txt], "No lo utilizo")
                pct_label = "% No lo utilizo"
            elif scale == "GRID6_NOSE":
                pct_na = _percent_match(df[header_txt], "No sé")
                pct_label = "% No sé"

        row = {
            "Pregunta": header_txt,
            "Promedio": mean,
            "n válidas": n_valid,
        }
        if pct_na is not None:
            row[pct_label] = pct_na

        rows.append(row)

    t = pd.DataFrame(rows)
    if not t.empty:
        # orden por promedio desc, pero conserva NaN al final
        t["_ord"] = t["Promedio"].fillna(-9999)
        t = t.sort_values("_ord", ascending=False).drop(columns=["_ord"]).reset_index(drop=True)
    return t


def _chart_questions_in_section(section_name: str, qt: pd.DataFrame) -> alt.Chart:
    """
    Gráfica vertical comparando preguntas (P1..Pn) y promedio.
    No dibuja texto debajo de barras; el detalle va en tooltip y en lista P1->Pregunta.
    """
    if qt.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []}))

    dfc = qt.copy()
    dfc = dfc[dfc["Promedio"].notna()].reset_index(drop=True)
    if dfc.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []}))

    dfc["Clave"] = ["P" + str(i + 1) for i in range(len(dfc))]
    dfc["Pregunta_full"] = dfc["Pregunta"].astype(str)

    chart = (
        alt.Chart(dfc)
        .mark_bar()
        .encode(
            x=alt.X("Clave:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Promedio:Q", title="Promedio", scale=alt.Scale(domain=[0, 5])),
            tooltip=[
                alt.Tooltip("Clave:N", title="Pregunta"),
                alt.Tooltip("Promedio:Q", title="Promedio", format=".2f"),
                alt.Tooltip("Pregunta_full:N", title="Texto"),
            ],
        )
        .properties(height=240)
    )

    return chart


# =========================
# Render principal
# =========================
def render_encuesta_calidad(vista: str, carrera: str | None):
    st.subheader("Encuesta de calidad")

    sheet_id = st.secrets.get("app", {}).get("sheet_id", "")
    if not sheet_id:
        st.error("Falta configurar el sheet_id en secrets.")
        return

    with st.spinner("Cargando datos oficiales (Google Sheets)…"):
        dfs, maps, catalogo, _dicc = _load_from_gsheets(sheet_id)

    df_v = dfs.get("Respuestas", pd.DataFrame())
    df_ee = dfs.get("Respuestas_EE", pd.DataFrame())
    mapa_v = _prepare_map(maps.get("Mapa_Preguntas", pd.DataFrame()))
    mapa_ee = _prepare_map(maps.get("Mapa_Preguntas_EE", pd.DataFrame()))

    # Virtual/Mixto: intenta merge catálogo si aplica
    if not df_v.empty:
        df_v = _merge_catalogo_virtual(df_v, catalogo)
        df_v = _ensure_standard_cols(df_v, modalidad_default="Virtual/Mixto")

    # EE
    if not df_ee.empty:
        df_ee = _ensure_standard_cols(df_ee, modalidad_default="Escolarizados/Ejecutivas")

    # Unión
    frames = []
    if not df_v.empty:
        frames.append(df_v)
    if not df_ee.empty:
        frames.append(df_ee)

    if not frames:
        st.warning("No se encontraron hojas de respuestas disponibles en el Sheet maestro.")
        return

    df_all = pd.concat(frames, ignore_index=True, sort=False)

    # Selección de mapa por modalidad (para cálculos)
    maps_by_modalidad = []
    if not mapa_v.empty:
        maps_by_modalidad.append(("Virtual/Mixto", mapa_v))
    if not mapa_ee.empty:
        maps_by_modalidad.append(("Escolarizados/Ejecutivas", mapa_ee))

    if not maps_by_modalidad:
        st.error("No se encontraron mapas (Mapa_Preguntas / Mapa_Preguntas_EE).")
        return

    # =========================
    # FILTROS (SIN SIDEBAR)
    # =========================
    # opciones
    modalidades = sorted([m for m in df_all["Modalidad"].dropna().unique().tolist()])
    servicios = sorted([s for s in df_all["Servicio"].dropna().unique().tolist()])
    anios = sorted([int(a) for a in df_all["Anio"].dropna().unique().tolist() if str(a).isdigit()])

    colA, colB, colC, colD = st.columns([1.2, 1.0, 1.2, 1.4], gap="large")

    with colA:
        modalidad_sel = st.selectbox("Modalidad", ["(Todas)"] + modalidades, index=0)

    with colB:
        anio_sel = st.selectbox("Año", ["(Todos)"] + [str(a) for a in anios], index=0)

    with colC:
        servicio_sel = st.selectbox("Servicio", ["(Todos)"] + servicios, index=0)

    # Carrera: en DG sí, en Director de carrera se bloquea
    if vista == "Dirección General":
        carreras = sorted([c for c in df_all["Carrera_Catalogo"].dropna().unique().tolist()])
        with colD:
            carrera_sel = st.selectbox("Carrera", ["(Todas)"] + carreras, index=0)
    else:
        carrera_sel = carrera if carrera else ""
        with colD:
            st.markdown("**Carrera**")
            st.write(carrera_sel if carrera_sel else "—")

    # Aplica filtros
    dff = df_all.copy()

    if modalidad_sel != "(Todas)":
        dff = dff[dff["Modalidad"] == modalidad_sel]

    if anio_sel != "(Todos)":
        dff = dff[dff["Anio"].astype(str) == str(anio_sel)]

    if servicio_sel != "(Todos)":
        dff = dff[dff["Servicio"] == servicio_sel]

    if vista == "Dirección General":
        if carrera_sel != "(Todas)":
            dff = dff[dff["Carrera_Catalogo"] == carrera_sel]
    else:
        if carrera_sel:
            dff = dff[dff["Carrera_Catalogo"] == carrera_sel]

    st.caption(f"Registros filtrados: {len(dff)}")

    # Determina mapa activo según modalidad filtrada (si es “todas”, usa unión de mapas)
    if modalidad_sel != "(Todas)":
        mapa_activo = None
        for m, mp in maps_by_modalidad:
            if m == modalidad_sel:
                mapa_activo = mp
                break
        if mapa_activo is None:
            # fallback: usa el primero
            mapa_activo = maps_by_modalidad[0][1]
    else:
        # unión de mapas disponibles
        mapa_activo = pd.concat([mp for _, mp in maps_by_modalidad], ignore_index=True, sort=False)
        mapa_activo = mapa_activo.drop_duplicates(subset=["header_num"]).reset_index(drop=True)

    # =========================
    # TABS
    # =========================
    tab_resumen, tab_seccion, tab_coment = st.tabs(["Resumen", "Por sección", "Comentarios"])

    with tab_resumen:
        # KPIs
        t_sec, global_likert, global_yes = _compute_section_stats(dff, mapa_activo)

        k1, k2, k3 = st.columns(3, gap="large")
        with k1:
            st.metric("Respuestas", value=str(len(dff)))
        with k2:
            st.metric("Promedio global (Likert 1–5)", value="—" if pd.isna(global_likert) else f"{global_likert:.2f}")
        with k3:
            st.metric("% Sí (preguntas Sí/No)", value="—" if pd.isna(global_yes) else f"{global_yes:.1f}%")

        st.markdown("### Promedio por sección (Likert 1–5)")
        if t_sec.empty:
            st.info("No hay datos suficientes para calcular promedios por sección con los filtros actuales.")
        else:
            st.dataframe(
                t_sec.style.format({"Promedio": "{:.2f}"}),
                use_container_width=True,
                hide_index=True,
            )

            # Gráfica por sección (vertical)
            dfc = t_sec.copy()
            dfc["Sección_full"] = dfc["Sección"].astype(str)
            dfc["Clave"] = ["S" + str(i + 1) for i in range(len(dfc))]

            chart = (
                alt.Chart(dfc)
                .mark_bar()
                .encode(
                    x=alt.X("Clave:N", title=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Promedio:Q", title="Promedio", scale=alt.Scale(domain=[0, 5])),
                    tooltip=[
                        alt.Tooltip("Clave:N", title="Sección"),
                        alt.Tooltip("Promedio:Q", title="Promedio", format=".2f"),
                        alt.Tooltip("Sección_full:N", title="Nombre"),
                    ],
                )
                .properties(height=260)
            )
            st.altair_chart(chart, use_container_width=True)

            # Mapeo debajo (clave -> nombre)
            st.markdown("**Secciones (clave → nombre):**")
            for _, r in dfc.iterrows():
                st.write(f"- **{r['Clave']}**: {r['Sección_full']}")

    with tab_seccion:
        if mapa_activo.empty:
            st.info("No hay mapa de preguntas para mostrar secciones.")
        else:
            # agrupa por sección
            for seccion, mp_sec in mapa_activo.groupby("seccion"):
                with st.expander(seccion, expanded=False):
                    qt = _question_table_for_section(dff, mp_sec)

                    # promedio sección
                    # (promedio de todos los promedios de preguntas con datos)
                    sec_mean = float(pd.Series(qt["Promedio"]).dropna().mean()) if not qt.empty else float("nan")
                    st.markdown(
                        f"**Promedio de la sección:** {'—' if pd.isna(sec_mean) else f'{sec_mean:.2f}'}"
                    )

                    # tabla preguntas
                    if qt.empty:
                        st.info("Sin datos para esta sección con los filtros actuales.")
                        continue

                    st.dataframe(
                        qt.style.format({c: "{:.2f}" for c in qt.columns if c in ["Promedio", "% No lo utilizo", "% No sé"]}),
                        use_container_width=True,
                        hide_index=True,
                    )

                    # gráfica comparativa por sección (solo si hay 2+ preguntas con promedio)
                    chart_df = qt[qt["Promedio"].notna()].copy()
                    if len(chart_df) >= 2:
                        chart = _chart_questions_in_section(seccion, qt)
                        st.altair_chart(chart, use_container_width=True)

                        # Mapeo P1->pregunta debajo (sin estorbar la gráfica)
                        chart_df = chart_df.reset_index(drop=True)
                        chart_df["Clave"] = ["P" + str(i + 1) for i in range(len(chart_df))]

                        st.markdown("**Preguntas (clave → texto):**")
                        for _, r in chart_df.iterrows():
                            st.write(f"- **{r['Clave']}**: {_wrap_lines(r['Pregunta'], width=90)}")

    with tab_coment:
        cols = _list_comment_columns(dff)
        if not cols:
            st.info("No se detectaron columnas de comentarios/sugerencias en las respuestas filtradas.")
        else:
            search = st.text_input("Buscar texto en comentarios (opcional)", value="").strip().lower()

            # recolecta comentarios
            records = []
            for c in cols:
                if c not in dff.columns:
                    continue
                ser = dff[c]
                for idx, val in ser.items():
                    if not _is_nonempty_text(val):
                        continue
                    txt = str(val).strip()
                    if search and (search not in txt.lower()):
                        continue

                    rec = {
                        "Modalidad": dff.at[idx, "Modalidad"] if "Modalidad" in dff.columns else "",
                        "Servicio": dff.at[idx, "Servicio"] if "Servicio" in dff.columns else "",
                        "Carrera": dff.at[idx, "Carrera_Catalogo"] if "Carrera_Catalogo" in dff.columns else "",
                        "Año": dff.at[idx, "Anio"] if "Anio" in dff.columns else "",
                        "Campo": c,
                        "Comentario": txt,
                    }
                    records.append(rec)

            if not records:
                st.info("No hay comentarios que coincidan con el filtro actual.")
            else:
                com_df = pd.DataFrame(records)
                st.dataframe(com_df, use_container_width=True, hide_index=True)
