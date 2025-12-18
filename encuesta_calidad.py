# encuesta_calidad.py
from __future__ import annotations

import re
from datetime import date
from typing import Dict, List, Tuple, Optional

import altair as alt
import pandas as pd
import streamlit as st
import gspread


# =========================
# Helpers: lectura segura de Sheets (headers duplicados)
# =========================
def _make_unique_headers(headers: List[str]) -> List[str]:
    """
    Convierte headers duplicados en únicos:
    Ej: ['¿Por qué?', '¿Por qué?'] -> ['¿Por qué?', '¿Por qué? (2)']
    """
    seen: Dict[str, int] = {}
    out = []
    for h in headers:
        h0 = str(h).strip()
        if h0 == "":
            h0 = "SIN_TITULO"
        if h0 not in seen:
            seen[h0] = 1
            out.append(h0)
        else:
            seen[h0] += 1
            out.append(f"{h0} ({seen[h0]})")
    return out


def _worksheet_to_df(ws) -> pd.DataFrame:
    """
    Lee una hoja completa y crea DataFrame aun si hay headers duplicados.
    """
    values = ws.get_all_values()
    if not values:
        return pd.DataFrame()
    headers = _make_unique_headers(values[0])
    rows = values[1:]
    df = pd.DataFrame(rows, columns=headers)
    # Normaliza: convierte strings vacíos en NaN
    df = df.replace({"": None})
    return df


@st.cache_data(show_spinner=False)
def _load_from_gsheets(sheet_id: str):
    # Auth desde secrets
    sa = dict(st.secrets["gcp_service_account"])
    gc = gspread.service_account_from_dict(sa)

    sh = gc.open_by_key(sheet_id)

    # Normalizador tolerante
    def norm(x: str) -> str:
        return str(x).strip().lower().replace(" ", "").replace("_", "")

    # Intentamos cargar estas pestañas si existen
    needed = {
        "Respuestas": "Respuestas",
        "Mapa_Preguntas": "Mapa_Preguntas",
        "Catalogo_Servicio": "Catalogo_Servicio",
        # opcionales (para Escolarizados/Ejecutivas)
        "Respuestas_EE": "Respuestas_EE",
        "Mapa_Preguntas_EE": "Mapa_Preguntas_EE",
    }

    all_ws = sh.worksheets()
    titles = [ws.title for ws in all_ws]
    titles_norm = {norm(t): t for t in titles}

    resolved = {}
    for k, desired in needed.items():
        dn = norm(desired)
        if dn in titles_norm:
            resolved[k] = titles_norm[dn]

    # Requeridos mínimos:
    for req in ["Respuestas", "Mapa_Preguntas", "Catalogo_Servicio"]:
        if req not in resolved:
            raise ValueError(
                f"No encontré la pestaña requerida: {needed[req]} | "
                f"Pestañas visibles para el service account: {', '.join(titles)}"
            )

    ws_resp = sh.worksheet(resolved["Respuestas"])
    ws_map = sh.worksheet(resolved["Mapa_Preguntas"])
    ws_cat = sh.worksheet(resolved["Catalogo_Servicio"])

    df_v = _worksheet_to_df(ws_resp)
    mapa_v = _worksheet_to_df(ws_map)
    cat = _worksheet_to_df(ws_cat)

    # Opcionales:
    df_ee = pd.DataFrame()
    mapa_ee = pd.DataFrame()
    if "Respuestas_EE" in resolved and "Mapa_Preguntas_EE" in resolved:
        ws_resp_ee = sh.worksheet(resolved["Respuestas_EE"])
        ws_map_ee = sh.worksheet(resolved["Mapa_Preguntas_EE"])
        df_ee = _worksheet_to_df(ws_resp_ee)
        mapa_ee = _worksheet_to_df(ws_map_ee)

    return df_v, mapa_v, df_ee, mapa_ee, cat


# =========================
# Normalización de datos
# =========================
def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def _parse_date_series(s: pd.Series) -> pd.Series:
    # Maneja "Marca temporal" tipo Forms
    dt = pd.to_datetime(s, errors="coerce", dayfirst=False)
    return dt.dt.date


def _ensure_year(df: pd.DataFrame, fecha_col: str) -> pd.DataFrame:
    df = df.copy()
    df["_fecha"] = _parse_date_series(df[fecha_col])
    df["_anio"] = df["_fecha"].apply(lambda x: x.year if isinstance(x, date) else None)
    return df


def _clean_mapa(mapa: pd.DataFrame) -> pd.DataFrame:
    """
    Espera columnas:
      - header_exacto
      - scale_code
      - header_num
    y además usamos 'seccion' para agrupación (si no existe, se intenta derivar).
    """
    m = mapa.copy()
    # Normaliza nombres esperados
    # (si tu hoja usa Sección, seccion, etc.)
    colmap = {c: c.strip() for c in m.columns}
    m = m.rename(columns=colmap)

    # Columnas mínimas
    for req in ["header_exacto", "scale_code", "header_num"]:
        if req not in m.columns:
            raise ValueError(f"Mapa_Preguntas debe tener columna '{req}'")

    # Sección: si no existe, intenta inferir por prefijo de header_num (ej: MAT_, DIR_, etc.)
    if "seccion" not in m.columns:
        def infer_seccion(hn: str) -> str:
            hn = str(hn)
            pref = hn.split("_")[0] if "_" in hn else hn[:4]
            return pref

        m["seccion"] = m["header_num"].apply(infer_seccion)

    # Limpieza básica
    m["header_exacto"] = m["header_exacto"].astype(str).str.strip()
    m["scale_code"] = m["scale_code"].astype(str).str.strip()
    m["header_num"] = m["header_num"].astype(str).str.strip()
    m["seccion"] = m["seccion"].astype(str).str.strip()

    # Quita filas vacías
    m = m[(m["header_exacto"] != "") & (m["header_num"] != "")]
    return m


def _merge_catalogo(df: pd.DataFrame, catalogo: pd.DataFrame) -> pd.DataFrame:
    """
    Intenta enriquecer df con "Servicio" y "Carrera" usando Catalogo_Servicio.
    Si no se puede, crea Carrera a partir del campo de programa disponible.
    """
    d = df.copy()
    if d.empty:
        return d

    # Normaliza nombres de columnas en catálogo
    cat = catalogo.copy()
    cat = cat.rename(columns={c: c.strip() for c in cat.columns})
    # esperamos (ideal): programa | servicio | carrera
    # tolerante:
    prog_col = None
    for c in ["programa", "Programa", "PROGRAMA", "Programa académico", "programa_academico"]:
        if c in cat.columns:
            prog_col = c
            break

    serv_col = None
    for c in ["servicio", "Servicio", "SERVICIO", "modalidad", "Modalidad"]:
        if c in cat.columns:
            serv_col = c
            break

    carr_col = None
    for c in ["carrera", "Carrera", "CARRERA"]:
        if c in cat.columns:
            carr_col = c
            break

    # Columna “programa” dentro de respuestas
    resp_prog = _pick_col(
        d,
        [
            "Selecciona el programa académico que estudias",
            "Servicio de procedencia",
            "Programa",
            "Carrera",
        ],
    )

    # Si catálogo tiene programa y al menos servicio/carrera, hacemos merge
    if prog_col and resp_prog and (serv_col or carr_col):
        cat2 = cat[[prog_col] + ([serv_col] if serv_col else []) + ([carr_col] if carr_col else [])].copy()
        cat2 = cat2.rename(columns={prog_col: resp_prog})
        d = d.merge(cat2, on=resp_prog, how="left")

        # Renombramos a nombres estándar
        if serv_col and serv_col in d.columns:
            d = d.rename(columns={serv_col: "Servicio"})
        if carr_col and carr_col in d.columns:
            d = d.rename(columns={carr_col: "Carrera"})

    # Si no existe Carrera, la creamos con resp_prog
    if "Carrera" not in d.columns:
        if resp_prog:
            d["Carrera"] = d[resp_prog]
        else:
            d["Carrera"] = "Sin clasificar"

    # Si no existe Servicio, lo dejamos como “Sin clasificar”
    if "Servicio" not in d.columns:
        d["Servicio"] = "Sin clasificar"

    return d


def _coerce_numeric_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    d = df.copy()
    for c in cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


# =========================
# Parche principal: stats sin KeyError por columnas faltantes
# =========================
def _compute_section_stats(
    df: pd.DataFrame,
    mapa: pd.DataFrame,
) -> Tuple[pd.DataFrame, float, float]:
    """
    Devuelve:
    - tabla por sección: Sección | Promedio | Preguntas
    - promedio_global_likert (excluye YESNO)
    - %sí global (solo YESNO)

    IMPORTANTE: filtra columnas que realmente existan en df para evitar KeyError.
    """
    if df.empty or mapa.empty:
        return (
            pd.DataFrame(columns=["Sección", "Promedio", "Preguntas"]),
            float("nan"),
            float("nan"),
        )

    # columnas por tipo (desde mapa)
    likert_nums_all = mapa.loc[mapa["scale_code"].str.upper() != "YESNO", "header_num"].tolist()
    yesno_nums_all = mapa.loc[mapa["scale_code"].str.upper() == "YESNO", "header_num"].tolist()

    # SOLO las que existen en el df (esto evita KeyError)
    likert_nums = [c for c in likert_nums_all if c in df.columns]
    yesno_nums = [c for c in yesno_nums_all if c in df.columns]

    # coerce num
    df = _coerce_numeric_cols(df, list(set(likert_nums + yesno_nums)))

    # global likert
    if likert_nums:
        global_likert = float(df[likert_nums].stack().mean())
    else:
        global_likert = float("nan")

    # global yes/no (promedio *100)
    if yesno_nums:
        global_yes = float(df[yesno_nums].stack().mean()) * 100.0
    else:
        global_yes = float("nan")

    # por sección
    rows = []
    for seccion, g in mapa.groupby("seccion"):
        cols = [c for c in g["header_num"].tolist() if c in df.columns]
        if not cols:
            continue
        prom = float(df[cols].stack().mean())
        rows.append({"Sección": seccion, "Promedio": prom, "Preguntas": len(cols)})

    t = pd.DataFrame(rows)
    if not t.empty:
        t = t.sort_values("Promedio", ascending=False).reset_index(drop=True)

    return t, global_likert, global_yes


# =========================
# Visualización
# =========================
def _kpi_row(total: int, global_likert: float, global_yes: float):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Respuestas", f"{total:,}")
    with c2:
        st.metric("Promedio global (1–5)", "-" if pd.isna(global_likert) else f"{global_likert:.2f}")
    with c3:
        st.metric("% Sí (Sí/No)", "-" if pd.isna(global_yes) else f"{global_yes:.1f}%")


def _build_question_avgs(df: pd.DataFrame, mapa: pd.DataFrame, seccion: str) -> pd.DataFrame:
    g = mapa[mapa["seccion"] == seccion].copy()
    cols = [c for c in g["header_num"].tolist() if c in df.columns]
    if not cols:
        return pd.DataFrame(columns=["idx", "pregunta", "promedio"])

    df = _coerce_numeric_cols(df, cols)

    # promedios por pregunta
    avgs = []
    for _, r in g.iterrows():
        col = r["header_num"]
        if col not in df.columns:
            continue
        prom = float(pd.to_numeric(df[col], errors="coerce").mean())
        avgs.append({"pregunta": r["header_exacto"], "promedio": prom})

    out = pd.DataFrame(avgs)
    if out.empty:
        return pd.DataFrame(columns=["idx", "pregunta", "promedio"])

    out = out.sort_values("promedio", ascending=False).reset_index(drop=True)
    out["idx"] = out.index + 1
    return out[["idx", "pregunta", "promedio"]]


def _chart_questions_bar(avgs: pd.DataFrame, title: str):
    """
    Barras verticales (hacia arriba).
    Para evitar el problema de etiquetas encimadas, NO mostramos texto largo en el eje X.
    Mostramos índice y el detalle va en hover + una “leyenda” debajo (en texto).
    """
    if avgs.empty:
        st.info("No hay preguntas con datos numéricos para esta sección en el filtro actual.")
        return

    base = (
        alt.Chart(avgs)
        .mark_bar()
        .encode(
            x=alt.X("idx:O", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("promedio:Q", title="Promedio", scale=alt.Scale(domain=[0, 5])),
            tooltip=[
                alt.Tooltip("promedio:Q", title="Promedio", format=".2f"),
                alt.Tooltip("pregunta:N", title="Pregunta"),
            ],
        )
        .properties(height=240, title=title)
    )

    st.altair_chart(base, use_container_width=True)

    # “Leyenda” debajo: 2 columnas para que quede en 2–3 renglones
    items = [f"**{int(r.idx)}.** {r.pregunta}" for r in avgs.itertuples(index=False)]
    colA, colB = st.columns(2)
    half = (len(items) + 1) // 2
    with colA:
        st.markdown("<br>".join(items[:half]), unsafe_allow_html=True)
    with colB:
        st.markdown("<br>".join(items[half:]), unsafe_allow_html=True)


def _comments_table(df: pd.DataFrame, mapa: pd.DataFrame):
    """
    Muestra columnas de texto relevantes (comentarios, sugerencias, ¿Por qué?, etc.)
    """
    if df.empty:
        st.info("No hay registros para los filtros actuales.")
        return

    # Columnas numéricas conocidas (para excluir)
    known_num = set([c for c in mapa["header_num"].tolist() if c in df.columns])

    # Detecta campos de texto tipo comentario
    def is_comment_col(c: str) -> bool:
        c0 = c.lower()
        if c in known_num:
            return False
        keywords = [
            "comentario",
            "sugerencia",
            "¿por qué",
            "por qué",
            "descríb",
            "describe",
        ]
        return any(k in c0 for k in keywords)

    comment_cols = [c for c in df.columns if is_comment_col(c)]
    if not comment_cols:
        st.info("No detecté columnas de comentarios/sugerencias en este filtro.")
        return

    fecha_col = "_fecha" if "_fecha" in df.columns else None
    base_cols = []
    if fecha_col:
        base_cols.append(fecha_col)
    if "Servicio" in df.columns:
        base_cols.append("Servicio")
    if "Carrera" in df.columns:
        base_cols.append("Carrera")

    rows = []
    for _, r in df.iterrows():
        for c in comment_cols:
            val = r.get(c)
            if val is None:
                continue
            sval = str(val).strip()
            if sval == "":
                continue
            row = {}
            if fecha_col:
                row["Fecha"] = r.get(fecha_col)
            if "Servicio" in df.columns:
                row["Servicio"] = r.get("Servicio")
            if "Carrera" in df.columns:
                row["Carrera"] = r.get("Carrera")
            row["Campo"] = c
            row["Comentario"] = sval
            rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        st.info("No hay comentarios para los filtros actuales.")
        return

    st.dataframe(out, use_container_width=True, hide_index=True)


# =========================
# Render principal
# =========================
def render_encuesta_calidad(vista: str, carrera: Optional[str] = None):
    st.subheader("Encuesta de calidad")

    sheet_id = st.secrets["app"]["sheet_id"]

    with st.spinner("Cargando datos oficiales (Google Sheets)…"):
        df_v, mapa_v, df_ee, mapa_ee, catalogo = _load_from_gsheets(sheet_id)

    # Limpia mapas
    mapa_v = _clean_mapa(mapa_v)
    if not df_ee.empty and not mapa_ee.empty:
        mapa_ee = _clean_mapa(mapa_ee)

    # Enriquecemos con catálogo
    df_v = _merge_catalogo(df_v, catalogo)
    if not df_ee.empty:
        df_ee = _merge_catalogo(df_ee, catalogo)

    # Aseguramos fecha/año (Marca temporal)
    fecha_v = _pick_col(df_v, ["Marca temporal", "Timestamp", "Marca de tiempo"])
    if fecha_v:
        df_v = _ensure_year(df_v, fecha_v)
    else:
        df_v["_fecha"] = None
        df_v["_anio"] = None

    if not df_ee.empty:
        fecha_ee = _pick_col(df_ee, ["Marca temporal", "Timestamp", "Marca de tiempo"])
        if fecha_ee:
            df_ee = _ensure_year(df_ee, fecha_ee)
        else:
            df_ee["_fecha"] = None
            df_ee["_anio"] = None

    # Etiqueta de “Servicio” por dataset (si el catálogo no lo llenó bien)
    # Esto ayuda a tener un filtro consistente por modalidad
    if "Servicio" in df_v.columns:
        df_v["Servicio"] = df_v["Servicio"].fillna("Virtual")
    else:
        df_v["Servicio"] = "Virtual"

    if not df_ee.empty:
        if "Servicio" in df_ee.columns:
            df_ee["Servicio"] = df_ee["Servicio"].fillna("Escolarizados/Ejecutivas")
        else:
            df_ee["Servicio"] = "Escolarizados/Ejecutivas"

    # Unificamos datasets con una columna de control de mapa
    df_v["_map_key"] = "V"
    df_ee["_map_key"] = "EE" if not df_ee.empty else None

    df_all = pd.concat([df_v, df_ee], ignore_index=True) if not df_ee.empty else df_v.copy()

    # =========================
    # Filtros (SIN sidebar)
    # =========================
    # Servicio disponible
    servicios = sorted([s for s in df_all["Servicio"].dropna().unique().tolist()])
    anios = sorted([a for a in df_all["_anio"].dropna().unique().tolist()])

    # Bloqueo por vista
    is_dir_carrera = (vista == "Director de carrera")
    carrera_bloqueada = carrera if is_dir_carrera else None

    # UI filtros debajo del título (fila)
    c1, c2, c3 = st.columns([1.2, 1.0, 1.8])

    with c1:
        servicio_sel = st.selectbox("Servicio", ["(Todos)"] + servicios, index=0)

    with c2:
        anio_sel = st.selectbox("Año", ["(Todos)"] + [str(a) for a in anios], index=0)

    with c3:
        # Carrera: solo visible en Dirección General
        if is_dir_carrera:
            # no mostramos nada para evitar que el director lo “vea”
            carrera_sel = carrera_bloqueada
        else:
            carreras = sorted([s for s in df_all["Carrera"].dropna().unique().tolist()])
            carrera_sel = st.selectbox("Carrera", ["(Todas)"] + carreras, index=0)

    # Aplica filtros
    dff = df_all.copy()
    if servicio_sel != "(Todos)":
        dff = dff[dff["Servicio"] == servicio_sel]

    if anio_sel != "(Todos)":
        dff = dff[dff["_anio"] == int(anio_sel)]

    if is_dir_carrera and carrera_bloqueada:
        dff = dff[dff["Carrera"] == carrera_bloqueada]
    elif (not is_dir_carrera) and carrera_sel != "(Todas)":
        dff = dff[dff["Carrera"] == carrera_sel]

    st.caption(f"Registros filtrados: {len(dff)}")

    # Selección de mapa según dataset
    # Si el filtro mezcla V y EE, entonces trabajamos por bloque:
    # - Resumen global: se calcula con columnas existentes (parche lo soporta)
    # - Por sección: usamos el mapa correspondiente en cada bloque
    # Para simplificar: si hay mezcla, mostramos secciones por dataset.
    has_v = (dff["_map_key"] == "V").any()
    has_ee = (dff["_map_key"] == "EE").any()

    tabs = st.tabs(["Resumen", "Por sección", "Comentarios"])

    # =========================
    # RESUMEN
    # =========================
    with tabs[0]:
        if dff.empty:
            st.info("No hay datos para los filtros actuales.")
        else:
            # Promedios globales: aplicamos el parche por cada mapa y promediamos ponderado por cantidad de respuestas
            def block_stats(block_df: pd.DataFrame, block_map: pd.DataFrame):
                sec, gl, gy = _compute_section_stats(block_df, block_map)
                return sec, gl, gy

            global_likert_vals = []
            global_yes_vals = []
            weights = []

            if has_v:
                d_v = dff[dff["_map_key"] == "V"].copy()
                _, gl, gy = block_stats(d_v, mapa_v)
                if not pd.isna(gl):
                    global_likert_vals.append(gl)
                    weights.append(len(d_v))
                if not pd.isna(gy):
                    global_yes_vals.append(gy)

            if has_ee and (not df_ee.empty) and (not mapa_ee.empty):
                d_e = dff[dff["_map_key"] == "EE"].copy()
                _, gl, gy = block_stats(d_e, mapa_ee)
                if not pd.isna(gl):
                    global_likert_vals.append(gl)
                    weights.append(len(d_e))
                if not pd.isna(gy):
                    global_yes_vals.append(gy)

            # Promedio ponderado (si hay 2 bloques)
            if global_likert_vals and weights and sum(weights) > 0:
                global_likert = sum(v * w for v, w in zip(global_likert_vals, weights)) / sum(weights)
            else:
                global_likert = float("nan")

            # %sí: si hay dos bloques, promediamos simple (son proporciones distintas según instrumento)
            global_yes = float("nan")
            if global_yes_vals:
                global_yes = sum(global_yes_vals) / len(global_yes_vals)

            _kpi_row(total=len(dff), global_likert=global_likert, global_yes=global_yes)

            st.markdown("### Promedio por sección (1–5)")
            # Tabla por sección: si hay mezcla, mostramos por bloque con subtítulos
            if has_v:
                sec_v, _, _ = _compute_section_stats(dff[dff["_map_key"] == "V"].copy(), mapa_v)
                if not sec_v.empty:
                    st.markdown("**Virtual**")
                    st.dataframe(sec_v, use_container_width=True, hide_index=True)
            if has_ee and (not df_ee.empty) and (not mapa_ee.empty):
                sec_e, _, _ = _compute_section_stats(dff[dff["_map_key"] == "EE"].copy(), mapa_ee)
                if not sec_e.empty:
                    st.markdown("**Escolarizados/Ejecutivas**")
                    st.dataframe(sec_e, use_container_width=True, hide_index=True)

    # =========================
    # POR SECCIÓN (expanders con gráfica por sección)
    # =========================
    with tabs[1]:
        if dff.empty:
            st.info("No hay datos para los filtros actuales.")
        else:
            def render_block(block_name: str, block_df: pd.DataFrame, block_map: pd.DataFrame):
                sec_tbl, _, _ = _compute_section_stats(block_df, block_map)
                if sec_tbl.empty:
                    st.info(f"No hay secciones con datos numéricos para: {block_name}")
                    return

                # Orden por promedio desc
                for _, row in sec_tbl.iterrows():
                    seccion = row["Sección"]
                    prom = row["Promedio"]
                    n = int(row["Preguntas"])

                    with st.expander(f"{seccion} — Promedio {prom:.2f} ({n} preguntas)", expanded=False):
                        avgs = _build_question_avgs(block_df, block_map, seccion)
                        _chart_questions_bar(avgs, title=f"{seccion} — comparación de preguntas")

                        st.markdown("#### Preguntas y promedios")
                        if not avgs.empty:
                            st.dataframe(
                                avgs[["pregunta", "promedio"]].rename(
                                    columns={"pregunta": "Pregunta", "promedio": "Promedio"}
                                ),
                                use_container_width=True,
                                hide_index=True,
                            )

            # Si hay mezcla, separamos por instrumento
            if has_v and has_ee:
                st.markdown("### Virtual")
                render_block("Virtual", dff[dff["_map_key"] == "V"].copy(), mapa_v)

                st.markdown("---")
                st.markdown("### Escolarizados/Ejecutivas")
                if not df_ee.empty and not mapa_ee.empty:
                    render_block("Escolarizados/Ejecutivas", dff[dff["_map_key"] == "EE"].copy(), mapa_ee)
            else:
                # Un solo bloque
                if has_v:
                    render_block("Virtual", dff[dff["_map_key"] == "V"].copy(), mapa_v)
                else:
                    if not df_ee.empty and not mapa_ee.empty:
                        render_block("Escolarizados/Ejecutivas", dff[dff["_map_key"] == "EE"].copy(), mapa_ee)

    # =========================
    # COMENTARIOS
    # =========================
    with tabs[2]:
        if dff.empty:
            st.info("No hay datos para los filtros actuales.")
        else:
            # Si hay mezcla, mostramos comentarios por bloque (para que no se mezclen campos distintos)
            if has_v and has_ee:
                st.markdown("### Virtual")
                _comments_table(dff[dff["_map_key"] == "V"].copy(), mapa_v)
                st.markdown("---")
                st.markdown("### Escolarizados/Ejecutivas")
                if not df_ee.empty and not mapa_ee.empty:
                    _comments_table(dff[dff["_map_key"] == "EE"].copy(), mapa_ee)
            else:
                if has_v:
                    _comments_table(dff[dff["_map_key"] == "V"].copy(), mapa_v)
                else:
                    if not df_ee.empty and not mapa_ee.empty:
                        _comments_table(dff[dff["_map_key"] == "EE"].copy(), mapa_ee)
