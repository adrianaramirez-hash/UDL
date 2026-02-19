import pandas as pd
import streamlit as st
import altair as alt
import gspread
import textwrap
import unicodedata

SECTION_LABELS = {
    "DIR": "Director/Coordinación",
    "SER": "Servicios (Administrativos/Generales)",
    "ADM": "Acceso a soporte administrativo",
    "ACD": "Servicios académicos",
    "APR": "Aprendizaje",
    "EVA": "Evaluación del conocimiento",
    "SEAC": "Plataforma SEAC",
    "PLAT": "Plataforma SEAC",
    "SAT": "Plataforma SEAC",
    "MAT": "Materiales en la plataforma",
    "UDL": "Comunicación con la Universidad",
    "COM": "Comunicación con compañeros",
    "INS": "Instalaciones y equipo tecnológico",
    "AMB": "Ambiente escolar",
    "REC": "Recomendación / Satisfacción",
    "OTR": "Otros",
}

MAX_VERTICAL_QUESTIONS = 7
MAX_VERTICAL_SECTIONS = 7

SHEET_PROCESADO = "PROCESADO"
SHEET_MAPA = "Mapa_Preguntas"
SHEET_CATALOGO = "Catalogo_Servicio"

FINANZAS_SHEET_ID = "11qszwEcEA6vvy7XYGo-w_WkqPp1kxoNG5GfJB_Wcc4A"
FINANZAS_SHEET_NAME = "VISTA_FINANZAS_NUM"


def _section_from_numcol(col: str) -> str:
    return col.split("_", 1)[0] if "_" in col else "OTR"


def _to_datetime_safe(s):
    return pd.to_datetime(s, errors="coerce", dayfirst=True)


def _wrap_text(s: str, width: int = 18, max_lines: int = 3) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s).strip()
    if not s:
        return ""
    lines = textwrap.wrap(s, width=width)
    if len(lines) <= max_lines:
        return "\n".join(lines)
    kept = lines[:max_lines]
    kept[-1] = (kept[-1][:-1] + "…") if len(kept[-1]) >= 1 else "…"
    return "\n".join(kept)


def _mean_numeric(series: pd.Series):
    return pd.to_numeric(series, errors="coerce").mean()


def _norm_txt(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip().lower()
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    s = " ".join(s.split())
    return s


def _is_exec_text(norm_s: str) -> bool:
    return ("ejecutiva" in norm_s) or ("licenciatura ejecutiva" in norm_s) or ("lic. ejecutiva" in norm_s)


def _strip_exec_prefix(s: str) -> str:
    t = _norm_txt(s)
    for p in ["licenciatura ejecutiva:", "lic. ejecutiva:"]:
        if t.startswith(p):
            t = _norm_txt(t.replace(p, "", 1))
    return t


def _strip_generic_prefixes(s: str) -> str:
    t = _norm_txt(s)
    for p in ["licenciatura:", "lic."]:
        if t.startswith(p):
            t = _norm_txt(t.replace(p, "", 1))
    return t


def _match_carrera_mask_dc(series: pd.Series, target_raw: str) -> pd.Series:
    s_norm = series.astype(str).map(_norm_txt)
    t_norm = _norm_txt(target_raw)
    target_is_exec = _is_exec_text(t_norm)

    if target_is_exec:
        allowed = s_norm.map(_is_exec_text)
        t_base = _strip_exec_prefix(target_raw)
    else:
        allowed = ~s_norm.map(_is_exec_text)
        t_base = _strip_generic_prefixes(target_raw)

    m_exact = (s_norm == t_norm)
    if t_base and t_base != t_norm:
        m_exact = m_exact | (s_norm == t_base)

    m_exact = m_exact & allowed
    if m_exact.any():
        return m_exact

    m_cont = pd.Series(False, index=series.index)
    if t_norm:
        m_cont = m_cont | s_norm.str.contains(t_norm, na=False)
    if t_base and t_base != t_norm:
        m_cont = m_cont | s_norm.str.contains(t_base, na=False)

    return m_cont & allowed


def _bar_chart_auto(
    df_in: pd.DataFrame,
    category_col: str,
    value_col: str,
    value_domain: list,
    value_title: str,
    tooltip_cols: list,
    max_vertical: int,
    wrap_width_vertical: int = 18,
    wrap_width_horizontal: int = 30,
    height_per_row: int = 28,
    base_height: int = 260,
    hide_category_labels: bool = True,
):
    if df_in is None or df_in.empty:
        return None

    df = df_in.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col])
    if df.empty:
        return None

    n = len(df)
    cat_axis_vertical = alt.Axis(title=None, labels=not hide_category_labels, ticks=not hide_category_labels, labelAngle=0, labelLimit=0)
    cat_axis_horizontal = alt.Axis(title=None, labels=not hide_category_labels, ticks=not hide_category_labels, labelLimit=0)

    if n <= max_vertical:
        df["_cat_wrapped"] = df[category_col].apply(lambda x: _wrap_text(x, width=wrap_width_vertical, max_lines=3))
        return (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("_cat_wrapped:N", sort=alt.SortField(field=value_col, order="descending"), axis=cat_axis_vertical),
                y=alt.Y(f"{value_col}:Q", scale=alt.Scale(domain=value_domain), axis=alt.Axis(title=value_title)),
                tooltip=tooltip_cols,
            )
            .properties(height=max(320, base_height))
        )

    df["_cat_wrapped"] = df[category_col].apply(lambda x: _wrap_text(x, width=wrap_width_horizontal, max_lines=3))
    dynamic_height = max(base_height, n * height_per_row)

    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y("_cat_wrapped:N", sort=alt.SortField(field=value_col, order="descending"), axis=cat_axis_horizontal),
            x=alt.X(f"{value_col}:Q", scale=alt.Scale(domain=value_domain), axis=alt.Axis(title=value_title)),
            tooltip=tooltip_cols,
        )
        .properties(height=dynamic_height)
    )


def _pick_fecha_col(df: pd.DataFrame) -> str | None:
    for c in ["Marca temporal", "Marca Temporal", "Fecha", "fecha", "timestamp", "Timestamp"]:
        if c in df.columns:
            return c
    return None


def _ensure_prepa_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Servicio" not in out.columns:
        out["Servicio"] = "Preparatoria"
    if "Carrera_Catalogo" not in out.columns:
        out["Carrera_Catalogo"] = "Preparatoria"
    return out


def _get_url_for_modalidad(modalidad: str) -> str:
    URL_KEYS = {
        "Virtual / Mixto": "EC_VIRTUAL_URL",
        "Escolarizado / Ejecutivas": "EC_ESCOLAR_URL",
        "Preparatoria": "EC_PREPA_URL",
    }
    key = URL_KEYS.get(modalidad)
    if not key:
        raise KeyError(f"Modalidad no reconocida: {modalidad}")
    url = st.secrets.get(key, "").strip()
    if not url:
        raise KeyError(f"Falta configurar {key} en Secrets.")
    return url


def _resolver_modalidad_auto(vista: str, carrera: str | None) -> str:
    if vista == "Dirección General":
        return ""
    c = (carrera or "").strip().lower()
    if c == "preparatoria":
        return "Preparatoria"
    return "Escolarizado / Ejecutivas"


def _best_carrera_col(df: pd.DataFrame) -> str | None:
    candidates = [
        "Carrera_Catalogo",
        "Servicio",
        "Selecciona el programa académico que estudias",
        "Servicio de procedencia",
        "Programa",
        "Carrera",
    ]
    for c in candidates:
        if c in df.columns:
            vals = df[c].dropna().astype(str).str.strip()
            if vals.nunique() >= 2:
                return c
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _normalize_mapa_to_expected_schema(mapa: pd.DataFrame) -> pd.DataFrame:
    m = mapa.copy()
    cols = set(m.columns)

    if {"header_exacto", "scale_code", "header_num"}.issubset(cols):
        m["header_exacto"] = m["header_exacto"].astype(str).str.strip()
        m["scale_code"] = m["scale_code"].astype(str).str.strip()
        m["header_num"] = m["header_num"].astype(str).str.strip()
        return m

    if not {"header_raw", "header_id"}.issubset(cols):
        return m

    m["header_exacto"] = m["header_raw"].astype(str).str.strip()

    if "tipo" in cols:
        t = m["tipo"].astype(str).str.strip().str.upper()
        m["scale_code"] = t.map({"LIKERT": "LIKERT_1_5", "YESNO": "YESNO_0_1", "ABIERTA": "ABIERTA"}).fillna(t)
    else:
        m["scale_code"] = "LIKERT_1_5"

    hid = m["header_id"].astype(str).str.strip()
    if "tipo" in cols:
        t = m["tipo"].astype(str).str.strip().str.upper()
        m["header_num"] = hid + t.map({"ABIERTA": "_txt"}).fillna("_num")
    else:
        m["header_num"] = hid + "_num"

    if "section_code" in cols:
        m["section_code"] = m["section_code"].astype(str).str.strip()
    if "section_name" in cols:
        m["section_name"] = m["section_name"].fillna("").astype(str).str.strip()

    return m


def _auto_classify_numcols(df: pd.DataFrame, cols: list[str]) -> tuple[list[str], list[str]]:
    if not cols:
        return [], []
    dnum = df[cols].apply(pd.to_numeric, errors="coerce")
    dnum = dnum.loc[:, ~dnum.columns.duplicated()]
    maxs = dnum.max(axis=0, skipna=True)

    likert_cols = []
    for c in cols:
        if c not in maxs.index:
            continue
        v = maxs.loc[c]
        if pd.notna(v) and float(v) > 1.0:
            likert_cols.append(c)

    yesno_cols = [c for c in cols if c not in likert_cols]
    return likert_cols, yesno_cols


def _resolve_open_cols_from_mapa(m_open: pd.DataFrame) -> list[tuple[str, str, str]]:
    out = []
    if m_open is None or m_open.empty or "header_num" not in m_open.columns:
        return out
    for _, r in m_open.iterrows():
        sec = str(r.get("section_code", "OTR")).strip()
        lbl = str(r.get("header_exacto", "")).strip()
        col = str(r.get("header_num", "")).strip()
        if col and col.endswith("_txt"):
            out.append((sec, lbl, col))
    return out


def _render_open_comments_box(
    *,
    f: pd.DataFrame,
    items: list[tuple[str, str, str]],
    sec_code: str,
    title: str,
    key_prefix: str,
):
    if not items:
        st.caption("Sin ABIERTAS en esta sección (o no existe la columna *_txt).")
        return

    st.divider()
    st.markdown(f"**{title}**")

    labels = [lbl for _, lbl, _ in items]
    sel_lbl = st.selectbox("Campo de comentario", labels, index=0, key=f"{key_prefix}_sel_{sec_code}")
    col_map = {lbl: col for _, lbl, col in items}
    sel_col = col_map[sel_lbl]

    cA, cB = st.columns([2.2, 1.0])
    with cA:
        q = st.text_input("Buscar texto (contiene)", value="", key=f"{key_prefix}_q_{sec_code}")
    with cB:
        ver_todos = st.checkbox("Ver todos", value=False, key=f"{key_prefix}_all_{sec_code}")

    textos = f[sel_col].dropna().astype(str)
    textos = textos[textos.str.strip() != ""]

    if (not ver_todos) and q.strip():
        qn = q.strip().lower()
        textos = textos[textos.str.lower().str.contains(qn, na=False)]

    st.caption(f"Comentarios encontrados: **{len(textos)}**")
    st.dataframe(pd.DataFrame({sel_lbl: textos.reset_index(drop=True)}), use_container_width=True)


def _download_filtered_dataset(df: pd.DataFrame, filename_prefix: str):
    if df is None or getattr(df, "empty", True):
        st.sidebar.caption("Descarga: no hay registros para exportar con los filtros actuales.")
        return
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.sidebar.download_button(
        label="⬇️ Descargar dataset filtrado (CSV)",
        data=csv,
        file_name=f"{filename_prefix}.csv",
        mime="text/csv",
        use_container_width=True,
    )


@st.cache_data(show_spinner=False, ttl=300)
def _load_from_gsheets_by_url(url: str):
    sa = dict(st.secrets["gcp_service_account_json"])
    gc = gspread.service_account_from_dict(sa)
    sh = gc.open_by_url(url)

    def norm(x: str) -> str:
        return str(x).strip().lower().replace(" ", "").replace("_", "")

    titles = [ws.title for ws in sh.worksheets()]
    titles_norm = {norm(t): t for t in titles}

    def resolve(sheet_name: str) -> str | None:
        return titles_norm.get(norm(sheet_name))

    ws_pro = resolve(SHEET_PROCESADO)
    ws_map = resolve(SHEET_MAPA)
    ws_cat = resolve(SHEET_CATALOGO)

    missing = []
    if not ws_pro:
        missing.append(SHEET_PROCESADO)
    if not ws_map:
        missing.append(SHEET_MAPA)
    if missing:
        raise ValueError("No encontré pestañas: " + ", ".join(missing) + " | Disponibles: " + ", ".join(titles))

    def ws_to_df(ws_title: str) -> pd.DataFrame:
        ws = sh.worksheet(ws_title)
        values = ws.get_all_values()
        if not values:
            return pd.DataFrame()
        headers = [h.strip() for h in values[0]]
        rows = values[1:]
        return pd.DataFrame(rows, columns=headers).replace("", pd.NA)

    df = ws_to_df(ws_pro)
    mapa = ws_to_df(ws_map)
    catalogo = ws_to_df(ws_cat) if ws_cat else pd.DataFrame()
    return df, mapa, catalogo


@st.cache_data(show_spinner=False, ttl=300)
def _load_finanzas_num():
    sa = dict(st.secrets["gcp_service_account_json"])
    gc = gspread.service_account_from_dict(sa)
    sh = gc.open_by_key(FINANZAS_SHEET_ID)
    ws = sh.worksheet(FINANZAS_SHEET_NAME)
    values = ws.get_all_values()
    if not values:
        return pd.DataFrame()
    headers = [h.strip() for h in values[0]]
    rows = values[1:]
    return pd.DataFrame(rows, columns=headers).replace("", pd.NA)


def render_encuesta_calidad(vista: str | None = None, carrera: str | None = None):
    st.subheader("Encuesta de calidad")
    vista = (vista or "Dirección General").strip()

    if vista == "Dirección Finanzas":
        st.caption("Vista restringida para Dirección de Finanzas (solo datos administrativos autorizados).")
        try:
            with st.spinner("Cargando datos (Finanzas)…"):
                df = _load_finanzas_num()
        except Exception as e:
            st.error("No se pudo cargar la hoja VISTA_FINANZAS_NUM.")
            st.exception(e)
            return

        if df.empty:
            st.warning("La hoja VISTA_FINANZAS_NUM está vacía.")
            return

        fecha_col = _pick_fecha_col(df)
        if fecha_col:
            df[fecha_col] = _to_datetime_safe(df[fecha_col])

        years = ["(Todos)"]
        if fecha_col and df[fecha_col].notna().any():
            years += sorted(df[fecha_col].dt.year.dropna().unique().astype(int).tolist(), reverse=True)

        with st.sidebar:
            st.markdown("### Filtros — Encuesta de calidad")
            st.caption("Vista: Dirección Finanzas")
            year_sel = st.selectbox("Año", years, index=0, key="ec_df_year")
            st.divider()

        f = df.copy()
        if year_sel != "(Todos)" and fecha_col:
            f = f[f[fecha_col].dt.year == int(year_sel)]

        _download_filtered_dataset(
            f,
            filename_prefix=f"encuesta_calidad_DF_{year_sel if year_sel!='(Todos)' else 'TODOS'}",
        )

        st.divider()
        st.markdown("**Fuente:** VISTA_FINANZAS_NUM")
        st.caption(f"Registros filtrados: **{len(f)}**")
        if len(f) == 0:
            st.warning("No hay registros con los filtros seleccionados.")
            return

        open_cols = [
            c for c in f.columns
            if any(k in str(c).lower() for k in ["¿por qué", "por qué", "comentario", "sugerencia", "escríbelo", "escribelo"])
        ]

        base_exclude = set([c for c in ["Marca temporal", "Marca Temporal", "Selecciona el programa académico que estudias"] if c in f.columns])

        num_candidates = []
        for c in f.columns:
            if c in base_exclude or c in open_cols:
                continue
            s = pd.to_numeric(f[c], errors="coerce")
            if s.notna().any():
                num_candidates.append(c)

        if not num_candidates:
            st.warning("No encontré columnas numéricas en VISTA_FINANZAS_NUM.")
            st.dataframe(f.head(30), use_container_width=True)
            return

        likert_cols, yesno_cols = _auto_classify_numcols(f, num_candidates)

        tab1, tab2, tab3 = st.tabs(["Resumen", "Por pregunta", "Comentarios"])

        with tab1:
            c1, c2, c3 = st.columns(3)
            c1.metric("Respuestas", f"{len(f)}")
            if likert_cols:
                overall = pd.to_numeric(f[likert_cols].stack(), errors="coerce").mean()
                c2.metric("Promedio global (Likert)", f"{overall:.2f}" if pd.notna(overall) else "—")
            else:
                c2.metric("Promedio global (Likert)", "—")
            if yesno_cols:
                pct_yes = pd.to_numeric(f[yesno_cols].stack(), errors="coerce").mean() * 100
                c3.metric("% Sí (Sí/No)", f"{pct_yes:.1f}%" if pd.notna(pct_yes) else "—")
            else:
                c3.metric("% Sí (Sí/No)", "—")

        with tab2:
            st.markdown("### Detalle por pregunta")
            tipo_sel = st.radio("Tipo", ["Likert (1–5)", "Sí/No (0–1)"], horizontal=True)
            cols = likert_cols if "Likert" in tipo_sel else yesno_cols
            if not cols:
                st.info("No hay preguntas de este tipo con los filtros actuales.")
            else:
                pregunta = st.selectbox("Pregunta", cols)
                s = pd.to_numeric(f[pregunta], errors="coerce").dropna()
                st.caption(f"Respuestas válidas: {len(s)}")
                if "Likert" in tipo_sel:
                    st.metric("Promedio", f"{s.mean():.2f}" if len(s) else "—")
                else:
                    st.metric("% Sí", f"{(s.mean() * 100):.1f}%" if len(s) else "—")

        with tab3:
            st.markdown("### Comentarios y respuestas abiertas (Finanzas)")
            if not open_cols:
                st.info("No se detectaron columnas de comentarios para esta vista.")
            else:
                col_sel = st.selectbox("Selecciona el campo a revisar", open_cols)
                textos = f[col_sel].dropna().astype(str)
                textos = textos[textos.str.strip() != ""]
                st.caption(f"Entradas con texto: {len(textos)}")
                st.dataframe(pd.DataFrame({col_sel: textos}), use_container_width=True)

        return

    modalidad = _resolver_modalidad_auto(vista, carrera) if vista != "Dirección General" else None

    url = None
    df = None
    mapa = None

    with st.sidebar:
        st.markdown("### Filtros — Encuesta de calidad")
        if vista == "Dirección General":
            modalidad = st.selectbox(
                "Modalidad",
                ["Virtual / Mixto", "Escolarizado / Ejecutivas", "Preparatoria"],
                index=0,
                key="ec_dg_modalidad",
            )
        else:
            st.caption(f"Modalidad: {modalidad}")

    try:
        url = _get_url_for_modalidad(str(modalidad))
        with st.spinner("Cargando datos (Google Sheets)…"):
            df, mapa, _ = _load_from_gsheets_by_url(url)
    except Exception as e:
        st.error("No se pudieron cargar las hojas requeridas (PROCESADO / Mapa_Preguntas).")
        st.exception(e)
        return

    if df is None or df.empty:
        st.warning("La hoja PROCESADO está vacía.")
        return

    if str(modalidad) == "Preparatoria":
        df = _ensure_prepa_columns(df)

    fecha_col = _pick_fecha_col(df)
    if fecha_col:
        df[fecha_col] = _to_datetime_safe(df[fecha_col])

    mapa = _normalize_mapa_to_expected_schema(mapa)
    required_cols = {"header_exacto", "scale_code", "header_num"}
    if not required_cols.issubset(set(mapa.columns)):
        st.error("Mapa_Preguntas debe traer header_exacto, scale_code, header_num (o LAB: header_raw, header_id, tipo).")
        st.caption(f"Columnas detectadas: {list(mapa.columns)}")
        return

    mapa = mapa.copy()
    mapa["header_num"] = mapa["header_num"].astype(str).str.strip()
    mapa["scale_code"] = mapa["scale_code"].astype(str).str.strip()
    mapa["header_exacto"] = mapa["header_exacto"].astype(str).str.strip()

    if "section_code" in mapa.columns and mapa["section_code"].notna().any():
        mapa["section_code"] = mapa["section_code"].astype(str).str.strip()
    else:
        mapa["section_code"] = mapa["header_num"].apply(_section_from_numcol)

    if "section_name" in mapa.columns and mapa["section_name"].notna().any():
        mapa["section_name"] = mapa["section_name"].fillna("").astype(str).str.strip()
        mapa.loc[mapa["section_name"] == "", "section_name"] = mapa["section_code"]
    else:
        mapa["section_name"] = mapa["section_code"]

    mapa["section_name"] = mapa["section_name"].astype(str).str.strip()
    mask_abbrev = (mapa["section_name"] == mapa["section_code"]) | (mapa["section_name"].str.len() <= 4)
    mapa.loc[mask_abbrev, "section_name"] = mapa.loc[mask_abbrev, "section_code"].map(SECTION_LABELS).fillna(mapa.loc[mask_abbrev, "section_code"])

    mapa["exists"] = mapa["header_num"].isin(df.columns)
    mapa_ok = mapa[mapa["exists"]].copy()

    mapa_ok_num = mapa_ok[
        mapa_ok["header_num"].astype(str).str.endswith("_num")
        & (mapa_ok["scale_code"].astype(str).str.upper() != "ABIERTA")
    ].copy()

    mapa_ok_open = mapa_ok[mapa_ok["scale_code"].astype(str).str.upper() == "ABIERTA"].copy()
    open_items_all = _resolve_open_cols_from_mapa(mapa_ok_open)

    num_cols = [c for c in df.columns if str(c).endswith("_num")]
    if not num_cols:
        st.warning("No encontré columnas *_num en PROCESADO.")
        st.dataframe(df.head(30), use_container_width=True)
        return

    likert_cols, yesno_cols = _auto_classify_numcols(df, num_cols)

    years = ["(Todos)"]
    if fecha_col and df[fecha_col].notna().any():
        years += sorted(df[fecha_col].dt.year.dropna().unique().astype(int).tolist(), reverse=True)

    carrera_param_fija = (carrera is not None) and str(carrera).strip() != ""
    carrera_col = _best_carrera_col(df)
    carrera_sel = "(Todas)"

    with st.sidebar:
        year_sel = st.selectbox("Año", years, index=0, key="ec_year")

        if vista == "Dirección General":
            if carrera_param_fija:
                carrera_sel = str(carrera).strip()
                st.text_input("Carrera/Servicio (fijo)", value=carrera_sel, disabled=True, key="ec_carrera_fija")
            else:
                if carrera_col:
                    opts = ["(Todas)"] + sorted(df[carrera_col].dropna().astype(str).str.strip().unique().tolist())
                    carrera_sel = st.selectbox("Carrera/Servicio", opts, index=0, key="ec_carrera_sel")
                else:
                    st.info("No encontré columna válida para filtrar por Carrera/Servicio.")
                    carrera_sel = "(Todas)"
        else:
            carrera_sel = (carrera or "").strip()
            st.text_input("Carrera (fija por vista)", value=carrera_sel, disabled=True, key="ec_dc_carrera")

        st.divider()

    f = df.copy()

    if year_sel != "(Todos)" and fecha_col:
        f = f[f[fecha_col].dt.year == int(year_sel)]

    if vista == "Dirección General":
        if carrera_param_fija:
            if carrera_col:
                f = f[f[carrera_col].astype(str).str.strip() == str(carrera_sel).strip()]
            else:
                candidates = [c for c in ["Carrera_Catalogo", "Servicio", "Selecciona el programa académico que estudias"] if c in f.columns]
                if candidates:
                    target = str(carrera_sel).strip()
                    mask = False
                    for c in candidates:
                        mask = mask | (f[c].astype(str).str.strip() == target)
                    f = f[mask]
        else:
            if carrera_col and carrera_sel != "(Todas)":
                f = f[f[carrera_col].astype(str).str.strip() == str(carrera_sel).strip()]
    else:
        if str(modalidad) != "Preparatoria":
            candidates = [
                c for c in [
                    "Carrera_Catalogo",
                    "Servicio",
                    "Servicio de procedencia",
                    "Selecciona el programa académico que estudias",
                    "Programa",
                    "Carrera",
                ] if c in f.columns
            ]
            if not candidates:
                st.warning("No encontré columnas para filtrar por carrera.")
                st.caption(f"Columnas disponibles: {list(f.columns)}")
                return
            target = str(carrera_sel).strip()
            mask = pd.Series(False, index=f.index)
            for c in candidates:
                mask = mask | _match_carrera_mask_dc(f[c], target)
            f = f[mask]

    filename_prefix = f"encuesta_calidad_{str(modalidad).replace('/','-').replace(' ','_')}_{year_sel if year_sel!='(Todos)' else 'TODOS'}"
    if vista == "Dirección General":
        filename_prefix += f"_{(carrera_sel if carrera_sel!='(Todas)' else 'TODAS')}"
    else:
        filename_prefix += f"_{carrera_sel}"
    _download_filtered_dataset(f, filename_prefix=filename_prefix)

    st.caption(f"Hoja usada: **PROCESADO** | Registros filtrados: **{len(f)}**")
    if len(f) == 0:
        st.warning("No hay registros con los filtros seleccionados.")
        return

    if vista == "Dirección General":
        tab1, tab2, tab4, tab3 = st.tabs(["Resumen", "Por sección", "Comparativo entre carreras", "Comentarios"])
    else:
        tab1, tab2, tab3 = st.tabs(["Resumen", "Por sección", "Comentarios"])
        tab4 = None

    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Respuestas", f"{len(f)}")

        if likert_cols:
            overall = pd.to_numeric(f[likert_cols].stack(), errors="coerce").mean()
            c2.metric("Promedio global (Likert)", f"{overall:.2f}" if pd.notna(overall) else "—")
        else:
            c2.metric("Promedio global (Likert)", "—")

        if yesno_cols:
            pct_yes = pd.to_numeric(f[yesno_cols].stack(), errors="coerce").mean() * 100
            c3.metric("% Sí (Sí/No)", f"{pct_yes:.1f}%" if pd.notna(pct_yes) else "—")
        else:
            c3.metric("% Sí (Sí/No)", "—")

        st.divider()
        st.markdown("### Promedio por sección (Likert)")

        rows = []
        for (sec_code, sec_name), g in mapa_ok_num.groupby(["section_code", "section_name"]):
            cols = [c for c in g["header_num"].tolist() if c in f.columns and c in likert_cols]
            if not cols:
                continue
            val = pd.to_numeric(f[cols].stack(), errors="coerce").mean()
            if pd.isna(val):
                continue
            rows.append({"Sección": sec_name, "Promedio": float(val), "Preguntas": len(cols), "sec_code": sec_code})

        if not rows:
            st.info("No hay datos suficientes para promedios por sección (Likert).")
        else:
            sec_df = pd.DataFrame(rows).sort_values("Promedio", ascending=False)
            st.dataframe(sec_df.drop(columns=["sec_code"], errors="ignore"), use_container_width=True)

            sec_chart = _bar_chart_auto(
                df_in=sec_df,
                category_col="Sección",
                value_col="Promedio",
                value_domain=[1, 5],
                value_title="Promedio",
                tooltip_cols=["Sección", alt.Tooltip("Promedio:Q", format=".2f"), "Preguntas"],
                max_vertical=MAX_VERTICAL_SECTIONS,
                wrap_width_vertical=22,
                wrap_width_horizontal=36,
                base_height=320,
                hide_category_labels=True,
            )
            if sec_chart is not None:
                st.altair_chart(sec_chart, use_container_width=True)

        if yesno_cols:
            st.divider()
            st.markdown("### Sí/No — % Sí por pregunta")

            yn_rows = []
            for _, m in mapa_ok_num.iterrows():
                col = m["header_num"]
                if col not in yesno_cols or col not in f.columns:
                    continue
                mean_val = _mean_numeric(f[col])
                if pd.isna(mean_val):
                    continue
                yn_rows.append({"Pregunta": m["header_exacto"], "% Sí": float(mean_val) * 100})

            yn_df = pd.DataFrame(yn_rows).sort_values("% Sí", ascending=False) if yn_rows else pd.DataFrame()
            if not yn_df.empty:
                st.dataframe(yn_df, use_container_width=True)

    with tab2:
        st.markdown("### Desglose por sección (preguntas)")

        rows = []
        for (sec_code, sec_name), g in mapa_ok_num.groupby(["section_code", "section_name"]):
            cols = [c for c in g["header_num"].tolist() if c in f.columns and c in likert_cols]
            if not cols:
                continue
            val = pd.to_numeric(f[cols].stack(), errors="coerce").mean()
            if pd.isna(val):
                continue
            rows.append({"Sección": sec_name, "Promedio": float(val), "Preguntas": len(cols), "sec_code": sec_code})

        sec_df2 = pd.DataFrame(rows).sort_values("Promedio", ascending=False) if rows else pd.DataFrame()
        if sec_df2.empty:
            st.info("No hay datos suficientes para mostrar secciones.")
            return

        for _, r in sec_df2.iterrows():
            sec_code = r["sec_code"]
            sec_name = r["Sección"]
            sec_avg = r["Promedio"]

            with st.expander(f"{sec_name} — Promedio: {sec_avg:.2f}", expanded=False):
                mm = mapa_ok_num[mapa_ok_num["section_code"] == sec_code].copy()

                qrows = []
                for _, m in mm.iterrows():
                    col = m["header_num"]
                    if col not in f.columns:
                        continue
                    mean_val = _mean_numeric(f[col])
                    if pd.isna(mean_val):
                        continue
                    if col in yesno_cols:
                        qrows.append({"Pregunta": m["header_exacto"], "% Sí": float(mean_val) * 100, "Tipo": "Sí/No"})
                    elif col in likert_cols:
                        qrows.append({"Pregunta": m["header_exacto"], "Promedio": float(mean_val), "Tipo": "Likert"})

                qdf = pd.DataFrame(qrows)
                if qdf.empty:
                    st.info("Sin datos para esta sección.")
                else:
                    qdf_l = qdf[qdf["Tipo"] == "Likert"].copy()
                    if not qdf_l.empty:
                        qdf_l = qdf_l.sort_values("Promedio", ascending=False)
                        st.markdown("**Preguntas Likert (1–5)**")
                        show_l = qdf_l[["Pregunta", "Promedio"]].reset_index(drop=True)
                        st.dataframe(show_l, use_container_width=True)

                        chart_l = _bar_chart_auto(
                            df_in=show_l,
                            category_col="Pregunta",
                            value_col="Promedio",
                            value_domain=[1, 5],
                            value_title="Promedio",
                            tooltip_cols=[alt.Tooltip("Promedio:Q", format=".2f"), alt.Tooltip("Pregunta:N", title="Pregunta")],
                            max_vertical=MAX_VERTICAL_QUESTIONS,
                            wrap_width_vertical=24,
                            wrap_width_horizontal=40,
                            base_height=340,
                            hide_category_labels=True,
                        )
                        if chart_l is not None:
                            st.altair_chart(chart_l, use_container_width=True)

                    qdf_y = qdf[qdf["Tipo"] == "Sí/No"].copy()
                    if not qdf_y.empty:
                        qdf_y = qdf_y.sort_values("% Sí", ascending=False)
                        st.markdown("**Preguntas Sí/No**")
                        show_y = qdf_y[["Pregunta", "% Sí"]].reset_index(drop=True)
                        st.dataframe(show_y, use_container_width=True)

                        chart_y = _bar_chart_auto(
                            df_in=show_y,
                            category_col="Pregunta",
                            value_col="% Sí",
                            value_domain=[0, 100],
                            value_title="% Sí",
                            tooltip_cols=[alt.Tooltip("% Sí:Q", format=".1f"), alt.Tooltip("Pregunta:N", title="Pregunta")],
                            max_vertical=MAX_VERTICAL_QUESTIONS,
                            wrap_width_vertical=24,
                            wrap_width_horizontal=40,
                            base_height=340,
                            hide_category_labels=True,
                        )
                        if chart_y is not None:
                            st.altair_chart(chart_y, use_container_width=True)

                items_sec = [(sec, lbl, col) for (sec, lbl, col) in open_items_all if sec == sec_code and col in f.columns]
                _render_open_comments_box(
                    f=f,
                    items=items_sec,
                    sec_code=sec_code,
                    title="Comentarios de esta sección",
                    key_prefix="open_sec",
                )

    if tab4 is not None:
        with tab4:
            st.markdown("### Comparativo entre carreras por sección")
            carrera_col2 = _best_carrera_col(f)
            if not carrera_col2:
                st.warning("No se encontró columna válida de Carrera/Servicio.")
            elif carrera_param_fija:
                st.info("Para ver comparativo, la Carrera/Servicio debe estar en **(Todas)**.")
            else:
                for (sec_code, sec_name), g in mapa_ok_num.groupby(["section_code", "section_name"]):
                    cols = [c for c in g["header_num"].tolist() if c in f.columns and c in likert_cols]
                    if not cols:
                        continue

                    rows = []
                    for carrera_val, df_c in f.groupby(carrera_col2):
                        vals = pd.to_numeric(df_c[cols].stack(), errors="coerce")
                        mean_val = vals.mean()
                        if pd.isna(mean_val):
                            continue
                        rows.append({
                            "Carrera/Servicio": str(carrera_val).strip(),
                            "Promedio": round(float(mean_val), 2),
                            "Respuestas": int(len(df_c)),
                            "Preguntas": int(len(cols)),
                        })

                    if not rows:
                        continue

                    sec_comp = pd.DataFrame(rows).sort_values("Promedio", ascending=False).reset_index(drop=True)
                    with st.expander(f"{sec_name}", expanded=False):
                        st.dataframe(sec_comp, use_container_width=True)
                        chart = _bar_chart_auto(
                            df_in=sec_comp,
                            category_col="Carrera/Servicio",
                            value_col="Promedio",
                            value_domain=[1, 5],
                            value_title="Promedio",
                            tooltip_cols=[
                                alt.Tooltip("Carrera/Servicio:N", title="Carrera/Servicio"),
                                alt.Tooltip("Promedio:Q", format=".2f"),
                                "Respuestas",
                                "Preguntas",
                            ],
                            max_vertical=MAX_VERTICAL_SECTIONS,
                            wrap_width_vertical=20,
                            wrap_width_horizontal=36,
                            base_height=320,
                            hide_category_labels=True,
                        )
                        if chart is not None:
                            st.altair_chart(chart, use_container_width=True)

    with tab3:
        st.markdown("### Comentarios y respuestas abiertas")

        if not open_items_all:
            st.info("No hay preguntas ABIERTA configuradas en el mapa.")
            return

        sec_codes = sorted({sec for (sec, _, _) in open_items_all})
        sec_map_name = {code: SECTION_LABELS.get(code, code) for code in sec_codes}

        opts = ["(Todas)"] + [f"{code} — {sec_map_name.get(code, code)}" for code in sec_codes]
        sec_sel = st.selectbox("Sección", opts, index=0)

        if sec_sel == "(Todas)":
            pool = [(sec, lbl, col) for (sec, lbl, col) in open_items_all if col in f.columns]
            sec_key = "ALL"
        else:
            sec_code = sec_sel.split("—", 1)[0].strip()
            pool = [(sec, lbl, col) for (sec, lbl, col) in open_items_all if sec == sec_code and col in f.columns]
            sec_key = sec_code

        if not pool:
            st.warning("No encontré columnas *_txt en PROCESADO para la sección seleccionada.")
            return

        labels = [lbl for _, lbl, _ in pool]
        sel_lbl = st.selectbox("Campo de comentario", labels, index=0, key=f"open_global_sel_{sec_key}")
        col_map = {lbl: col for _, lbl, col in pool}
        sel_col = col_map[sel_lbl]

        cA, cB = st.columns([2.2, 1.0])
        with cA:
            q = st.text_input("Buscar texto (contiene)", value="", key=f"open_global_q_{sec_key}")
        with cB:
            ver_todos = st.checkbox("Ver todos", value=False, key=f"open_global_all_{sec_key}")

        textos = f[sel_col].dropna().astype(str)
        textos = textos[textos.str.strip() != ""]

        if (not ver_todos) and q.strip():
            qn = q.strip().lower()
            textos = textos[textos.str.lower().str.contains(qn, na=False)]

        st.caption(f"Comentarios encontrados: **{len(textos)}**")
        st.dataframe(pd.DataFrame({sel_lbl: textos.reset_index(drop=True)}), use_container_width=True)
