# encuesta_calidad.py
import re
from typing import Dict, List, Optional, Tuple

import altair as alt
import pandas as pd
import streamlit as st
import gspread


# =========================
# Helpers básicos
# =========================
def _norm(x: str) -> str:
    return str(x).strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def _wrap_text(s: str, width: int = 22, max_lines: int = 3) -> str:
    """Corta texto en 2-3 renglones para etiquetas del eje X."""
    s = str(s).strip()
    if not s:
        return s
    words = s.split()
    lines = []
    cur = ""
    for w in words:
        if len(cur) + len(w) + (1 if cur else 0) <= width:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur)
            cur = w
            if len(lines) >= max_lines - 1:
                break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    # Si sobró texto, no lo pegamos; el tooltip mostrará el completo
    return "\n".join([ln for ln in lines if ln])


# =========================
# Parche: columnas duplicadas
# =========================
def _ensure_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra columnas duplicadas: Col, Col (2), Col (3)..."""
    if df is None or df.empty:
        return df
    if df.columns.is_unique:
        return df

    cols = list(df.columns)
    counts: Dict[str, int] = {}
    new_cols = []
    for c in cols:
        c = str(c)
        if c not in counts:
            counts[c] = 1
            new_cols.append(c)
        else:
            counts[c] += 1
            new_cols.append(f"{c} ({counts[c]})")

    out = df.copy()
    out.columns = new_cols
    return out


def _coalesce_numbered_duplicates(df: pd.DataFrame, base: str) -> pd.DataFrame:
    """
    Si existen columnas base, base (2), base (3)... las fusiona en 'base' (fillna).
    """
    if df is None or df.empty:
        return df

    pattern = re.compile(rf"^{re.escape(base)} \((\d+)\)$")
    cols = [c for c in df.columns if c == base or pattern.match(str(c))]
    if len(cols) <= 1:
        return df

    s = df[cols[0]].copy()
    for c in cols[1:]:
        s = s.fillna(df[c])

    out = df.drop(columns=cols, errors="ignore").copy()
    out[base] = s
    return out


# =========================
# Lectura Google Sheets (robusta)
# =========================
def _worksheet_to_df(ws) -> pd.DataFrame:
    """
    Lee TODO como values para evitar:
    - error de encabezados duplicados (ej. '¿Por qué?')
    - gspread.get_all_records() que exige headers únicos
    """
    values = ws.get_all_values()
    if not values:
        return pd.DataFrame()

    header = values[0]
    rows = values[1:]

    # Headers únicos
    counts: Dict[str, int] = {}
    unique_header = []
    for h in header:
        h = str(h).strip()
        if h not in counts:
            counts[h] = 1
            unique_header.append(h)
        else:
            counts[h] += 1
            unique_header.append(f"{h} ({counts[h]})")

    df = pd.DataFrame(rows, columns=unique_header)
    # Normaliza vacíos
    df = df.replace({"": None})
    return df


@st.cache_data(show_spinner=False)
def _load_from_gsheets(sheet_id: str):
    # Auth desde secrets
    sa = dict(st.secrets["gcp_service_account"])
    gc = gspread.service_account_from_dict(sa)
    sh = gc.open_by_key(sheet_id)

    # Resolver pestañas (tolerante a espacios/guiones/_)
    need = {
        "Respuestas": "Respuestas",
        "Mapa_Preguntas": "Mapa_Preguntas",
        "Catalogo_Servicio": "Catalogo_Servicio",
        "Respuestas_EE": "Respuestas_EE",
        "Mapa_Preguntas_EE": "Mapa_Preguntas_EE",
    }

    titles = [ws.title for ws in sh.worksheets()]
    titles_norm = {_norm(t): t for t in titles}

    def resolve(name: str) -> Optional[str]:
        target = _norm(name)
        return titles_norm.get(target)

    # Obligatorias
    t_resp = resolve(need["Respuestas"])
    t_map = resolve(need["Mapa_Preguntas"])
    t_cat = resolve(need["Catalogo_Servicio"])
    if not t_resp or not t_map or not t_cat:
        raise ValueError(
            "No encuentro pestañas obligatorias. Requiero: Respuestas, Mapa_Preguntas, Catalogo_Servicio. "
            f"Disponibles: {titles}"
        )

    ws_resp = sh.worksheet(t_resp)
    ws_map = sh.worksheet(t_map)
    ws_cat = sh.worksheet(t_cat)

    df_v = _worksheet_to_df(ws_resp)
    mapa_v = _worksheet_to_df(ws_map)
    catalogo = _worksheet_to_df(ws_cat)

    # Opcionales (EE)
    t_resp_ee = resolve(need["Respuestas_EE"])
    t_map_ee = resolve(need["Mapa_Preguntas_EE"])

    df_ee = pd.DataFrame()
    mapa_ee = pd.DataFrame()
    if t_resp_ee and t_map_ee:
        df_ee = _worksheet_to_df(sh.worksheet(t_resp_ee))
        mapa_ee = _worksheet_to_df(sh.worksheet(t_map_ee))

    return df_v, mapa_v, catalogo, df_ee, mapa_ee


# =========================
# Mapeos y preparación
# =========================
def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    norm_map = {_norm(c): c for c in cols}
    for c in candidates:
        if _norm(c) in norm_map:
            return norm_map[_norm(c)]
    return None


def _merge_catalogo(df: pd.DataFrame, catalogo: pd.DataFrame) -> pd.DataFrame:
    """
    Une catalogo para obtener Servicio/Carrera cuando exista una llave común.
    Es tolerante: si no encuentra columnas, no rompe.
    """
    if df is None or df.empty or catalogo is None or catalogo.empty:
        return df

    df = df.copy()
    catalogo = catalogo.copy()

    # Candidatos de llave en respuestas
    key_df = _pick_col(
        df,
        [
            "Selecciona el programa académico que estudias",
            "Programa",
            "Servicio de procedencia",
        ],
    )
    # Candidatos de llave en catálogo
    key_cat = _pick_col(
        catalogo,
        [
            "Selecciona el programa académico que estudias",
            "Programa",
            "Servicio de procedencia",
            "programa",
        ],
    )

    col_serv = _pick_col(catalogo, ["Servicio", "servicio"])
    col_car = _pick_col(catalogo, ["Carrera", "carrera"])

    if not key_df or not key_cat or (not col_serv and not col_car):
        return df  # no hay con qué unir

    tmp = catalogo[[c for c in [key_cat, col_serv, col_car] if c]].copy()
    tmp = tmp.rename(columns={key_cat: "__key__"})
    df["__key__"] = df[key_df]

    merged = df.merge(tmp, on="__key__", how="left")
    merged = merged.drop(columns=["__key__"], errors="ignore")

    # Si ya existen columnas Servicio/Carrera, preferimos la existente y rellenamos con catálogo
    if col_serv and "Servicio" in merged.columns and col_serv in merged.columns:
        merged["Servicio"] = merged["Servicio"].fillna(merged[col_serv])
        if col_serv != "Servicio":
            merged = merged.drop(columns=[col_serv], errors="ignore")
    elif col_serv and col_serv in merged.columns and "Servicio" not in merged.columns:
        merged = merged.rename(columns={col_serv: "Servicio"})

    if col_car and "Carrera" in merged.columns and col_car in merged.columns:
        merged["Carrera"] = merged["Carrera"].fillna(merged[col_car])
        if col_car != "Carrera":
            merged = merged.drop(columns=[col_car], errors="ignore")
    elif col_car and col_car in merged.columns and "Carrera" not in merged.columns:
        merged = merged.rename(columns={col_car: "Carrera"})

    return merged


def _infer_section_from_header_num(header_num: str) -> str:
    hn = str(header_num or "")
    if hn.startswith("DIR_") or hn.startswith("DIRESC_") or hn.startswith("DIR_ESC_"):
        return "Director/ Coordinador"
    if hn.startswith("APR_"):
        return "Aprendizaje"
    if hn.startswith("MAT_"):
        return "Materiales en la plataforma"
    if hn.startswith("EVA_"):
        return "Evaluación del conocimiento"
    if hn.startswith("SEAC_"):
        return "Soporte académico / SEAC"
    if hn.startswith("ADM_"):
        return "Acceso a soporte administrativo"
    if hn.startswith("COM_"):
        return "Comunicación con compañeros"
    if hn.startswith("REC_"):
        return "Recomendación"
    if hn.startswith("PLAT_"):
        return "Plataforma SEAC"
    if hn.startswith("UDL_"):
        return "Comunicación con la Universidad"

    # EE
    if hn.startswith("SER_ESC_"):
        return "Servicios"
    if hn.startswith("ACD_ESC_"):
        return "Servicios académicos"
    if hn.startswith("INS_ESC_"):
        return "Instalaciones y equipo tecnológico"
    if hn.startswith("AMB_ESC_"):
        return "Ambiente escolar"
    if hn.startswith("REC_ESC_"):
        return "Recomendación"

    return "Sin sección"


def _prepare_mapa(mapa: pd.DataFrame) -> pd.DataFrame:
    """
    Espera al menos:
      header_exacto | header_num | scale_code
    Tolerante a variaciones de encabezados.
    """
    if mapa is None or mapa.empty:
        return pd.DataFrame(columns=["header_exacto", "header_num", "scale_code", "seccion", "pregunta", "grupo"])

    mapa = mapa.copy()

    c_hex = _pick_col(mapa, ["header_exacto", "header exacto", "header"])
    c_hnum = _pick_col(mapa, ["header_num", "header num"])
    c_scale = _pick_col(mapa, ["scale_code", "scale code", "scale"])

    if not c_hex or not c_hnum or not c_scale:
        raise ValueError("Mapa_Preguntas debe tener columnas: header_exacto, header_num, scale_code.")

    out = mapa[[c_hex, c_hnum, c_scale]].copy()
    out = out.rename(columns={c_hex: "header_exacto", c_hnum: "header_num", c_scale: "scale_code"})

    out["seccion"] = out["header_num"].apply(_infer_section_from_header_num)
    out["pregunta"] = out["header_exacto"].astype(str).str.strip()

    # Grupo para KPIs
    sc = out["scale_code"].astype(str).str.upper()

    def group_scale(s: str) -> str:
        if "YESNO" in s or "SI_NO" in s:
            return "YESNO"
        if "GRID6" in s:
            return "GRID6"
        # todo lo demás lo tratamos como 1–5
        return "LIKERT5"

    out["grupo"] = sc.apply(group_scale)
    return out


def _rename_numeric_columns_from_mapa(df: pd.DataFrame, mapa_pre: pd.DataFrame) -> pd.DataFrame:
    """
    Renombra columnas del DF usando header_exacto -> header_num si existen.
    No rompe si no encuentra una columna.
    """
    if df is None or df.empty or mapa_pre is None or mapa_pre.empty:
        return df

    df = df.copy()
    rename = {}
    cols = set(df.columns)

    # Match exacto por nombre original
    for _, r in mapa_pre.iterrows():
        hx = str(r["header_exacto"]).strip()
        hn = str(r["header_num"]).strip()
        if hx in cols:
            rename[hx] = hn

    # Match suave: si en respuestas viene con ']' final o espacios raros
    if not rename:
        # intentamos encontrar "parecidos" por normalizado
        norm_cols = {_norm(c): c for c in df.columns}
        for _, r in mapa_pre.iterrows():
            hx = str(r["header_exacto"]).strip()
            hn = str(r["header_num"]).strip()
            k = _norm(hx.replace("]", ""))
            if k in norm_cols:
                rename[norm_cols[k]] = hn

    if rename:
        df = df.rename(columns=rename)

    return df


def _parse_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    c_ts = _pick_col(df, ["Marca temporal", "Timestamp", "Marca de tiempo"])
    if c_ts:
        df["__timestamp__"] = pd.to_datetime(df[c_ts], errors="coerce")
    else:
        df["__timestamp__"] = pd.NaT
    df["__year__"] = df["__timestamp__"].dt.year
    return df


def _fill_zero_for_special_scales(df: pd.DataFrame, mapa_pre: pd.DataFrame) -> pd.DataFrame:
    """
    Si scale_code incluye NOUSO/NOSE, rellenamos NA como 0 (porque 0 es un valor real).
    """
    if df is None or df.empty or mapa_pre is None or mapa_pre.empty:
        return df
    df = df.copy()

    zero_scales = {"NOUSO", "NOSE"}
    for _, r in mapa_pre.iterrows():
        hn = str(r["header_num"]).strip()
        sc = str(r["scale_code"]).upper()
        if hn in df.columns and any(z in sc for z in zero_scales):
            df[hn] = pd.to_numeric(df[hn], errors="coerce").fillna(0)

    return df


def _build_long_numeric(df: pd.DataFrame, mapa_pre: pd.DataFrame, fuente: str) -> pd.DataFrame:
    """
    Regresa formato largo:
      response_id | timestamp | year | Servicio | Carrera | fuente | seccion | pregunta | grupo | header_num | value
    """
    if df is None or df.empty or mapa_pre is None or mapa_pre.empty:
        return pd.DataFrame(
            columns=[
                "response_id", "timestamp", "year", "Servicio", "Carrera", "fuente",
                "seccion", "pregunta", "grupo", "header_num", "value"
            ]
        )

    df = df.copy()
    df = _parse_timestamp(df)

    # id
    df["response_id"] = [f"{fuente}-{i+1}" for i in range(len(df))]

    # Asegura columnas Servicio/Carrera existen aunque estén vacías
    if "Servicio" not in df.columns:
        df["Servicio"] = None
    if "Carrera" not in df.columns:
        df["Carrera"] = None

    # Value vars: header_num
    value_vars = [hn for hn in mapa_pre["header_num"].tolist() if hn in df.columns]
    if not value_vars:
        return pd.DataFrame()

    # Numeric coercion
    for hn in value_vars:
        df[hn] = pd.to_numeric(df[hn], errors="coerce")

    # Melt
    id_vars = ["response_id", "__timestamp__", "__year__", "Servicio", "Carrera"]
    long = df[id_vars + value_vars].melt(
        id_vars=id_vars, value_vars=value_vars, var_name="header_num", value_name="value"
    )

    long = long.rename(columns={"__timestamp__": "timestamp", "__year__": "year"})
    long["fuente"] = fuente

    # Enriquecer con meta (sección/pregunta/grupo)
    meta = mapa_pre[["header_num", "seccion", "pregunta", "grupo"]].copy()
    long = long.merge(meta, on="header_num", how="left")

    return long


def _extract_comments(df: pd.DataFrame, fuente: str) -> pd.DataFrame:
    """
    Extrae columnas de texto relevantes (comentarios, sugerencias, por qué, descríbelo).
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["timestamp", "year", "Servicio", "Carrera", "fuente", "columna", "comentario"])

    df = df.copy()
    df = _parse_timestamp(df)

    if "Servicio" not in df.columns:
        df["Servicio"] = None
    if "Carrera" not in df.columns:
        df["Carrera"] = None

    comment_cols = []
    for c in df.columns:
        cn = _norm(c)
        if any(k in cn for k in ["comentario", "sugerencia", "porqué", "por que", "descr", "observ"]):
            comment_cols.append(c)

    if not comment_cols:
        return pd.DataFrame(columns=["timestamp", "year", "Servicio", "Carrera", "fuente", "columna", "comentario"])

    rows = []
    for c in comment_cols:
        sub = df[[c, "__timestamp__", "__year__", "Servicio", "Carrera"]].copy()
        sub = sub.rename(columns={c: "comentario", "__timestamp__": "timestamp", "__year__": "year"})
        sub["columna"] = c
        sub["fuente"] = fuente
        rows.append(sub)

    out = pd.concat(rows, ignore_index=True)
    out["comentario"] = out["comentario"].astype(str).str.strip()
    out = out[(out["comentario"].notna()) & (out["comentario"] != "") & (out["comentario"].str.lower() != "none")]
    return out


# =========================
# UI principal
# =========================
def render_encuesta_calidad(vista: str, carrera: Optional[str] = None):
    st.subheader("Encuesta de calidad")

    sheet_id = st.secrets.get("app", {}).get("sheet_id", None)
    if not sheet_id:
        st.error("Falta configurar en secrets: [app] sheet_id")
        return

    with st.spinner("Cargando datos oficiales (Google Sheets)…"):
        df_v, mapa_v_raw, catalogo_raw, df_ee, mapa_ee_raw = _load_from_gsheets(sheet_id)

    # Limpieza de columnas duplicadas
    df_v = _ensure_unique_columns(df_v)
    df_ee = _ensure_unique_columns(df_ee)

    # Merge catálogo (si aplica)
    catalogo_raw = _ensure_unique_columns(catalogo_raw)
    df_v = _merge_catalogo(df_v, catalogo_raw)
    df_ee = _merge_catalogo(df_ee, catalogo_raw)

    # Parche adicional: si merge dejó Servicio/Carrera duplicadas, colapsar
    df_v = _ensure_unique_columns(df_v)
    df_v = _coalesce_numbered_duplicates(df_v, "Servicio")
    df_v = _coalesce_numbered_duplicates(df_v, "Carrera")
    df_v = _ensure_unique_columns(df_v)

    df_ee = _ensure_unique_columns(df_ee)
    df_ee = _coalesce_numbered_duplicates(df_ee, "Servicio")
    df_ee = _coalesce_numbered_duplicates(df_ee, "Carrera")
    df_ee = _ensure_unique_columns(df_ee)

    # Mapas
    mapa_v = _prepare_mapa(_ensure_unique_columns(mapa_v_raw))
    mapa_ee = _prepare_mapa(_ensure_unique_columns(mapa_ee_raw)) if (mapa_ee_raw is not None and not mapa_ee_raw.empty) else pd.DataFrame()

    # Renombrar numéricas por header_num
    df_v = _rename_numeric_columns_from_mapa(df_v, mapa_v)
    df_v = _fill_zero_for_special_scales(df_v, mapa_v)

    if df_ee is not None and not df_ee.empty and mapa_ee is not None and not mapa_ee.empty:
        df_ee = _rename_numeric_columns_from_mapa(df_ee, mapa_ee)
        df_ee = _fill_zero_for_special_scales(df_ee, mapa_ee)

    # Fuente / modalidad (si no viene, la damos)
    if "Servicio" not in df_v.columns:
        df_v["Servicio"] = "Virtual"
    df_v["Servicio"] = df_v["Servicio"].fillna("Virtual")

    if df_ee is not None and not df_ee.empty:
        if "Servicio" not in df_ee.columns:
            df_ee["Servicio"] = "Escolarizados/Ejecutivas"
        df_ee["Servicio"] = df_ee["Servicio"].fillna("Escolarizados/Ejecutivas")

    # Long numeric
    long_v = _build_long_numeric(df_v, mapa_v, fuente="Virtual/Mixto")
    long_ee = (
        _build_long_numeric(df_ee, mapa_ee, fuente="Escolarizados/Ejecutivas")
        if (df_ee is not None and not df_ee.empty and mapa_ee is not None and not mapa_ee.empty)
        else pd.DataFrame()
    )

    # Concat LONG (más seguro)
    long_all = pd.concat([long_v, long_ee], ignore_index=True, sort=False) if not long_ee.empty else long_v.copy()

    # Comentarios
    com_v = _extract_comments(df_v, fuente="Virtual/Mixto")
    com_ee = _extract_comments(df_ee, fuente="Escolarizados/Ejecutivas") if (df_ee is not None and not df_ee.empty) else pd.DataFrame()
    comments_all = pd.concat([com_v, com_ee], ignore_index=True, sort=False) if not com_ee.empty else com_v.copy()

    # =========================
    # Filtros (SIN sidebar)
    # =========================
    # Vista Director: NO mostrar filtro de carrera (ya viene desde arriba)
    # Vista Dirección General: sí permitir filtrar carrera (opcional)
    c1, c2, c3 = st.columns([1.2, 1.0, 1.4])

    # Servicio (incluye modalidades)
    servicios = sorted([s for s in long_all["Servicio"].dropna().unique().tolist() if str(s).strip() != ""])
    servicio_sel = c1.selectbox("Servicio", ["(Todos)"] + servicios, index=0)

    # Año (desde marca temporal)
    years = sorted([int(y) for y in long_all["year"].dropna().unique().tolist() if str(y) != "nan"])
    year_sel = c2.selectbox("Año", ["(Todos)"] + years, index=0)

    carrera_sel = None
    if vista == "Dirección General":
        carreras = sorted([c for c in long_all["Carrera"].dropna().unique().tolist() if str(c).strip() != ""])
        carrera_sel = c3.selectbox("Carrera", ["(Todas)"] + carreras, index=0)
    else:
        # Director de carrera: se bloquea a lo que venga arriba
        carrera_sel = carrera

    # Aplicar filtros a LONG
    f = long_all.copy()

    if servicio_sel != "(Todos)":
        f = f[f["Servicio"] == servicio_sel]

    if year_sel != "(Todos)":
        f = f[f["year"] == int(year_sel)]

    if carrera_sel and carrera_sel != "(Todas)":
        f = f[f["Carrera"] == carrera_sel]

    # Comentarios con mismos filtros
    fc = comments_all.copy()
    if servicio_sel != "(Todos)":
        fc = fc[fc["Servicio"] == servicio_sel]
    if year_sel != "(Todos)":
        fc = fc[fc["year"] == int(year_sel)]
    if carrera_sel and carrera_sel != "(Todas)":
        fc = fc[fc["Carrera"] == carrera_sel]

    # Respuestas únicas (submissions)
    n_resp = f["response_id"].nunique()

    st.caption(f"Registros filtrados: {n_resp}")

    # =========================
    # Tabs (Resumen / Por sección / Comentarios)
    # =========================
    tab_res, tab_sec, tab_com = st.tabs(["Resumen", "Por sección", "Comentarios"])

    # -------------------------
    # RESUMEN
    # -------------------------
    with tab_res:
        # KPIs
        k1, k2, k3 = st.columns(3)
        k1.metric("Respuestas", f"{n_resp}")

        # Promedios por grupo
        lik = f[f["grupo"] == "LIKERT5"].copy()
        g6 = f[f["grupo"] == "GRID6"].copy()
        yn = f[f["grupo"] == "YESNO"].copy()

        lik_avg = float(lik["value"].mean()) if not lik.empty else None
        g6_avg = float(g6["value"].mean()) if not g6.empty else None
        yn_pct = float(yn["value"].mean() * 100) if not yn.empty else None

        k2.metric("Promedio global (1–5)", "-" if lik_avg is None else f"{lik_avg:.2f}")
        k3.metric("% Sí (Sí/No)", "-" if yn_pct is None else f"{yn_pct:.1f}%")

        # Promedio por sección (tabla)
        st.subheader("Promedio por sección")
        by_sec = (
            lik.groupby("seccion", as_index=False)["value"]
            .mean()
            .rename(columns={"value": "promedio"})
            .sort_values("promedio", ascending=False)
        )
        if by_sec.empty:
            st.info("No hay datos tipo 1–5 para graficar en este filtro.")
        else:
            st.dataframe(by_sec, use_container_width=True, hide_index=True)

            # Gráfica por sección (barras verticales)
            chart_df = by_sec.copy()
            chart_df["seccion_wrapped"] = chart_df["seccion"].apply(lambda x: _wrap_text(x, width=18, max_lines=3))
            chart = (
                alt.Chart(chart_df)
                .mark_bar()
                .encode(
                    x=alt.X("seccion_wrapped:N", sort="-y", axis=alt.Axis(title=None, labelAngle=0)),
                    y=alt.Y("promedio:Q", title="Promedio (1–5)"),
                    tooltip=[
                        alt.Tooltip("seccion:N", title="Sección"),
                        alt.Tooltip("promedio:Q", title="Promedio", format=".2f"),
                    ],
                )
                .properties(height=320, width="container")
            )
            st.altair_chart(chart, use_container_width=True)

    # -------------------------
    # POR SECCIÓN
    # -------------------------
    with tab_sec:
        if lik.empty:
            st.info("No hay datos tipo 1–5 para mostrar por sección con estos filtros.")
        else:
            # Orden de secciones por promedio
            sec_order = (
                lik.groupby("seccion")["value"].mean().sort_values(ascending=False).index.tolist()
            )

            for sec in sec_order:
                sec_df = lik[lik["seccion"] == sec].copy()
                sec_avg = float(sec_df["value"].mean()) if not sec_df.empty else None

                with st.expander(f"{sec} — Promedio: {sec_avg:.2f}" if sec_avg is not None else sec, expanded=False):
                    q = (
                        sec_df.groupby(["pregunta"], as_index=False)["value"]
                        .mean()
                        .rename(columns={"value": "promedio"})
                        .sort_values("promedio", ascending=False)
                    )
                    q["pregunta_wrapped"] = q["pregunta"].apply(lambda x: _wrap_text(x, width=26, max_lines=3))

                    # Gráfica: comparación entre preguntas dentro de la sección
                    chart_q = (
                        alt.Chart(q)
                        .mark_bar()
                        .encode(
                            x=alt.X(
                                "pregunta_wrapped:N",
                                sort="-y",
                                axis=alt.Axis(title=None, labelAngle=0, labelLimit=300),
                            ),
                            y=alt.Y("promedio:Q", title="Promedio (1–5)"),
                            tooltip=[
                                alt.Tooltip("pregunta:N", title="Pregunta"),
                                alt.Tooltip("promedio:Q", title="Promedio", format=".2f"),
                            ],
                        )
                        .properties(height=320, width="container")
                    )
                    st.altair_chart(chart_q, use_container_width=True)

                    # Tabla numérica
                    st.dataframe(q[["pregunta", "promedio"]], use_container_width=True, hide_index=True)

    # -------------------------
    # COMENTARIOS
    # -------------------------
    with tab_com:
        st.subheader("Comentarios")
        if fc.empty:
            st.info("No hay comentarios con estos filtros.")
        else:
            # Limpiar y presentar
            show = fc.copy()
            show["timestamp"] = pd.to_datetime(show["timestamp"], errors="coerce")
            show = show.sort_values("timestamp", ascending=False)

            cols_show = ["timestamp", "Servicio", "Carrera", "fuente", "columna", "comentario"]
            cols_show = [c for c in cols_show if c in show.columns]

            st.dataframe(show[cols_show], use_container_width=True, hide_index=True)
