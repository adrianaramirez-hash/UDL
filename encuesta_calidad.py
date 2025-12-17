import pandas as pd
import streamlit as st
import altair as alt
import gspread


# ---------------------------
# Config visible (nombres completos)
# ---------------------------
SECTION_LABELS = {
    "DIR": "Director/Coordinador",
    "APR": "Aprendizaje",
    "MAT": "Materiales en la plataforma",
    "EVA": "Evaluación del conocimiento",
    "SEAC": "Soporte académico / SEAC",
    "ADM": "Acceso a soporte administrativo",
    "COM": "Comunicación con compañeros",
    "PLAT": "Plataforma SEAC",
    "UDL": "Comunicación con la universidad",
    "REC": "Recomendación",
}

YESNO_COLS = {
    "ADM_ContactoExiste_num",
    "REC_Volveria_num",
    "REC_Recomendaria_num",
    "UDL_ViasComunicacion_num",
}


def _section_from_numcol(col: str) -> str:
    return col.split("_", 1)[0] if "_" in col else "OTROS"


def _is_yesno_col(col: str) -> bool:
    return col in YESNO_COLS


def _to_datetime_safe(s):
    return pd.to_datetime(s, errors="coerce", dayfirst=True)


@st.cache_data(show_spinner=False)
def _load_from_gsheets(sheet_id: str):
    # Auth desde secrets
    sa = dict(st.secrets["gcp_service_account"])
    gc = gspread.service_account_from_dict(sa)

    sh = gc.open_by_key(sheet_id)

    # Normalizador para comparar títulos de pestañas
    def norm(x: str) -> str:
        return str(x).strip().lower().replace(" ", "").replace("_", "")

    targets = {
        "Respuestas": "Respuestas",
        "Mapa_Preguntas": "Mapa_Preguntas",
        "Catalogo_Servicio": "Catalogo_Servicio",
    }
    targets_norm = {k: norm(v) for k, v in targets.items()}

    all_ws = sh.worksheets()
    titles = [ws.title for ws in all_ws]
    titles_norm = {norm(t): t for t in titles}

    missing = []
    resolved = {}
    for key, tnorm in targets_norm.items():
        if tnorm in titles_norm:
            resolved[key] = titles_norm[tnorm]
        else:
            missing.append(targets[key])

    if missing:
        raise ValueError(
            "No encontré estas pestañas: "
            + ", ".join(missing)
            + " | Pestañas disponibles para el service account: "
            + ", ".join(titles)
        )

    ws_resp = sh.worksheet(resolved["Respuestas"])
    ws_map = sh.worksheet(resolved["Mapa_Preguntas"])
    ws_cat = sh.worksheet(resolved["Catalogo_Servicio"])

    df = pd.DataFrame(ws_resp.get_all_records())
    mapa = pd.DataFrame(ws_map.get_all_records())
    catalogo = pd.DataFrame(ws_cat.get_all_records())

    return df, mapa, catalogo


def _merge_catalogo(df: pd.DataFrame, catalogo: pd.DataFrame) -> pd.DataFrame:
    # Espera columnas: programa, servicio, carrera (carrera opcional)
    cat = catalogo.copy()
    cat.columns = [c.strip().lower() for c in cat.columns]

    if "programa" not in cat.columns or "servicio" not in cat.columns:
        return df

    key = "Selecciona el programa académico que estudias"
    if key not in df.columns:
        return df

    cat["programa"] = cat["programa"].astype(str).str.strip()
    cat["servicio"] = cat["servicio"].astype(str).str.strip()

    out = df.copy()
    out[key] = out[key].astype(str).str.strip()

    cols = ["programa", "servicio"]
    if "carrera" in cat.columns:
        cat["carrera"] = cat["carrera"].astype(str).str.strip()
        cols.append("carrera")

    out = out.merge(cat[cols], how="left", left_on=key, right_on="programa")
    out.drop(columns=["programa"], inplace=True, errors="ignore")

    out.rename(columns={"servicio": "Servicio"}, inplace=True)
    if "carrera" in out.columns:
        out.rename(columns={"carrera": "Carrera_Catalogo"}, inplace=True)

    out["Servicio"] = out["Servicio"].fillna("SIN_CLASIFICAR")
    return out


def render_encuesta_calidad(vista: str, carrera: str | None):
    st.subheader("Encuesta de calidad")

    sheet_id = st.secrets.get("app", {}).get("sheet_id", "")
    if not sheet_id:
        st.error("Falta app.sheet_id en Secrets.")
        return

    with st.spinner("Cargando datos oficiales (Google Sheets)…"):
        df, mapa, catalogo = _load_from_gsheets(sheet_id)

    df = _merge_catalogo(df, catalogo)

    # Fecha
    if "Marca temporal" in df.columns:
        df["Marca temporal"] = _to_datetime_safe(df["Marca temporal"])

    # Validación mínima del mapa
    required_cols = {"header_exacto", "scale_code", "header_num"}
    if not required_cols.issubset(set(mapa.columns)):
        st.error("La hoja 'Mapa_Preguntas' no trae: header_exacto, scale_code, header_num.")
        return

    mapa = mapa.copy()
    mapa["section_code"] = mapa["header_num"].astype(str).apply(_section_from_numcol)
    mapa["section_name"] = mapa["section_code"].map(SECTION_LABELS).fillna(mapa["section_code"])
    mapa["exists"] = mapa["header_num"].isin(df.columns)
    mapa_ok = mapa[mapa["exists"]].copy()

    num_cols = [c for c in df.columns if c.endswith("_num")]
    likert_cols = [c for c in num_cols if not _is_yesno_col(c)]
    yesno_cols = [c for c in num_cols if _is_yesno_col(c)]

    # ---------------------------
    # Filtros (sin duplicar carrera)
    # ---------------------------
    with st.sidebar:
        st.markdown("### Filtros – Encuesta de calidad")

        if "Servicio" in df.columns:
            servicios = ["(Todos)"] + sorted(df["Servicio"].dropna().unique().tolist())
            servicio_sel = st.selectbox("Servicio", servicios, index=0)
        else:
            servicio_sel = "(Todos)"

        key_prog = "Selecciona el programa académico que estudias"
        if vista == "Dirección General" and key_prog in df.columns:
            programas = ["(Todos)"] + sorted(df[key_prog].dropna().unique().tolist())
            prog_sel = st.selectbox("Programa", programas, index=0)
        else:
            prog_sel = "(Todos)"

        if vista == "Dirección General" and "Carrera_Catalogo" in df.columns:
            carreras = ["(Todas)"] + sorted(df["Carrera_Catalogo"].dropna().unique().tolist())
            carrera_sel = st.selectbox("Carrera", carreras, index=0)
        else:
            carrera_sel = "(Todas)"

        if "Marca temporal" in df.columns and df["Marca temporal"].notna().any():
            dmin = df["Marca temporal"].min().date()
            dmax = df["Marca temporal"].max().date()
            dr = st.date_input("Rango de fechas", value=(dmin, dmax))
        else:
            dr = None

    f = df.copy()
    if servicio_sel != "(Todos)" and "Servicio" in f.columns:
        f = f[f["Servicio"] == servicio_sel]

    if prog_sel != "(Todos)" and key_prog in f.columns:
        f = f[f[key_prog] == prog_sel]

    if vista == "Director de carrera" and carrera and "Carrera_Catalogo" in f.columns:
        f = f[f["Carrera_Catalogo"] == carrera]

    if vista == "Dirección General" and carrera_sel != "(Todas)" and "Carrera_Catalogo" in f.columns:
        f = f[f["Carrera_Catalogo"] == carrera_sel]

    if dr and "Marca temporal" in f.columns and len(dr) == 2:
        f = f[(f["Marca temporal"].dt.date >= dr[0]) & (f["Marca temporal"].dt.date <= dr[1])]

    if vista == "Director de carrera" and carrera:
        st.info(f"Vista Director de carrera: mostrando únicamente **{carrera}** (bloqueado desde la selección superior).")

    st.caption(f"Registros filtrados: **{len(f)}**")

    tab1, tab2, tab3 = st.tabs(["Resumen", "Por pregunta", "Comentarios"])

    # ---------------------------
    # Resumen
    # ---------------------------
    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Respuestas", f"{len(f)}")

        if likert_cols:
            overall = pd.to_numeric(f[likert_cols].stack(), errors="coerce").mean()
            c2.metric("Promedio global (Likert 1–5)", f"{overall:.2f}" if pd.notna(overall) else "—")
        else:
            c2.metric("Promedio global (Likert 1–5)", "—")

        if yesno_cols:
            pct_yes = pd.to_numeric(f[yesno_cols].stack(), errors="coerce").mean() * 100
            c3.metric("% Sí (preguntas Sí/No)", f"{pct_yes:.1f}%" if pd.notna(pct_yes) else "—")
        else:
            c3.metric("% Sí (preguntas Sí/No)", "—")

        st.markdown("### Promedio por sección (Likert 1–5)")
        rows = []
        for (_, sec_name), g in mapa_ok.groupby(["section_code", "section_name"]):
            cols = [c for c in g["header_num"].tolist() if c in f.columns and c in likert_cols]
            if not cols:
                continue
            val = pd.to_numeric(f[cols].stack(), errors="coerce").mean()
            rows.append({"Sección": sec_name, "Promedio": val, "Preguntas": len(cols)})

        sec_df = pd.DataFrame(rows).sort_values("Promedio", ascending=False)
        st.dataframe(sec_df, use_container_width=True)

        if not sec_df.empty:
            st.altair_chart(
                alt.Chart(sec_df).mark_bar().encode(
                    x=alt.X("Sección:N", sort="-y"),
                    y=alt.Y("Promedio:Q"),
                ),
                use_container_width=True,
            )

        st.markdown("### Promedio por pregunta")
        qrows = []
        for _, m in mapa_ok.iterrows():
            col = m["header_num"]
            if col not in f.columns:
                continue
            mean_val = pd.to_numeric(f[col], errors="coerce").mean()
            if pd.isna(mean_val):
                continue
            qrows.append({
                "Sección": m["section_name"],
                "Pregunta": m["header_exacto"],
                "Promedio": mean_val,
                "Tipo": "Sí/No" if _is_yesno_col(col) else "Likert",
                "% Sí": (mean_val * 100) if _is_yesno_col(col) else None,
            })

        qdf = pd.DataFrame(qrows)
        if not qdf.empty:
            qdf = qdf.sort_values(["Sección", "Promedio"], ascending=[True, False])
            st.dataframe(qdf, use_container_width=True)
        else:
            st.info("No hay promedios por pregunta disponibles con los filtros actuales.")

    # ---------------------------
    # Por pregunta
    # ---------------------------
    with tab2:
        if mapa_ok.empty:
            st.info("No hay mapeo de preguntas disponible.")
            return

        pregunta = st.selectbox("Selecciona una pregunta", mapa_ok["header_exacto"].tolist())
        row = mapa_ok[mapa_ok["header_exacto"] == pregunta].iloc[0]
        num_col = row["header_num"]

        c1, c2 = st.columns(2)
        mean_val = pd.to_numeric(f[num_col], errors="coerce").mean() if num_col in f.columns else None
        c1.metric("Promedio", f"{mean_val:.2f}" if pd.notna(mean_val) else "—")
        if _is_yesno_col(num_col) and pd.notna(mean_val):
            c2.metric("% Sí", f"{(mean_val * 100):.1f}%")
        else:
            c2.metric("Sección", row["section_name"])

        if pregunta in f.columns:
            dist = f[pregunta].astype(str).replace("nan", pd.NA).dropna().value_counts().reset_index()
            dist.columns = ["Respuesta", "Conteo"]
            st.dataframe(dist, use_container_width=True)
        else:
            st.warning("No encuentro la columna de texto original para esa pregunta (header_exacto).")

    # ---------------------------
    # Comentarios
    # ---------------------------
    with tab3:
        st.markdown("### Comentarios y respuestas abiertas")
        open_cols = [
            c for c in f.columns
            if (not c.endswith("_num")) and any(k in c.lower() for k in ["¿por qué", "comentario", "sugerencia", "escríbelo", "escribelo"])
        ]

        if not open_cols:
            st.info("No detecté columnas de comentarios con la heurística actual.")
            return

        col_sel = st.selectbox("Selecciona el campo a revisar", open_cols)
        textos = f[col_sel].dropna().astype(str)
        textos = textos[textos.str.strip() != ""]

        st.caption(f"Entradas con texto: {len(textos)}")
        st.dataframe(pd.DataFrame({col_sel: textos}), use_container_width=True)
