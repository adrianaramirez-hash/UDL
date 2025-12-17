import pandas as pd
import streamlit as st
import altair as alt


# ---------------------------
# Helpers
# ---------------------------
def _to_datetime_safe(s):
    return pd.to_datetime(s, errors="coerce", dayfirst=True)


def _detect_responses_sheet(xls: pd.ExcelFile) -> str:
    # Detecta la hoja de respuestas como la primera que NO sea Diccionario/Mapa/Catalogo
    skip = {"Diccionario", "Mapa_Preguntas", "Catalogo_Servicio"}
    candidates = [name for name in xls.sheet_names if name not in skip]
    if not candidates:
        raise ValueError("No encontré hoja de respuestas (distinta de Diccionario/Mapa_Preguntas/Catalogo_Servicio).")
    return candidates[0]


def _section_from_numcol(col: str) -> str:
    # Ej: DIR_Disponible_num -> DIR
    return col.split("_", 1)[0] if "_" in col else "OTROS"


def _is_yesno_col(col: str) -> bool:
    # Ajusta aquí si agregas más columnas SI/NO
    return col in {
        "ADM_ContactoExiste_num",
        "REC_Volveria_num",
        "REC_Recomendaria_num",
        "UDL_ViasComunicacion_num",
    }


@st.cache_data(show_spinner=False)
def _load_workbook(uploaded_file) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, str]:
    xls = pd.ExcelFile(uploaded_file)
    resp_sheet = _detect_responses_sheet(xls)

    df = pd.read_excel(uploaded_file, sheet_name=resp_sheet)
    mapa = pd.read_excel(uploaded_file, sheet_name="Mapa_Preguntas")

    catalogo = None
    if "Catalogo_Servicio" in xls.sheet_names:
        catalogo = pd.read_excel(uploaded_file, sheet_name="Catalogo_Servicio")

    return df, mapa, catalogo, resp_sheet


def _merge_catalogo(df: pd.DataFrame, catalogo: pd.DataFrame | None) -> pd.DataFrame:
    """
    Une Catalogo_Servicio usando:
    - df["Selecciona el programa académico que estudias"]
    - catalogo["programa"] -> servicio (y opcional carrera)
    """
    if catalogo is None:
        return df

    key = "Selecciona el programa académico que estudias"
    if key not in df.columns:
        return df

    cat = catalogo.copy()
    cat.columns = [c.strip().lower() for c in cat.columns]

    if "programa" not in cat.columns or "servicio" not in cat.columns:
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

    out["Servicio"] = out.get("Servicio", pd.Series(index=out.index, dtype="object")).fillna("SIN_CLASIFICAR")
    return out


def render_encuesta_calidad(vista: str, carrera: str | None):
    st.subheader("Encuesta de calidad")
    st.caption("Carga el archivo procesado (.xlsx) que incluye: hoja de respuestas + Diccionario + Mapa_Preguntas + Catalogo_Servicio.")

    uploaded = st.file_uploader("Sube el XLSX procesado", type=["xlsx"])

    if not uploaded:
        st.info("Sube el archivo para visualizar promedios y vistas tipo Google Forms (Resumen / Pregunta / Individual).")
        return

    df, mapa, catalogo, resp_sheet = _load_workbook(uploaded)
    df = _merge_catalogo(df, catalogo)

    # Normaliza fecha si existe
    if "Marca temporal" in df.columns:
        df["Marca temporal"] = _to_datetime_safe(df["Marca temporal"])

    # Mapa: solo lo que exista en el DF
    mapa = mapa.copy()
    mapa["section"] = mapa["header_num"].astype(str).apply(_section_from_numcol)
    mapa["exists"] = mapa["header_num"].isin(df.columns)
    mapa_ok = mapa[mapa["exists"]].copy()

    # Columnas num
    num_cols = [c for c in df.columns if c.endswith("_num")]
    likert_cols = [c for c in num_cols if not _is_yesno_col(c)]
    yesno_cols = [c for c in num_cols if _is_yesno_col(c)]

    # ---------------------------
    # Filtros
    # ---------------------------
    with st.sidebar:
        st.markdown("### Filtros – Encuesta de calidad")

        # Servicio (si existe por Catalogo_Servicio)
        if "Servicio" in df.columns:
            servicios = ["(Todos)"] + sorted(df["Servicio"].dropna().unique().tolist())
            servicio_sel = st.selectbox("Servicio", servicios, index=0)
        else:
            servicio_sel = "(Todos)"

        # Programa (si existe)
        key_prog = "Selecciona el programa académico que estudias"
        if key_prog in df.columns:
            programas = ["(Todos)"] + sorted(df[key_prog].dropna().unique().tolist())
            prog_sel = st.selectbox("Programa", programas, index=0)
        else:
            prog_sel = "(Todos)"

        # Vista Director de carrera: si en tu Catalogo_Servicio agregas columna carrera, podemos filtrar por ella.
        # (No uso tu lista de carreras como filtro directo porque el archivo trae "programas", no necesariamente las carreras SEP/UDL.)
        if vista == "Director de carrera" and carrera:
            st.info(f"Vista: Director de carrera ({carrera}).")
            # Si tienes Carrera_Catalogo, aplicamos filtro por coincidencia exacta
            usar_filtro_carrera = st.checkbox("Aplicar filtro por carrera (si existe en Catálogo)", value=True)
        else:
            usar_filtro_carrera = False

        # Fechas
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
    if usar_filtro_carrera and "Carrera_Catalogo" in f.columns:
        f = f[f["Carrera_Catalogo"] == carrera]
    if dr and "Marca temporal" in f.columns and len(dr) == 2:
        f = f[(f["Marca temporal"].dt.date >= dr[0]) & (f["Marca temporal"].dt.date <= dr[1])]

    st.caption(f"Hoja de respuestas detectada: **{resp_sheet}** | Registros filtrados: **{len(f)}**")

    tab1, tab2, tab3 = st.tabs(["Resumen", "Por pregunta", "Individual"])

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
        for section, g in mapa_ok.groupby("section"):
            cols = [c for c in g["header_num"].tolist() if c in f.columns and c in likert_cols]
            if not cols:
                continue
            val = pd.to_numeric(f[cols].stack(), errors="coerce").mean()
            rows.append({"Sección": section, "Promedio": val, "Preguntas": len(cols)})

        sec_df = pd.DataFrame(rows).sort_values("Promedio", ascending=False)
        st.dataframe(sec_df, use_container_width=True)

        if not sec_df.empty:
            chart = (
                alt.Chart(sec_df)
                .mark_bar()
                .encode(x=alt.X("Sección:N", sort="-y"), y=alt.Y("Promedio:Q"))
            )
            st.altair_chart(chart, use_container_width=True)

        st.markdown("### Sí/No (% Sí por pregunta)")
        if yesno_cols:
            yn = []
            for col in yesno_cols:
                pct = pd.to_numeric(f[col], errors="coerce").mean() * 100
                yn.append({"Columna": col, "% Sí": pct})
            st.dataframe(pd.DataFrame(yn).sort_values("% Sí", ascending=False), use_container_width=True)
        else:
            st.info("No se detectaron columnas Sí/No.")

    # ---------------------------
    # Por pregunta
    # ---------------------------
    with tab2:
        st.markdown("### Análisis por pregunta")

        if mapa_ok.empty:
            st.info("No hay mapeo de preguntas disponible.")
            return

        pregunta = st.selectbox("Selecciona una pregunta", mapa_ok["header_exacto"].tolist())
        row = mapa_ok[mapa_ok["header_exacto"] == pregunta].iloc[0]
        num_col = row["header_num"]
        scale = row["scale_code"]

        m1, m2 = st.columns(2)
        if num_col in f.columns:
            mean_val = pd.to_numeric(f[num_col], errors="coerce").mean()
            m1.metric("Promedio", f"{mean_val:.2f}" if pd.notna(mean_val) else "—")

            if _is_yesno_col(num_col):
                m2.metric("% Sí", f"{(mean_val*100):.1f}%" if pd.notna(mean_val) else "—")
            else:
                m2.metric("Escala", scale)

        # Distribución texto original (si existe)
        if pregunta in f.columns:
            dist = (
                f[pregunta].astype(str).replace("nan", pd.NA).dropna()
                .value_counts().reset_index()
            )
            dist.columns = ["Respuesta", "Conteo"]
            st.dataframe(dist, use_container_width=True)
        else:
            st.warning("No encuentro la columna de texto original con ese encabezado exacto.")

    # ---------------------------
    # Individual
    # ---------------------------
    with tab3:
        st.markdown("### Vista individual (tipo Forms)")

        if len(f) == 0:
            st.info("No hay registros con los filtros actuales.")
            return

        idx = st.selectbox("Selecciona registro", list(range(len(f))), format_func=lambda i: f"Registro #{i+1}")
        r = f.iloc[idx]

        head = {}
        if "Marca temporal" in f.columns:
            head["Marca temporal"] = r.get("Marca temporal")
        if "Servicio" in f.columns:
            head["Servicio"] = r.get("Servicio")
        key_prog = "Selecciona el programa académico que estudias"
        if key_prog in f.columns:
            head[key_prog] = r.get(key_prog)

        st.write(head)

        st.markdown("#### Preguntas cerradas (texto + num)")
        show = []
        for _, m in mapa_ok.iterrows():
            q = m["header_exacto"]
            ncol = m["header_num"]
            show.append({
                "Pregunta": q,
                "Respuesta": r.get(q, ""),
                "Num": r.get(ncol, ""),
            })
        st.dataframe(pd.DataFrame(show), use_container_width=True)

        st.markdown("#### Comentarios (abiertas)")
        open_cols = [
            c for c in f.columns
            if (not c.endswith("_num")) and any(k in c.lower() for k in ["¿por qué", "comentario", "sugerencia", "escríbelo", "escribelo"])
        ]
        for c in open_cols:
            val = r.get(c, "")
            if pd.notna(val) and str(val).strip():
                st.markdown(f"**{c}**")
                st.write(val)
