def render_encuesta_calidad(vista: str | None = None, carrera: str | None = None):
    st.subheader("Encuesta de calidad")
    vista = (vista or "Dirección General").strip()

    # =========================================================
    # VISTA FINANZAS
    # =========================================================
    if vista == "Dirección Finanzas":
        st.caption("Vista restringida para Dirección de Finanzas (solo datos administrativos autorizados).")

        try:
            with st.spinner("Cargando datos (Finanzas)…"):
                df = _load_finanzas_num()
        except Exception as e:
            st.error("No se pudo cargar la hoja VISTA_FINANZAS_NUM.")
            st.exception(e)
            return

        if df is None or df.empty:
            st.warning("La hoja VISTA_FINANZAS_NUM está vacía.")
            return

        fecha_col = _pick_fecha_col(df)
        if fecha_col:
            df[fecha_col] = _to_datetime_safe(df[fecha_col])

        years = ["(Todos)"]
        if fecha_col and df[fecha_col].notna().any():
            years += sorted(df[fecha_col].dt.year.dropna().unique().astype(int).tolist(), reverse=True)

        servicio_col = _best_carrera_col(df)
        servicio_opts = None
        if servicio_col:
            servicio_vals = df[servicio_col].dropna().astype(str).str.strip()
            if servicio_vals.nunique() >= 2:
                servicio_opts = ["(Todos)"] + sorted(servicio_vals.unique().tolist())

        with st.sidebar:
            st.markdown("### Filtros — Encuesta de calidad")
            st.caption("Vista: Dirección Finanzas")
            year_sel = st.selectbox("Año", years, index=0, key="ec_df_year")
            if servicio_opts:
                servicio_sel = st.selectbox(f"{servicio_col}", servicio_opts, index=0, key="ec_df_servicio")
            else:
                servicio_sel = "(Todos)"
            st.divider()

        f = df.copy()
        if year_sel != "(Todos)" and fecha_col:
            f = f[f[fecha_col].dt.year == int(year_sel)]
        if servicio_opts and servicio_sel != "(Todos)":
            f = f[f[servicio_col].astype(str).str.strip() == str(servicio_sel).strip()]

        if len(f) == 0:
            st.warning("No hay registros con los filtros seleccionados.")
            return

        open_cols = [
            c
            for c in f.columns
            if any(
                k in str(c).lower()
                for k in [
                    "¿por qué",
                    "por qué",
                    "comentario",
                    "sugerencia",
                    "escríbelo",
                    "escribelo",
                    "observacion",
                    "observación",
                ]
            )
        ]

        base_exclude = {c for c in ["Marca temporal", "Marca Temporal"] if c in f.columns}
        num_candidates = []
        for c in f.columns:
            if c in base_exclude or c in open_cols:
                continue
            s = pd.to_numeric(f[c], errors="coerce")
            if s.notna().any():
                num_candidates.append(c)

        likert_cols, yesno_cols = _auto_classify_numcols(f, num_candidates)

        overall_likert = pd.to_numeric(f[likert_cols].stack(), errors="coerce").mean() if likert_cols else pd.NA
        overall_yes = (pd.to_numeric(f[yesno_cols].stack(), errors="coerce").mean() * 100) if yesno_cols else pd.NA

        kpis = pd.DataFrame(
            {
                "Indicador": ["Respuestas", "Promedio global (Likert)", "% Sí (Sí/No)"],
                "Valor": [
                    int(len(f)),
                    (round(float(overall_likert), 2) if pd.notna(overall_likert) else "—"),
                    (f"{round(float(overall_yes), 1)}%" if pd.notna(overall_yes) else "—"),
                ],
            }
        )

        fname = f"encuesta_calidad_DF_{year_sel if year_sel!='(Todos)' else 'TODOS'}"
        if servicio_opts and servicio_sel != "(Todos)":
            fname += f"_{str(servicio_sel).strip()}"
        _download_buttons_body(f, filename_prefix=fname, kpis_df=kpis)

        st.caption(f"Fuente: **VISTA_FINANZAS_NUM** | Registros filtrados: **{len(f)}**")

        mapa_rows = []
        for c in num_candidates:
            sec = _section_from_numcol(str(c))
            sec_name = SECTION_LABELS.get(sec, sec)
            mapa_rows.append(
                {
                    "header_exacto": str(c),
                    "header_num": str(c),
                    "scale_code": "AUTO",
                    "section_code": sec,
                    "section_name": str(sec_name),
                }
            )
        mapa_ok_num = pd.DataFrame(mapa_rows)

        open_items_all = []
        for c in open_cols:
            sec = _section_from_numcol(str(c))
            open_items_all.append((sec, str(c), str(c)))

        tab1, tab2, tab3 = st.tabs(["Resumen", "Por sección", "Comentarios"])

        with tab1:
            c1, c2, c3 = st.columns(3)
            c1.metric("Respuestas", f"{len(f)}")
            c2.metric("Promedio global (Likert)", f"{float(overall_likert):.2f}" if pd.notna(overall_likert) else "—")
            c3.metric("% Sí (Sí/No)", f"{float(overall_yes):.1f}%" if pd.notna(overall_yes) else "—")

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
                rows.append({"Sección": str(sec_name), "Promedio": float(val), "Preguntas": len(cols), "sec_code": sec_code})

            if rows:
                sec_df = pd.DataFrame(rows).sort_values("Promedio", ascending=False)
                st.dataframe(sec_df.drop(columns=["sec_code"], errors="ignore"), use_container_width=True)

                sec_chart = _bar_chart_auto(
                    df_in=sec_df,
                    category_col="Sección",
                    value_col="Promedio",
                    value_domain=[1, 5],
                    value_title="Promedio",
                    tooltip_cols=[
                        alt.Tooltip("Sección:N", title="Sección"),
                        alt.Tooltip("Promedio:Q", title="Promedio", format=".2f"),
                        alt.Tooltip("Preguntas:Q", title="Preguntas"),
                    ],
                    max_vertical=MAX_VERTICAL_SECTIONS,
                    wrap_width_vertical=22,
                    wrap_width_horizontal=36,
                    base_height=320,
                    hide_category_labels=True,
                )
                if sec_chart is not None:
                    st.altair_chart(sec_chart, use_container_width=True)
            else:
                st.info("No hay datos Likert para promedios por sección.")

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
                rows.append({"Sección": str(sec_name), "Promedio": float(val), "Preguntas": len(cols), "sec_code": sec_code})

            sec_df2 = pd.DataFrame(rows).sort_values("Promedio", ascending=False) if rows else pd.DataFrame()
            if sec_df2.empty:
                st.info("No hay datos suficientes para mostrar secciones.")
            else:
                for _, r in sec_df2.iterrows():
                    sec_code = r["sec_code"]
                    sec_name = str(r["Sección"])
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
                                qrows.append({"Pregunta": str(col), "% Sí": float(mean_val) * 100, "Tipo": "Sí/No"})
                            elif col in likert_cols:
                                qrows.append({"Pregunta": str(col), "Promedio": float(mean_val), "Tipo": "Likert"})

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
                                    tooltip_cols=[
                                        alt.Tooltip("Pregunta:N", title="Pregunta"),
                                        alt.Tooltip("Promedio:Q", title="Promedio", format=".2f"),
                                    ],
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
                                    tooltip_cols=[
                                        alt.Tooltip("Pregunta:N", title="Pregunta"),
                                        alt.Tooltip("% Sí:Q", title="% Sí", format=".1f"),
                                    ],
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
                            key_prefix="open_sec_df",
                        )

        with tab3:
            st.markdown("### Comentarios y respuestas abiertas (Finanzas)")

            if not open_items_all:
                st.info("No se detectaron columnas de comentarios para esta vista.")
                return

            sec_codes = sorted({sec for (sec, _, _) in open_items_all})
            sec_map_name = {code: SECTION_LABELS.get(code, code) for code in sec_codes}

            opts = ["(Todas)"] + [sec_map_name.get(code, code) for code in sec_codes]
            sec_sel = st.selectbox("Sección", opts, index=0, key="df_open_sec_sel")

            if sec_sel == "(Todas)":
                pool = [(sec, lbl, col) for (sec, lbl, col) in open_items_all if col in f.columns]
                sec_key = "ALL"
            else:
                sec_code = next(k for k, v in sec_map_name.items() if v == sec_sel)
                pool = [(sec, lbl, col) for (sec, lbl, col) in open_items_all if sec == sec_code and col in f.columns]
                sec_key = sec_code

            if not pool:
                st.warning("No hay columnas de comentarios con los filtros actuales.")
                return

            labels = [lbl for _, lbl, _ in pool]
            sel_lbl = st.selectbox("Campo de comentario", labels, index=0, key=f"df_open_field_{sec_key}")
            col_map = {lbl: col for _, lbl, col in pool}
            sel_col = col_map[sel_lbl]

            cA, cB = st.columns([2.2, 1.0])
            with cA:
                q = st.text_input("Buscar texto (contiene)", value="", key=f"df_open_q_{sec_key}")
            with cB:
                ver_todos = st.checkbox("Ver todos", value=False, key=f"df_open_all_{sec_key}")

            textos = f[sel_col].dropna().astype(str)
            textos = textos[textos.str.strip() != ""]

            if (not ver_todos) and q.strip():
                qn = q.strip().lower()
                textos = textos[textos.str.lower().str.contains(qn, na=False)]

            st.caption(f"Comentarios encontrados: **{len(textos)}**")
            st.dataframe(pd.DataFrame({sel_lbl: textos.reset_index(drop=True)}), use_container_width=True)

        return

    # =========================================================
    # VISTA GENERAL / DIRECTORES (corregido: Modalidad siempre seleccionable
    # + Carrera/Servicio por modalidad + Virtual en mantenimiento + sin mostrar "OTR")
    # =========================================================

    with st.sidebar:
        st.markdown("### Filtros — Encuesta de calidad")

        modalidad = st.selectbox(
            "Modalidad",
            ["Escolarizado / Ejecutivas", "Preparatoria", "Virtual / Mixto"],
            index=0,
            key=f"ec_modalidad__{vista}",
        )

        # Aviso y bloqueo SOLO para Virtual (pero conservando que siempre puedas cambiar modalidad)
        if str(modalidad) == "Virtual / Mixto":
            st.warning("🛠️ Virtual / Mixto: **En mantenimiento**.")
            st.info("Cambia la Modalidad para consultar datos disponibles.")
            st.stop()

    try:
        url = _get_url_for_modalidad(str(modalidad))
        with st.spinner("Cargando datos (Google Sheets)…"):
            df, mapa, _ = _load_from_gsheets_by_url(url)
    except Exception as e:
        st.error("No se pudieron cargar las hojas requeridas (PROCESADO / MAPA_PREGUNTAS).")
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
        st.error("MAPA_PREGUNTAS debe traer header_exacto, scale_code, header_num (o LAB: header_raw, header_id, tipo).")
        st.caption(f"Columnas detectadas: {list(mapa.columns)}")
        return

    mapa = mapa.copy()
    mapa["header_num"] = mapa["header_num"].astype(str).str.strip()
    mapa["scale_code"] = mapa["scale_code"].astype(str).str.strip()
    mapa["header_exacto"] = mapa["header_exacto"].astype(str).str.strip()

    # Sección: siempre por section_code/section_name del mapa (y si falta, se infiere)
    if "section_code" in mapa.columns and mapa["section_code"].notna().any():
        mapa["section_code"] = mapa["section_code"].astype(str).str.strip()
    else:
        mapa["section_code"] = mapa["header_num"].apply(_section_from_numcol)

    mapa["section_name"] = mapa.get("section_name", pd.Series([""] * len(mapa))).fillna("").astype(str).str.strip()

    # Si el mapa trae nombres vacíos o abreviaturas, los completamos; PERO "OTR" no se mostrará después
    mask_abbrev = (
        (mapa["section_name"] == "")
        | (mapa["section_name"] == mapa["section_code"])
        | (mapa["section_name"].str.len() <= 4)
    )
    mapa.loc[mask_abbrev, "section_name"] = (
        mapa.loc[mask_abbrev, "section_code"].map(SECTION_LABELS).fillna(mapa.loc[mask_abbrev, "section_code"])
    )
    mapa["section_name"] = mapa["section_name"].astype(str)

    mapa["exists"] = mapa["header_num"].isin(df.columns)
    mapa_ok = mapa[mapa["exists"]].copy()

    mapa_ok_num = mapa_ok[
        mapa_ok["header_num"].astype(str).str.endswith("_num")
        & (mapa_ok["scale_code"].astype(str).str.upper() != "ABIERTA")
    ].copy()

    mapa_ok_open = mapa_ok[mapa_ok["scale_code"].astype(str).str.upper() == "ABIERTA"].copy()
    open_items_all = _resolve_open_cols_from_mapa(mapa_ok_open)

    # **NO mostrar "OTR"** (se filtra en los outputs/agrupaciones)
    mapa_ok_num = mapa_ok_num[mapa_ok_num["section_code"].astype(str).str.strip() != "OTR"].copy()

    num_cols = [c for c in df.columns if str(c).endswith("_num")]
    if not num_cols:
        st.warning("No encontré columnas *_num en PROCESADO.")
        st.dataframe(df.head(30), use_container_width=True)
        return

    likert_cols, yesno_cols = _auto_classify_numcols(df, num_cols)

    years = ["(Todos)"]
    if fecha_col and df[fecha_col].notna().any():
        years += sorted(df[fecha_col].dt.year.dropna().unique().astype(int).tolist(), reverse=True)

    carrera_col = _best_carrera_col(df)

    # Sidebar: Año + Carrera/Servicio (SIEMPRE seleccionable, como DG)
    with st.sidebar:
        year_sel = st.selectbox("Año", years, index=0, key=f"ec_year__{vista}__{modalidad}")

        if carrera_col:
            opts = ["(Todas)"] + sorted(df[carrera_col].dropna().astype(str).str.strip().unique().tolist())
            # Si viene carrera por parámetro, la usamos como default si existe
            if carrera and str(carrera).strip() in opts:
                default_idx = opts.index(str(carrera).strip())
            else:
                default_idx = 0
            carrera_sel = st.selectbox("Carrera/Servicio", opts, index=default_idx, key=f"ec_carrera__{vista}__{modalidad}")
        else:
            st.info("No encontré columna válida para filtrar por Carrera/Servicio.")
            carrera_sel = "(Todas)"

        # Secciones visibles (muestra SOLO las secciones reales del mapa; sin OTR)
        sec_vis = (
            mapa_ok_num[["section_code", "section_name"]]
            .dropna()
            .drop_duplicates()
            .sort_values(["section_name"])
        )
        if not sec_vis.empty:
            st.caption("**Secciones visibles en esta modalidad:**")
            st.write(" • " + "\n • ".join(sec_vis["section_name"].astype(str).tolist()))

        st.divider()

    # Aplicación de filtros
    f = df.copy()

    if year_sel != "(Todos)" and fecha_col:
        f = f[f[fecha_col].dt.year == int(year_sel)]

    if carrera_col and carrera_sel != "(Todas)":
        f = f[f[carrera_col].astype(str).str.strip() == str(carrera_sel).strip()]

    if len(f) == 0:
        st.warning("No hay registros con los filtros seleccionados.")
        return

    filename_prefix = f"encuesta_calidad_{str(modalidad).replace('/','-').replace(' ','_')}_{year_sel if year_sel!='(Todos)' else 'TODOS'}_{(carrera_sel if carrera_sel!='(Todas)' else 'TODAS')}"
        overall_likert = pd.to_numeric(f[likert_cols].stack(), errors="coerce").mean() if likert_cols else pd.NA
    overall_yes = (pd.to_numeric(f[yesno_cols].stack(), errors="coerce").mean() * 100) if yesno_cols else pd.NA

    kpis = pd.DataFrame(
        {
            "Indicador": ["Respuestas", "Promedio global (Likert)", "% Sí (Sí/No)"],
            "Valor": [
                int(len(f)),
                (round(float(overall_likert), 2) if pd.notna(overall_likert) else "—"),
                (f"{round(float(overall_yes), 1)}%" if pd.notna(overall_yes) else "—"),
            ],
        }
    )

    _download_buttons_body(f, filename_prefix=filename_prefix, kpis_df=kpis)

    st.caption(f"Hoja usada: **PROCESADO** | Registros filtrados: **{len(f)}**")

    # Tabs (comparativo SOLO cuando estás en Dirección General y con (Todas))
    if vista == "Dirección General":
        tab1, tab2, tab4, tab3 = st.tabs(["Resumen", "Por sección", "Comparativo entre carreras", "Comentarios"])
    else:
        tab1, tab2, tab3 = st.tabs(["Resumen", "Por sección", "Comentarios"])
        tab4 = None

    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Respuestas", f"{len(f)}")

        if likert_cols:
            c2.metric("Promedio global (Likert)", f"{float(overall_likert):.2f}" if pd.notna(overall_likert) else "—")
        else:
            c2.metric("Promedio global (Likert)", "—")

        if yesno_cols:
            c3.metric("% Sí (Sí/No)", f"{float(overall_yes):.1f}%" if pd.notna(overall_yes) else "—")
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
            rows.append({"Sección": str(sec_name), "Promedio": float(val), "Preguntas": len(cols), "sec_code": sec_code})

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
                tooltip_cols=[
                    alt.Tooltip("Sección:N", title="Sección"),
                    alt.Tooltip("Promedio:Q", title="Promedio", format=".2f"),
                    alt.Tooltip("Preguntas:Q", title="Preguntas"),
                ],
                max_vertical=MAX_VERTICAL_SECTIONS,
                wrap_width_vertical=22,
                wrap_width_horizontal=36,
                base_height=320,
                hide_category_labels=True,
            )
            if sec_chart is not None:
                st.altair_chart(sec_chart, use_container_width=True)

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
            rows.append({"Sección": str(sec_name), "Promedio": float(val), "Preguntas": len(cols), "sec_code": sec_code})

        sec_df2 = pd.DataFrame(rows).sort_values("Promedio", ascending=False) if rows else pd.DataFrame()
        if sec_df2.empty:
            st.info("No hay datos suficientes para mostrar secciones.")
            return

        for _, r in sec_df2.iterrows():
            sec_code = r["sec_code"]
            sec_name = str(r["Sección"])
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
                            tooltip_cols=[
                                alt.Tooltip("Pregunta:N", title="Pregunta"),
                                alt.Tooltip("Promedio:Q", title="Promedio", format=".2f"),
                            ],
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
                            tooltip_cols=[
                                alt.Tooltip("Pregunta:N", title="Pregunta"),
                                alt.Tooltip("% Sí:Q", title="% Sí", format=".1f"),
                            ],
                            max_vertical=MAX_VERTICAL_QUESTIONS,
                            wrap_width_vertical=24,
                            wrap_width_horizontal=40,
                            base_height=340,
                            hide_category_labels=True,
                        )
                        if chart_y is not None:
                            st.altair_chart(chart_y, use_container_width=True)

                # Comentarios por sección: NO mostrar OTR y solo si hay columnas
                items_sec = [(sec, lbl, col) for (sec, lbl, col) in open_items_all if sec == sec_code and col in f.columns and sec != "OTR"]
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
            elif carrera_sel != "(Todas)":
                st.info("Para ver comparativo, en **Carrera/Servicio** selecciona **(Todas)**.")
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
                        rows.append(
                            {
                                "Carrera/Servicio": str(carrera_val).strip(),
                                "Promedio": round(float(mean_val), 2),
                                "Respuestas": int(len(df_c)),
                                "Preguntas": int(len(cols)),
                            }
                        )

                    if not rows:
                        continue

                    sec_comp = pd.DataFrame(rows).sort_values("Promedio", ascending=False).reset_index(drop=True)
                    with st.expander(str(sec_name), expanded=False):
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
                                alt.Tooltip("Respuestas:Q", title="Respuestas"),
                                alt.Tooltip("Preguntas:Q", title="Preguntas"),
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

        # NO mostrar "OTR" y siempre mostrar nombre real de sección (como DG)
        sec_codes = sorted({sec for (sec, _, _) in open_items_all if sec != "OTR"})
        sec_map_name = {code: SECTION_LABELS.get(code, code) for code in sec_codes}

        opts = ["(Todas)"] + [sec_map_name.get(code, code) for code in sec_codes]
        sec_sel = st.selectbox("Sección", opts, index=0, key="open_global_sec_sel")

        if sec_sel == "(Todas)":
            pool = [(sec, lbl, col) for (sec, lbl, col) in open_items_all if col in f.columns and sec != "OTR"]
            sec_key = "ALL"
        else:
            # resolver code por label
            inv = {v: k for k, v in sec_map_name.items()}
            sec_code = inv.get(sec_sel, sec_sel)
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
