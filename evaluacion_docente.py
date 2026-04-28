# evaluacion_docente.py
import pandas as pd
import streamlit as st
import gspread
import textwrap
import re
import altair as alt
import unicodedata

SHEET_BASE = "BASE"
COL_CARRERA_OFICIAL = "carrera_oficial"

# ── Umbrales institucionales ────────────────────────────────────────────────
UMBRAL_CRITICO = 70.0       # < 70
UMBRAL_SEGUIMIENTO = 79.0   # 70 - 78.9
UMBRAL_BUENO = 85.0         # 79 - 84.9
UMBRAL_DESTACADO = 90.0     # 90 - 100


def _clasificar(prom):
    try:
        p = float(prom)
    except Exception:
        return "Sin dato", "⚪"

    if p < UMBRAL_CRITICO:
        return "Crítico", "🔴"
    if p < UMBRAL_SEGUIMIENTO:
        return "Seguimiento", "🟡"
    if p < UMBRAL_BUENO:
        return "Aceptable", "🟢"
    if p < UMBRAL_DESTACADO:
        return "Bueno", "🔵"
    return "Destacado", "⭐"


def _prioridad_operativa(n_criticos: int, n_seguimiento: int, promedio: float) -> str:
    try:
        p = float(promedio)
    except Exception:
        p = None

    if n_criticos > 0 or (p is not None and p < UMBRAL_SEGUIMIENTO):
        return "🔴 Alta"
    if n_seguimiento > 0:
        return "🟡 Media"
    return "🟢 Baja"


def _to_float(x):
    try:
        return float(str(x).replace("%", "").strip())
    except Exception:
        return pd.NA


def _to_int(x):
    try:
        return int(float(str(x).strip()))
    except Exception:
        return pd.NA


def _wrap_text(s: str, width: int = 40, max_lines: int = 2) -> str:
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


def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def _norm_text(s: str) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s).replace("\u00A0", " ").replace("\u200B", "").strip().lower()
    s = _strip_accents(s)
    s = re.sub(r"\s+", " ", s)
    return s


def _norm_key(s: str) -> str:
    s = _norm_text(s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _safe_percent(n, d):
    try:
        if pd.isna(n) or pd.isna(d) or float(d) == 0:
            return pd.NA
        return (float(n) / float(d)) * 100.0
    except Exception:
        return pd.NA


def _cycle_sort_key(c: str):
    s = str(c).strip()
    m = re.match(r"^(\d{2,4})\s*-\s*(\d{1,2})$", s)
    if not m:
        return (9999, 99, s)
    y = int(m.group(1))
    if y < 100:
        y = 2000 + y
    p = int(m.group(2))
    return (y, p, s)


@st.cache_data(show_spinner=False, ttl=300)
def _load_sheet_as_df(url: str, sheet_name: str) -> pd.DataFrame:
    sa = dict(st.secrets["gcp_service_account_json"])
    gc = gspread.service_account_from_dict(sa)
    sh = gc.open_by_url(url)

    def norm(x: str) -> str:
        return str(x).strip().lower().replace(" ", "").replace("_", "")

    titles = [ws.title for ws in sh.worksheets()]
    titles_norm = {norm(t): t for t in titles}
    resolved = titles_norm.get(norm(sheet_name))
    if not resolved:
        raise ValueError(
            f"No encontré la pestaña '{sheet_name}'. "
            f"Pestañas disponibles: {', '.join(titles)}"
        )

    ws = sh.worksheet(resolved)
    values = ws.get_all_values()
    if not values:
        return pd.DataFrame()

    headers = [h.strip() for h in values[0]]
    rows = values[1:]
    return pd.DataFrame(rows, columns=headers).replace("", pd.NA)


def _promedio_ponderado(dfx: pd.DataFrame) -> float:
    w = pd.to_numeric(dfx["total"], errors="coerce")
    y = pd.to_numeric(dfx["promedio"], errors="coerce")
    denom = w.sum(skipna=True)
    if pd.notna(denom) and float(denom) > 0:
        val = (y * w).sum(skipna=True) / denom
        return float(val) if pd.notna(val) else float("nan")
    val = y.mean()
    return float(val) if pd.notna(val) else float("nan")


def _make_line_chart(df_line: pd.DataFrame, x: str, y: str, title: str):
    if df_line.empty:
        st.info("No hay datos suficientes para graficar.")
        return

    yvals = pd.to_numeric(df_line[y], errors="coerce").dropna()
    y_min, y_max = (yvals.min(), yvals.max()) if not yvals.empty else (None, None)
    scale = None
    if y_min is not None and y_max is not None and pd.notna(y_min) and pd.notna(y_max):
        pad = max(0.5, (float(y_max) - float(y_min)) * 0.15) if float(y_max) != float(y_min) else 1.0
        scale = alt.Scale(domain=[float(y_min) - pad, float(y_max) + pad])

    rules_data = pd.DataFrame([
        {"y": UMBRAL_CRITICO, "label": "Crítico (<70)"},
        {"y": UMBRAL_SEGUIMIENTO, "label": "Seguimiento (70-78.9)"},
        {"y": UMBRAL_BUENO, "label": "Bueno (85-89.9)"},
        {"y": UMBRAL_DESTACADO, "label": "Destacado (>=90)"},
    ])

    rules = (
        alt.Chart(rules_data)
        .mark_rule(strokeDash=[4, 3], opacity=0.45)
        .encode(
            y=alt.Y("y:Q"),
            tooltip=[alt.Tooltip("label:N", title="Umbral")],
        )
    )

    line = (
        alt.Chart(df_line)
        .mark_line(point=alt.OverlayMarkDef(size=80, filled=True))
        .encode(
            x=alt.X(f"{x}:O", sort=df_line[x].tolist(), title="Ciclo"),
            y=alt.Y(f"{y}:Q", title="Promedio", scale=scale),
            tooltip=[
                alt.Tooltip(x, title="Ciclo"),
                alt.Tooltip(y, title="Promedio", format=".2f"),
            ],
        )
        .properties(height=300, title=title)
    )

    st.altair_chart(rules + line, use_container_width=True)


def _kpi_card(label: str, value: str, sub: str = "", color: str = "#1565C0"):
    st.markdown(
        f"""
        <div style="background:#f8f9fc;border-left:4px solid {color};
                    border-radius:8px;padding:14px 18px;margin-bottom:4px;">
          <p style="margin:0;font-size:0.75rem;color:#666;font-weight:600;
                    text-transform:uppercase;letter-spacing:.05em">{label}</p>
          <p style="margin:4px 0 0;font-size:1.6rem;font-weight:700;
                    color:{color};line-height:1">{value}</p>
          <p style="margin:2px 0 0;font-size:0.78rem;color:#888">{sub}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _texto_ejecutivo(ciclo: str, prom: float, parte: float, n_criticos: int,
                     n_seguimiento: int, n_destacados: int, total_grupos: int) -> str:
    prom_str = f"{prom:.1f}" if pd.notna(prom) else "N/D"
    part_str = f"{parte:.1f}%" if pd.notna(parte) else "N/D"
    pct_crit = _safe_percent(n_criticos, total_grupos)
    pct_seg = _safe_percent(n_seguimiento, total_grupos)

    lineas = [
        f"**Ciclo {ciclo}** · Promedio **{prom_str}** · Participación **{part_str}** · "
        f"**{total_grupos}** grupos evaluados."
    ]

    alertas = []
    if n_criticos > 0:
        alertas.append(f"🔴 **{n_criticos}** críticos ({pct_crit:.1f}%)")
    if n_seguimiento > 0:
        alertas.append(f"🟡 **{n_seguimiento}** en seguimiento ({pct_seg:.1f}%)")
    if alertas:
        lineas.append("Áreas de atención: " + " · ".join(alertas) + ".")
    else:
        lineas.append("No se detectan grupos críticos ni en seguimiento con los filtros actuales.")

    if n_destacados > 0:
        lineas.append(f"⭐ **{n_destacados}** grupo(s) con desempeño destacado (≥ 90).")

    return "  \n".join(lineas)


def render_evaluacion_docente(vista: str | None = None, carrera: str | None = None, ed_url: str | None = None):
    st.subheader("📋 Evaluación Docente")

    # Este módulo no debe usarse para Dirección Finanzas.
    if vista == "Dirección Finanzas":
        st.warning("Este módulo no está disponible para Dirección Finanzas.")
        return

    if not vista:
        vista = "Dirección General"

    if not ed_url:
        ed_url = st.secrets.get("EDOCENTE_URL", "").strip()
    if not ed_url:
        st.error("Falta configurar la URL de Evaluación Docente (EDOCENTE_URL en Secrets).")
        return

    try:
        with st.spinner("Cargando datos desde Google Sheets…"):
            df = _load_sheet_as_df(ed_url, SHEET_BASE)
    except Exception as e:
        st.error("No se pudo cargar la pestaña BASE de Evaluación Docente.")
        st.exception(e)
        return

    if df.empty:
        st.warning("La hoja BASE está vacía.")
        return

    required = {"profesor", "grupo", "materia", "aplicaron", "total", "promedio", "ciclo", COL_CARRERA_OFICIAL}
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error("Faltan columnas en BASE: " + ", ".join(missing))
        return

    df = df.copy()
    df["aplicaron"] = df["aplicaron"].apply(_to_int)
    df["total"] = df["total"].apply(_to_int)
    df["promedio"] = df["promedio"].apply(_to_float)
    df["participacion_pct"] = [_safe_percent(n, d) for n, d in zip(df["aplicaron"], df["total"])]
    df["_prof_key"] = df["profesor"].apply(_norm_key)
    df["_car_key"] = df[COL_CARRERA_OFICIAL].apply(_norm_key)

    ciclos = sorted(df["ciclo"].dropna().astype(str).str.strip().unique().tolist(), key=_cycle_sort_key)
    if not ciclos:
        st.warning("No encontré valores de ciclo.")
        return

    with st.sidebar:
        st.markdown("## 🎛️ Filtros Evaluación Docente")
        st.divider()

        ciclo_sel = st.selectbox("📅 Ciclo", ciclos, index=max(0, len(ciclos) - 1), key="ed_ciclo")
        base_ciclo = df[df["ciclo"].astype(str).str.strip() == str(ciclo_sel).strip()].copy()

        if vista == "Dirección General":
            carreras = sorted(base_ciclo[COL_CARRERA_OFICIAL].dropna().astype(str).str.strip().unique().tolist())
            carrera_opts = ["(Todas)"] + carreras
            carrera_sel = st.selectbox("Carrera / Servicio", carrera_opts, index=0, key="ed_carrera")
            carrera_key_sel = "" if carrera_sel == "(Todas)" else _norm_key(carrera_sel)
        else:
            carrera_sel = (carrera or "").strip()
            carrera_key_sel = _norm_key(carrera_sel)
            st.text_input("Carrera / Servicio", value=carrera_sel, disabled=True, key="ed_carrera_fija")

        st.divider()
        st.markdown("#### 🚦 Criterios")
        st.caption(
            "Crítico **< 70**  \n"
            "Seguimiento **70–78.9**  \n"
            "Aceptable **79–84.9**  \n"
            "Bueno **85–89.9**  \n"
            "Destacado **≥ 90**"
        )

        mostrar_criticos = st.checkbox("🔴 Mostrar Críticos", value=True, key="ed_show_crit")
        mostrar_seguimiento = st.checkbox("🟡 Mostrar Seguimiento", value=True, key="ed_show_seg")

        st.divider()
        st.markdown("#### 🏆 Top Docentes")
        min_grupos = st.number_input("Mínimo de grupos", min_value=1, max_value=20, value=1, step=1, key="ed_min_grupos")
        min_part = st.number_input("Participación mínima %", min_value=0, max_value=100, value=0, step=5, key="ed_min_part")

        st.divider()
        orden_riesgo = st.radio(
            "Orden de riesgo",
            ["Menor a mayor promedio", "Mayor a menor promedio"],
            index=0,
            key="ed_orden_riesgo",
        )

    f = base_ciclo.copy()
    if vista == "Dirección General":
        if carrera_key_sel:
            f = f[f["_car_key"] == carrera_key_sel]
    else:
        if not carrera_key_sel:
            st.error("No hay carrera/servicio asignado para esta vista.")
            return
        f = f[f["_car_key"] == carrera_key_sel]

    if vista != "Dirección General" and len(f) == 0:
        st.error("No hay registros para esta carrera con el filtro actual.")
        st.caption(f"Clave buscada: '{carrera_key_sel}'")
        uniques = sorted(base_ciclo[COL_CARRERA_OFICIAL].dropna().astype(str).str.strip().unique())
        st.dataframe(pd.DataFrame({COL_CARRERA_OFICIAL: list(uniques)[:25]}), use_container_width=True)
        return

    if len(f) == 0:
        st.warning("No hay registros con los filtros seleccionados.")
        return

    prom_num = pd.to_numeric(f["promedio"], errors="coerce")
    prom_global = prom_num.mean()
    part_global = _safe_percent(
        pd.to_numeric(f["aplicaron"], errors="coerce").sum(),
        pd.to_numeric(f["total"], errors="coerce").sum(),
    )
    n_criticos = int((prom_num < UMBRAL_CRITICO).sum())
    n_seguimiento = int(((prom_num >= UMBRAL_CRITICO) & (prom_num < UMBRAL_SEGUIMIENTO)).sum())
    n_aceptable = int(((prom_num >= UMBRAL_SEGUIMIENTO) & (prom_num < UMBRAL_BUENO)).sum())
    n_bueno = int(((prom_num >= UMBRAL_BUENO) & (prom_num < UMBRAL_DESTACADO)).sum())
    n_destacado = int((prom_num >= UMBRAL_DESTACADO).sum())
    pct_criticos = _safe_percent(n_criticos, len(f))
    pct_seguimiento = _safe_percent(n_seguimiento, len(f))

    st.markdown(_texto_ejecutivo(ciclo_sel, prom_global, part_global, n_criticos, n_seguimiento, n_destacado, len(f)))
    st.divider()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        _kpi_card(
            "Promedio ciclo",
            f"{prom_global:.1f}" if pd.notna(prom_global) else "—",
            _clasificar(prom_global)[1] + " " + _clasificar(prom_global)[0],
            "#1565C0",
        )
    with c2:
        _kpi_card(
            "Participación",
            f"{part_global:.1f}%" if pd.notna(part_global) else "—",
            f"{int(pd.to_numeric(f['aplicaron'], errors='coerce').sum())} / {int(pd.to_numeric(f['total'], errors='coerce').sum())} estudiantes",
            "#00695C",
        )
    with c3:
        _kpi_card("Grupos evaluados", str(len(f)), f"{n_bueno} buenos · {n_destacado} destacados", "#37474F")
    with c4:
        _kpi_card("🔴 Críticos", f"{n_criticos}", f"{pct_criticos:.1f}% del total" if pd.notna(pct_criticos) else "", "#B71C1C")
    with c5:
        _kpi_card("🟡 Seguimiento", f"{n_seguimiento}", f"{pct_seguimiento:.1f}% del total" if pd.notna(pct_seguimiento) else "", "#E65100")

    st.divider()

    tab_resumen, tab_tendencia, tab_criticos, tab_seguimiento, tab_top = st.tabs([
        "📊 Resumen",
        "📈 Tendencia",
        "🔴 Casos Críticos",
        "🟡 Casos en Seguimiento",
        "⭐ Top Docentes",
    ])

    with tab_resumen:
        if vista == "Dirección General" and carrera_key_sel == "":
            st.markdown("#### Promedio por carrera/servicio")
            rows = []
            for car, dfx in f.groupby(COL_CARRERA_OFICIAL):
                prom_x = pd.to_numeric(dfx["promedio"], errors="coerce").mean()
                part_x = _safe_percent(
                    pd.to_numeric(dfx["aplicaron"], errors="coerce").sum(),
                    pd.to_numeric(dfx["total"], errors="coerce").sum(),
                )
                pn = pd.to_numeric(dfx["promedio"], errors="coerce")
                crit_x = int((pn < UMBRAL_CRITICO).sum())
                seg_x = int(((pn >= UMBRAL_CRITICO) & (pn < UMBRAL_SEGUIMIENTO)).sum())
                rows.append({
                    "Carrera/Servicio": str(car).strip(),
                    "Clasificación": _clasificar(prom_x)[1] + " " + _clasificar(prom_x)[0],
                    "Prioridad": _prioridad_operativa(crit_x, seg_x, prom_x),
                    "Promedio": round(float(prom_x), 2) if pd.notna(prom_x) else pd.NA,
                    "Participación %": round(float(part_x), 1) if pd.notna(part_x) else pd.NA,
                    "Grupos": int(len(dfx)),
                    "🔴 Críticos": crit_x,
                    "🟡 Seguimiento": seg_x,
                    "🟢 Aceptables": int(((pn >= UMBRAL_SEGUIMIENTO) & (pn < UMBRAL_BUENO)).sum()),
                    "🔵 Buenos": int(((pn >= UMBRAL_BUENO) & (pn < UMBRAL_DESTACADO)).sum()),
                    "⭐ Destacados": int((pn >= UMBRAL_DESTACADO).sum()),
                })
            out = pd.DataFrame(rows).sort_values(["Prioridad", "Promedio"], ascending=[True, False], na_position="last")
            st.dataframe(out.reset_index(drop=True), use_container_width=True)
        else:
            st.markdown("#### Promedio por docente")
            rows = []
            for prof, dfx in f.groupby("profesor"):
                prom_w = _promedio_ponderado(dfx)
                part_x = _safe_percent(
                    pd.to_numeric(dfx["aplicaron"], errors="coerce").sum(),
                    pd.to_numeric(dfx["total"], errors="coerce").sum(),
                )
                dfx2 = dfx.copy()
                dfx2["_total_num"] = pd.to_numeric(dfx2["total"], errors="coerce")
                by_car = (
                    dfx2.groupby(COL_CARRERA_OFICIAL, dropna=False)
                    .agg(peso_total=("_total_num", "sum"), grupos=("grupo", "count"))
                    .reset_index()
                    .sort_values(["peso_total", "grupos"], ascending=[False, False], na_position="last")
                )
                carrera_principal = by_car.iloc[0][COL_CARRERA_OFICIAL] if not by_car.empty else pd.NA
                pn = pd.to_numeric(dfx["promedio"], errors="coerce")
                crit_x = int((pn < UMBRAL_CRITICO).sum())
                seg_x = int(((pn >= UMBRAL_CRITICO) & (pn < UMBRAL_SEGUIMIENTO)).sum())
                rows.append({
                    "Carrera/Servicio": _wrap_text(str(carrera_principal).strip() if pd.notna(carrera_principal) else "—", 32, 2),
                    "Profesor": _wrap_text(prof, 45, 2),
                    "Clasificación": _clasificar(prom_w)[1] + " " + _clasificar(prom_w)[0],
                    "Prioridad": _prioridad_operativa(crit_x, seg_x, prom_w),
                    "Promedio (ponderado)": round(prom_w, 2) if pd.notna(prom_w) else pd.NA,
                    "Participación %": round(float(part_x), 1) if pd.notna(part_x) else pd.NA,
                    "Grupos": int(len(dfx)),
                    "🔴 Críticos": crit_x,
                    "🟡 Seguimiento": seg_x,
                })
            out = pd.DataFrame(rows).sort_values(["Prioridad", "Promedio (ponderado)"], ascending=[True, False], na_position="last")
            st.dataframe(out.reset_index(drop=True), use_container_width=True)

    with tab_tendencia:
        trend_base = df.copy()
        if vista == "Dirección General":
            if carrera_key_sel:
                trend_base = trend_base[trend_base["_car_key"] == carrera_key_sel]
                title = f"Tendencia — {carrera_sel}"
            else:
                title = "Tendencia institucional de promedio"
        else:
            trend_base = trend_base[trend_base["_car_key"] == carrera_key_sel]
            title = f"Tendencia — {carrera_sel}"

        rows = []
        for cyc, dfx in trend_base.groupby("ciclo"):
            prom_x = pd.to_numeric(dfx["promedio"], errors="coerce").mean()
            rows.append({"ciclo": str(cyc).strip(), "promedio": float(prom_x) if pd.notna(prom_x) else pd.NA})

        df_line = pd.DataFrame(rows)
        if df_line.empty:
            st.info("No hay datos suficientes para la tendencia.")
        else:
            df_line["sort_key"] = df_line["ciclo"].apply(_cycle_sort_key)
            df_line = df_line.sort_values("sort_key").drop(columns=["sort_key"])
            _make_line_chart(df_line, x="ciclo", y="promedio", title=title)
            st.dataframe(
                df_line.assign(Clasificación=df_line["promedio"].apply(lambda x: _clasificar(x)[1] + " " + _clasificar(x)[0])).reset_index(drop=True),
                use_container_width=True,
            )

    with tab_criticos:
        st.markdown("#### 🔴 Grupos con promedio menor a 70")
        criticos = f[pd.to_numeric(f["promedio"], errors="coerce") < UMBRAL_CRITICO].copy()

        if not mostrar_criticos:
            st.info("Filtro desactivado desde el panel lateral.")
        elif criticos.empty:
            st.success("✅ No hay grupos en estado crítico para el ciclo y filtros seleccionados.")
        else:
            criticos["Participación %"] = criticos["participacion_pct"]
            criticos["Clasificación"] = "🔴 Crítico"
            show_cols = [COL_CARRERA_OFICIAL, "profesor", "materia", "grupo", "promedio", "Participación %", "ciclo", "Clasificación"]
            out = criticos[show_cols].copy().rename(columns={COL_CARRERA_OFICIAL: "Carrera/Servicio"})
            out["_prom"] = pd.to_numeric(out["promedio"], errors="coerce")
            asc = orden_riesgo == "Menor a mayor promedio"
            out = out.sort_values("_prom", ascending=asc, na_position="last").drop(columns=["_prom"])
            out["Carrera/Servicio"] = out["Carrera/Servicio"].apply(lambda x: _wrap_text(x, 32, 2))
            out["profesor"] = out["profesor"].apply(lambda x: _wrap_text(x, 35, 2))
            out["materia"] = out["materia"].apply(lambda x: _wrap_text(x, 45, 2))
            st.dataframe(out.reset_index(drop=True), use_container_width=True)

    with tab_seguimiento:
        st.markdown("#### 🟡 Grupos con promedio entre 70 y 78.9")
        pn_f = pd.to_numeric(f["promedio"], errors="coerce")
        seguimiento = f[(pn_f >= UMBRAL_CRITICO) & (pn_f < UMBRAL_SEGUIMIENTO)].copy()

        if not mostrar_seguimiento:
            st.info("Filtro desactivado desde el panel lateral.")
        elif seguimiento.empty:
            st.success("✅ No hay grupos en seguimiento para el ciclo y filtros seleccionados.")
        else:
            seguimiento["Participación %"] = seguimiento["participacion_pct"]
            seguimiento["Clasificación"] = "🟡 Seguimiento"
            show_cols = [COL_CARRERA_OFICIAL, "profesor", "materia", "grupo", "promedio", "Participación %", "ciclo", "Clasificación"]
            out = seguimiento[show_cols].copy().rename(columns={COL_CARRERA_OFICIAL: "Carrera/Servicio"})
            out["_prom"] = pd.to_numeric(out["promedio"], errors="coerce")
            asc = orden_riesgo == "Menor a mayor promedio"
            out = out.sort_values("_prom", ascending=asc, na_position="last").drop(columns=["_prom"])
            out["Carrera/Servicio"] = out["Carrera/Servicio"].apply(lambda x: _wrap_text(x, 32, 2))
            out["profesor"] = out["profesor"].apply(lambda x: _wrap_text(x, 35, 2))
            out["materia"] = out["materia"].apply(lambda x: _wrap_text(x, 45, 2))
            st.dataframe(out.reset_index(drop=True), use_container_width=True)

    with tab_top:
        st.markdown("#### ⭐ Docentes con promedio ponderado ≥ 90")
        rows = []
        for prof, dfx in f.groupby("profesor"):
            prom_w = _promedio_ponderado(dfx)
            grupos_n = len(dfx)
            part_p = _safe_percent(
                pd.to_numeric(dfx["aplicaron"], errors="coerce").sum(),
                pd.to_numeric(dfx["total"], errors="coerce").sum(),
            )

            if grupos_n < int(min_grupos):
                continue
            if pd.notna(part_p) and float(part_p) < float(min_part):
                continue
            if not (pd.notna(prom_w) and float(prom_w) >= UMBRAL_DESTACADO):
                continue

            dfx2 = dfx.copy()
            dfx2["_total_num"] = pd.to_numeric(dfx2["total"], errors="coerce")
            by_car = (
                dfx2.groupby(COL_CARRERA_OFICIAL, dropna=False)
                .agg(peso_total=("_total_num", "sum"), grupos=("grupo", "count"))
                .reset_index()
                .sort_values(["peso_total", "grupos"], ascending=[False, False], na_position="last")
            )
            carrera_principal = by_car.iloc[0][COL_CARRERA_OFICIAL] if not by_car.empty else pd.NA

            rows.append({
                "Carrera/Servicio": _wrap_text(str(carrera_principal).strip() if pd.notna(carrera_principal) else "—", 32, 2),
                "Profesor": _wrap_text(prof, 50, 2),
                "Clasificación": "⭐ Destacado",
                "Promedio (ponderado)": round(prom_w, 2),
                "Participación %": round(float(part_p), 1) if pd.notna(part_p) else pd.NA,
                "Grupos": int(grupos_n),
            })

        out = pd.DataFrame(rows)
        if out.empty:
            st.info("No hay docentes destacados (≥ 90) con los criterios seleccionados.")
        else:
            out = out.sort_values("Promedio (ponderado)", ascending=False, na_position="last")
            st.dataframe(out.reset_index(drop=True), use_container_width=True)
