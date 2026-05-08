import streamlit as st
import pandas as pd
import plotly.express as px
import unicodedata
import re

# =====================================================
# CONFIG
# =====================================================
SHEET_ID = "1Dgu3_UMAYecX-KCxLhHUe_EYiXCF68rvE2XpYwOx9lM"

# Hoja principal con estatus ya calculado (FINALIZADO / EN_PROCESO)
URL_SEGUIMIENTO = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet=SEGUIMIENTO_ACTUAL"
# Hoja con área de adscripción por docente/curso
URL_INSCRIPCIONES = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet=INSCRIPCIONES_SYNC"
# Hoja con correcciones manuales que sobreescriben SEGUIMIENTO_ACTUAL
URL_AJUSTES = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet=AJUSTES_MANUALES"

COLOR_AZUL = "#2F80ED"
COLOR_VERDE = "#10B981"
COLOR_ROJO = "#D7263D"
COLOR_NARANJA = "#F97316"
COLOR_GRIS = "#374151"

COLOR_MAP_SIMPLE = {
    "Finalizados": COLOR_VERDE,
    "En proceso": COLOR_AZUL,
}


# =====================================================
# UTILIDADES
# =====================================================
def normalizar_texto(texto):
    if not isinstance(texto, str):
        return ""
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    texto = texto.lower().strip()
    texto = re.sub(r"\s+", "_", texto)
    texto = re.sub(r"[^\w]", "", texto)
    return texto


def limpiar_id(texto):
    if pd.isna(texto):
        return ""
    texto = str(texto).strip()
    texto = texto.replace(".0", "") if texto.endswith(".0") else texto
    return texto.lower()


def limpiar_correo(texto):
    if pd.isna(texto):
        return ""
    return str(texto).strip().lower()


def normalizar_valor(texto):
    if pd.isna(texto):
        return ""
    return str(texto).strip()


def normalizar_columnas(df):
    columnas_originales = list(df.columns)
    columnas_norm = [normalizar_texto(c) for c in columnas_originales]
    mapa_original = dict(zip(columnas_norm, columnas_originales))
    df.columns = columnas_norm
    df.attrs["mapa_original"] = mapa_original
    return df


@st.cache_data(ttl=300)
def cargar_datos():
    # Hoja SEGUIMIENTO_ACTUAL — fuente oficial de estatus
    seguimiento = pd.read_csv(URL_SEGUIMIENTO)
    seguimiento = normalizar_columnas(seguimiento)

    # Hoja INSCRIPCIONES_SYNC — fuente del área de adscripción
    try:
        inscripciones = pd.read_csv(URL_INSCRIPCIONES)
        inscripciones = normalizar_columnas(inscripciones)
    except Exception:
        inscripciones = pd.DataFrame()

    # Hoja AJUSTES_MANUALES — correcciones manuales con prioridad sobre SEGUIMIENTO_ACTUAL
    try:
        ajustes = pd.read_csv(URL_AJUSTES)
        ajustes = normalizar_columnas(ajustes)
    except Exception:
        ajustes = pd.DataFrame()


    return seguimiento, inscripciones, ajustes


# =====================================================
# ALIAS
# =====================================================
ALIAS_AREA = [
    "area_adscripcion",
    "area_de_adscripcion",
    "adscripcion",
    "area",
    "carrera",
    "departamento",
]

ALIAS_NOMBRE = [
    "nombre_seac",
    "nombre_forms",
    "nombre_normalizado",
    "nombre",
    "docente",
    "nombre_docente",
    "nombre_completo",
]

ALIAS_CORREO = [
    "correo_docente",
    "correo",
    "correo_electronico",
    "direccion_de_correo_electronico",
    "correo_seac",
]

ALIAS_ESTATUS = [
    "estatus_final",
    "estatus",
    "estado",
    "estatus_seac",
    "estado_final",
    "situacion",
]

ALIAS_AVANCE = [
    "avance_pct",
    "avance",
    "porcentaje_avance",
    "porcentaje",
]


def detectar_columna(df, aliases):
    for alias in aliases:
        cand = normalizar_texto(alias)
        for col in df.columns:
            if normalizar_texto(col) == cand:
                return col

    for alias in aliases:
        cand = normalizar_texto(alias)
        for col in df.columns:
            if cand in normalizar_texto(col):
                return col

    return None


# =====================================================
# CURSO
# =====================================================
def extraer_curso_desde_columna(nombre_columna_original):
    texto = str(nombre_columna_original)

    m = re.search(r"\[(.*?)\]", texto)
    if m:
        return m.group(1).strip()

    texto = texto.replace("Selecciona el taller al que deseas inscribirte", "")
    texto = texto.replace(
        "en caso de no no desees inscribirte a algún curso selecciona NO INSCRIBIRME",
        "",
    )
    texto = texto.replace(
        "en caso de no desees inscribirte a algún curso selecciona NO INSCRIBIRME",
        "",
    )
    texto = texto.replace(",", "")
    texto = texto.strip()

    return texto if texto else "SIN CURSO"


def construir_curso_si_no_existe(df):
    df = df.copy()

    if "curso" in df.columns and df["curso"].notna().any():
        df["curso"] = df["curso"].astype(str).str.strip()
        return df

    mapa_original = df.attrs.get("mapa_original", {})

    posibles_cols_curso = []
    for col in df.columns:
        col_norm = normalizar_texto(col)
        col_original = mapa_original.get(col, col)

        if (
            "taller" in col_norm
            or "curso" in col_norm
            or "capacitacion" in col_norm
            or "inscribirte" in normalizar_texto(str(col_original))
        ):
            posibles_cols_curso.append(col)

    posibles_cols_curso = [
        c for c in posibles_cols_curso
        if c not in ["curso", "nombre_curso", "estatus_final", "tipo_correo"]
    ]

    if not posibles_cols_curso:
        df["curso"] = "SIN CURSO"
        return df

    registros = []
    columnas_base = [c for c in df.columns if c not in posibles_cols_curso]

    for _, row in df.iterrows():
        for col in posibles_cols_curso:
            valor = normalizar_valor(row.get(col, ""))
            valor_norm = normalizar_texto(valor)

            if not valor_norm:
                continue
            if "no_inscribirme" in valor_norm or "no_me_inscribo" in valor_norm:
                continue
            if valor_norm in ["nan", "sin_dato", "no"]:
                continue

            nuevo = row[columnas_base].to_dict()
            col_original = mapa_original.get(col, col)
            curso_desde_columna = extraer_curso_desde_columna(col_original)

            if "inscrib" in valor_norm or valor_norm in ["si", "sí"]:
                nuevo["curso"] = curso_desde_columna
            else:
                nuevo["curso"] = valor

            registros.append(nuevo)

    if registros:
        return pd.DataFrame(registros)

    df["curso"] = "SIN CURSO"
    return df


# =====================================================
# LIMPIEZA SEGUIMIENTO
# =====================================================
def limpiar_seguimiento(df):
    df = df.copy()

    col_nombre = detectar_columna(df, ALIAS_NOMBRE)
    col_correo = detectar_columna(df, ALIAS_CORREO)
    col_avance = detectar_columna(df, ALIAS_AVANCE)

    if col_nombre and col_nombre != "nombre_normalizado":
        df = df.rename(columns={col_nombre: "nombre_normalizado"})

    if col_correo and col_correo != "correo_docente":
        df = df.rename(columns={col_correo: "correo_docente"})

    if col_avance and col_avance != "avance_pct":
        df = df.rename(columns={col_avance: "avance_pct"})

    # estatus_final ya viene con los valores FINALIZADO / EN_PROCESO desde SEGUIMIENTO_ACTUAL
    # no se renombra ni se sobreescribe

    df = construir_curso_si_no_existe(df)

    defaults = {
        "curso": "SIN CURSO",
        "nombre_normalizado": "SIN NOMBRE",
        "correo_docente": "",
        "avance_pct": 0,
        "tareas_entregadas": 0,
        "tareas_totales": 0,
        "matricula": "",
        "requiere_correo": "",
        "tipo_correo": "",
        "fecha_ultimo_corte": "",
        "observaciones": "",
        "estatus_final": "EN_PROCESO",
        "area_de_adscripcion": "SIN ÁREA",
    }

    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    df["tareas_entregadas"] = pd.to_numeric(df["tareas_entregadas"], errors="coerce").fillna(0)
    df["tareas_totales"] = pd.to_numeric(df["tareas_totales"], errors="coerce").fillna(0)

    df["avance_pct"] = pd.to_numeric(df["avance_pct"], errors="coerce").fillna(0)

    if df["avance_pct"].max() <= 1 and df["avance_pct"].max() > 0:
        df["avance_pct"] = df["avance_pct"] * 100

    mask_avance_cero = (df["avance_pct"] == 0) & (df["tareas_totales"] > 0)
    df.loc[mask_avance_cero, "avance_pct"] = (
        df.loc[mask_avance_cero, "tareas_entregadas"]
        / df.loc[mask_avance_cero, "tareas_totales"]
        * 100
    )

    df["avance_pct"] = df["avance_pct"].clip(lower=0, upper=100)

    df["curso"] = (
        df["curso"]
        .astype(str)
        .str.strip()
        .replace("nan", "SIN CURSO")
        .replace("", "SIN CURSO")
    )

    df["nombre_normalizado"] = (
        df["nombre_normalizado"]
        .astype(str)
        .str.strip()
        .replace("nan", "SIN NOMBRE")
        .replace("", "SIN NOMBRE")
    )

    df["correo_docente"] = (
        df["correo_docente"]
        .astype(str)
        .str.strip()
        .replace("nan", "")
    )

    df["matricula"] = df["matricula"].apply(limpiar_id)

    # Leer estatus_final directamente de la hoja y derivar finalizado_oficial
    df["estatus_final"] = (
        df["estatus_final"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace("NAN", "EN_PROCESO")
        .replace("", "EN_PROCESO")
    )
    df["finalizado_oficial"] = df["estatus_final"] == "FINALIZADO"

    df["curso_key"] = df["curso"].apply(normalizar_texto)
    df["matricula_key"] = df["matricula"].apply(limpiar_id)
    df["correo_key"] = df["correo_docente"].apply(limpiar_correo)
    df["nombre_key"] = df["nombre_normalizado"].apply(normalizar_texto)

    return df


# =====================================================
# APLICAR AJUSTES MANUALES
# =====================================================
def aplicar_ajustes_manuales(df, ajustes):
    """
    Sobreescribe campos en SEGUIMIENTO_ACTUAL con los valores de AJUSTES_MANUALES.
    Llave de cruce: correo_docente + curso (prioridad), luego matricula + curso.
    Campos que sobreescribe: tareas_entregadas, tareas_totales, avance_pct,
    estatus_final, observaciones.
    """
    if ajustes.empty:
        return df

    aj = ajustes.copy()

    # Preparar claves de ajuste
    aj["_correo_key"] = aj["correo_docente"].apply(limpiar_correo) if "correo_docente" in aj.columns else ""
    aj["_matricula_key"] = aj["matricula"].apply(limpiar_id) if "matricula" in aj.columns else ""
    aj["_curso_key"] = aj["curso"].apply(normalizar_texto) if "curso" in aj.columns else ""

    # Convertir campos numéricos
    for campo in ["tareas_entregadas_manual", "tareas_totales_manual", "avance_pct_manual"]:
        if campo in aj.columns:
            aj[campo] = pd.to_numeric(aj[campo], errors="coerce")

    if "estatus_final_manual" in aj.columns:
        aj["estatus_final_manual"] = aj["estatus_final_manual"].astype(str).str.strip().str.upper()

    if "observaciones_manual" in aj.columns:
        aj["observaciones_manual"] = aj["observaciones_manual"].astype(str).str.strip().replace("nan", "")

    # Construir mapas de ajuste: clave -> dict de campos a sobreescribir
    mapa_correo_curso = {}
    mapa_matricula_curso = {}

    for _, row in aj.iterrows():
        campos = {}
        if "tareas_entregadas_manual" in row and pd.notna(row["tareas_entregadas_manual"]):
            campos["tareas_entregadas"] = float(row["tareas_entregadas_manual"])
        if "tareas_totales_manual" in row and pd.notna(row["tareas_totales_manual"]):
            campos["tareas_totales"] = float(row["tareas_totales_manual"])
        if "avance_pct_manual" in row and pd.notna(row["avance_pct_manual"]):
            campos["avance_pct"] = float(row["avance_pct_manual"])
        estatus_m = str(row.get("estatus_final_manual", "")).strip()
        if estatus_m and estatus_m not in ("", "NAN", "NAN"):
            campos["estatus_final"] = estatus_m
        obs_m = str(row.get("observaciones_manual", "")).strip()
        if obs_m and obs_m not in ("", "nan"):
            campos["observaciones"] = obs_m

        if not campos:
            continue

        correo = str(row.get("_correo_key", "")).strip()
        matricula = str(row.get("_matricula_key", "")).strip()
        curso_key = str(row.get("_curso_key", "")).strip()

        if correo and curso_key:
            mapa_correo_curso[(correo, curso_key)] = campos
        if matricula and curso_key:
            mapa_matricula_curso[(matricula, curso_key)] = campos

    # Aplicar ajustes fila por fila
    df = df.copy()
    for idx, row in df.iterrows():
        correo = limpiar_correo(str(row.get("correo_docente", "")))
        matricula = limpiar_id(str(row.get("matricula", "")))
        curso_key = normalizar_texto(str(row.get("curso", "")))

        ajuste = mapa_correo_curso.get((correo, curso_key))
        if not ajuste:
            ajuste = mapa_matricula_curso.get((matricula, curso_key))

        if ajuste:
            for campo, valor in ajuste.items():
                df.at[idx, campo] = valor

    # Re-normalizar avance y recalcular finalizado_oficial tras ajustes
    df["avance_pct"] = pd.to_numeric(df["avance_pct"], errors="coerce").fillna(0).clip(0, 100)
    df["estatus_final"] = (
        df["estatus_final"].astype(str).str.strip().str.upper()
        .replace("NAN", "EN_PROCESO").replace("", "EN_PROCESO")
    )
    df["finalizado_oficial"] = df["estatus_final"] == "FINALIZADO"

    return df



# =====================================================
# INCORPORAR ÁREA DESDE INSCRIPCIONES_SYNC
# =====================================================
def incorporar_area(seguimiento, inscripciones):
    """
    INSCRIPCIONES_SYNC tiene: correo_docente, curso_inscrito, area_adscripcion.
    Se cruza por correo + curso para traer el área a SEGUIMIENTO_ACTUAL.
    Fallback: solo por correo si no hay coincidencia correo+curso.
    """
    df = seguimiento.copy()

    if inscripciones.empty:
        df["area_de_adscripcion"] = "SIN ÁREA"
        return df

    ins = inscripciones.copy()

    col_area_ins = detectar_columna(ins, ALIAS_AREA)
    col_correo_ins = detectar_columna(ins, ALIAS_CORREO)
    col_curso_ins = detectar_columna(ins, ["curso_inscrito", "curso", "taller"])

    if not col_area_ins or not col_correo_ins:
        df["area_de_adscripcion"] = "SIN ÁREA"
        return df

    ins = ins.rename(columns={
        col_area_ins: "area_ins",
        col_correo_ins: "correo_ins",
    })

    if col_curso_ins:
        ins = ins.rename(columns={col_curso_ins: "curso_ins"})
        ins["curso_ins"] = ins["curso_ins"].astype(str).str.strip()
    else:
        ins["curso_ins"] = ""

    ins["correo_ins"] = ins["correo_ins"].apply(limpiar_correo)
    ins["area_ins"] = (
        ins["area_ins"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace("NAN", "SIN ÁREA")
        .replace("", "SIN ÁREA")
    )

    # Mapa 1: correo + curso → área
    mapa_correo_curso = {}
    for _, row in ins.iterrows():
        key = (row["correo_ins"], normalizar_texto(str(row["curso_ins"])))
        if key not in mapa_correo_curso and row["area_ins"] not in ("SIN ÁREA", ""):
            mapa_correo_curso[key] = row["area_ins"]

    # Mapa 2: solo correo → área (fallback)
    mapa_correo = {}
    for _, row in ins.iterrows():
        key = row["correo_ins"]
        if key not in mapa_correo and row["area_ins"] not in ("SIN ÁREA", ""):
            mapa_correo[key] = row["area_ins"]

    def obtener_area(row):
        correo = limpiar_correo(str(row.get("correo_docente", "")))
        curso_key = normalizar_texto(str(row.get("curso", "")))

        area = mapa_correo_curso.get((correo, curso_key))
        if area:
            return area
        area = mapa_correo.get(correo)
        if area:
            return area
        return "SIN ÁREA"

    df["area_de_adscripcion"] = df.apply(obtener_area, axis=1)

    return df


# =====================================================
# FILTROS Y CÁLCULOS
# =====================================================
def filtrar_por_carrera_si_aplica(df, vista, carrera):
    if vista != "Director de carrera" or not carrera:
        return df

    carrera_norm = normalizar_texto(carrera)

    return df[
        df["area_de_adscripcion"].apply(normalizar_texto) == carrera_norm
    ]


def calcular_kpis(df):
    n = len(df)
    docentes_unicos = df["nombre_normalizado"].nunique()
    cursos_unicos = df["curso"].replace("SIN CURSO", pd.NA).dropna().nunique()

    finalizados = int(df["finalizado_oficial"].sum())
    en_proceso = int(n - finalizados)

    return {
        "docentes": int(docentes_unicos),
        "inscripciones": int(n),
        "cursos": int(cursos_unicos),
        "finalizados": finalizados,
        "en_proceso": en_proceso,
        "pct_fin": round(finalizados / n * 100, 1) if n else 0,
        "pct_en_proceso": round(en_proceso / n * 100, 1) if n else 0,
    }


def resumen_por_curso(df):
    registros = []
    df = df[df["curso"] != "SIN CURSO"].copy()

    for curso, grp in df.groupby("curso"):
        n = len(grp)
        finalizados = int(grp["finalizado_oficial"].sum())
        en_proceso = int(n - finalizados)

        registros.append({
            "Curso": curso,
            "Docentes inscritos": grp["nombre_normalizado"].nunique(),
            "Inscripciones": n,
            "% En proceso": round(en_proceso / n * 100, 1) if n else 0,
            "% Finalizados": round(finalizados / n * 100, 1) if n else 0,
            "Finalizados": finalizados,
            "En proceso": en_proceso,
        })

    if not registros:
        return pd.DataFrame(columns=[
            "Curso", "Docentes inscritos", "Inscripciones",
            "% En proceso", "% Finalizados", "Finalizados", "En proceso"
        ])

    return pd.DataFrame(registros).sort_values("Inscripciones", ascending=False)


def resumen_por_area(df):
    registros = []

    for area, grp in df.groupby("area_de_adscripcion"):
        n = len(grp)
        finalizados = int(grp["finalizado_oficial"].sum())
        en_proceso = int(n - finalizados)

        registros.append({
            "Área": area,
            "Docentes inscritos": grp["nombre_normalizado"].nunique(),
            "Inscripciones": n,
            "Finalizados": finalizados,
            "En proceso": en_proceso,
            "% Finalizados": round(finalizados / n * 100, 1) if n else 0,
        })

    return pd.DataFrame(registros).sort_values("Docentes inscritos", ascending=False)


def tabla_resumen_docentes(df):
    grp = df.groupby("nombre_normalizado")

    resumen = grp.agg(
        correo=("correo_docente", "first"),
        area=("area_de_adscripcion", "first"),
        cursos_inscritos=("curso", lambda x: x[x != "SIN CURSO"].nunique()),
        inscripciones=("curso", "count"),
        cursos_finalizados=("finalizado_oficial", "sum"),
    ).reset_index()

    resumen["cursos_finalizados"] = resumen["cursos_finalizados"].astype(int)
    resumen["cursos_en_proceso"] = resumen["inscripciones"] - resumen["cursos_finalizados"]

    resumen["pct_fin"] = (
        resumen["cursos_finalizados"] / resumen["inscripciones"] * 100
    ).round(1).where(resumen["inscripciones"] > 0, 0)

    resumen = resumen.rename(columns={
        "nombre_normalizado": "Docente",
        "correo": "Correo",
        "area": "Área",
        "cursos_inscritos": "Cursos inscritos",
        "inscripciones": "Inscripciones",
        "cursos_finalizados": "Cursos finalizados",
        "cursos_en_proceso": "Cursos en proceso",
        "pct_fin": "% Finalización",
    })

    return resumen.sort_values(["Cursos finalizados", "Cursos inscritos"], ascending=False).reset_index(drop=True)


def docentes_top_finalizados(df):
    t = tabla_resumen_docentes(df)
    t = t[t["Cursos finalizados"] > 0].copy()
    return t[[
        "Docente", "Área", "Cursos inscritos", "Cursos finalizados", "% Finalización"
    ]].sort_values(["Cursos finalizados", "% Finalización"], ascending=False).head(15)


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


def mostrar_kpis_generales(kpis):
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi_card("Docentes inscritos", kpis["docentes"], "Docentes únicos sin duplicar", COLOR_AZUL)

    with c2:
        kpi_card("Inscripciones", kpis["inscripciones"], "Total docente-curso", COLOR_VERDE)

    with c3:
        kpi_card("Cursos activos", kpis["cursos"], "Cursos detectados", COLOR_GRIS)

    with c4:
        kpi_card("Finalizados", kpis["finalizados"], "Según SEGUIMIENTO_ACTUAL", COLOR_VERDE)


def mostrar_tarjetas_finalizacion_por_curso(tabla_cursos):
    st.markdown("### % de finalización por curso")

    if tabla_cursos.empty:
        st.info("No hay cursos detectados para calcular finalización.")
        return

    cols_por_fila = 3
    cursos = tabla_cursos.to_dict("records")

    for i in range(0, len(cursos), cols_por_fila):
        fila = cursos[i:i + cols_por_fila]
        cols = st.columns(cols_por_fila)

        for idx, row in enumerate(fila):
            curso = row["Curso"]
            pct = row["% Finalizados"]
            finalizados = row["Finalizados"]
            inscripciones = row["Inscripciones"]

            with cols[idx]:
                kpi_card(
                    curso,
                    f"{pct}%",
                    f"{finalizados} de {inscripciones} inscripciones finalizadas",
                    COLOR_VERDE,
                )


# =====================================================
# GRÁFICAS
# =====================================================
def grafica_inscritos_area(df):
    resumen = resumen_por_area(df)

    fig = px.bar(
        resumen,
        x="Área",
        y="Docentes inscritos",
        color="Docentes inscritos",
        color_continuous_scale=["#DBEAFE", COLOR_AZUL],
        text="Docentes inscritos",
    )

    fig.update_layout(
        height=420,
        coloraxis_showscale=False,
        xaxis_tickangle=-35,
        yaxis_title="Docentes inscritos",
        margin=dict(t=10, b=90, l=10, r=10),
    )

    return fig


def grafica_finalizacion_curso(df):
    resumen = resumen_por_curso(df)

    if resumen.empty:
        return px.bar(title="No hay cursos detectados")

    fig = px.bar(
        resumen,
        x="Curso",
        y="% Finalizados",
        color="% Finalizados",
        color_continuous_scale=["#DBEAFE", COLOR_VERDE],
        range_color=[0, 100],
        text="% Finalizados",
    )

    fig.update_traces(texttemplate="%{text:.1f}%")
    fig.update_layout(
        height=380,
        coloraxis_showscale=False,
        xaxis_tickangle=-25,
        yaxis_title="% Finalizados",
        margin=dict(t=10, b=90, l=10, r=10),
    )

    return fig


def grafica_proceso_vs_finalizado(df):
    resumen = resumen_por_curso(df)

    if resumen.empty:
        return px.bar(title="No hay cursos detectados")

    datos = resumen.melt(
        id_vars=["Curso"],
        value_vars=["% En proceso", "% Finalizados"],
        var_name="Estatus",
        value_name="Porcentaje",
    )

    datos["Estatus"] = datos["Estatus"].replace({
        "% En proceso": "En proceso",
        "% Finalizados": "Finalizados",
    })

    fig = px.bar(
        datos,
        x="Curso",
        y="Porcentaje",
        color="Estatus",
        color_discrete_map=COLOR_MAP_SIMPLE,
        barmode="stack",
        text="Porcentaje",
    )

    fig.update_traces(texttemplate="%{text:.1f}%")
    fig.update_layout(
        height=380,
        xaxis_tickangle=-25,
        yaxis_title="Porcentaje",
        legend_title_text="Estatus",
        margin=dict(t=10, b=90, l=10, r=10),
    )

    return fig


# =====================================================
# FUNCIÓN PRINCIPAL
# =====================================================
def render_capacitacion_docente(vista=None, carrera=None):
    aplicar_estilos()

    st.title("Capacitación Docente")
    st.caption("Seguimiento de docentes inscritos, cursos activos, avance y finalización.")

    with st.spinner("Cargando datos de capacitación..."):
        df_raw, inscripciones_raw, ajustes_raw = cargar_datos()

    seguimiento = limpiar_seguimiento(df_raw)
    seguimiento = aplicar_ajustes_manuales(seguimiento, ajustes_raw)
    df = incorporar_area(seguimiento, inscripciones_raw)

    df_permitido = filtrar_por_carrera_si_aplica(df, vista, carrera)

    if df_permitido.empty:
        st.warning("No hay registros de capacitación para la carrera/servicio asignado.")
        st.caption(f"Vista: {vista} | Carrera/servicio: {carrera}")
        return

    tabla_cursos_general = resumen_por_curso(df_permitido)
    kpis = calcular_kpis(df_permitido)

    mostrar_kpis_generales(kpis)
    st.divider()

    st.sidebar.markdown("### Capacitación Docente")
    vista_modulo = st.sidebar.radio(
        "Vista del módulo",
        [
            "Resumen",
            "Director de Carrera",
            "Por curso",
            "Rankings",
            "Detalle general",
        ],
        key="nav_capacitacion_docente",
    )

    # ==================================================
    # RESUMEN
    # ==================================================
    if vista_modulo == "Resumen":
        mostrar_tarjetas_finalizacion_por_curso(tabla_cursos_general)
        st.divider()

        st.subheader("Resumen por curso de capacitación")

        st.dataframe(
            tabla_cursos_general[[
                "Curso",
                "Docentes inscritos",
                "Inscripciones",
                "% En proceso",
                "% Finalizados",
                "Finalizados",
                "En proceso",
            ]],
            use_container_width=True,
            hide_index=True,
            key="tabla_resumen_cursos",
            column_config={
                "% En proceso": st.column_config.ProgressColumn(
                    "% En proceso", min_value=0, max_value=100, format="%.1f%%"
                ),
                "% Finalizados": st.column_config.ProgressColumn(
                    "% Finalizados", min_value=0, max_value=100, format="%.1f%%"
                ),
            },
        )

        st.markdown("### Visualización general")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("**Docentes inscritos por área de adscripción**")
            st.plotly_chart(
                grafica_inscritos_area(df_permitido),
                use_container_width=True,
                key="grafica_inscritos_area",
            )

        with col2:
            st.markdown("**% de finalizados por curso**")
            st.plotly_chart(
                grafica_finalizacion_curso(df_permitido),
                use_container_width=True,
                key="grafica_finalizacion_curso",
            )

        if vista != "Director de carrera":
            st.markdown("### Ver detalle por carrera / área")
            areas = sorted(df_permitido["area_de_adscripcion"].dropna().unique())
            area_sel = st.selectbox("Selecciona una carrera / área", areas, key="dg_selector_area_resumen")
            df_area = df_permitido[df_permitido["area_de_adscripcion"] == area_sel]

            st.dataframe(
                tabla_resumen_docentes(df_area),
                use_container_width=True,
                hide_index=True,
                key="tabla_detalle_area_resumen",
                column_config={
                    "% Finalización": st.column_config.ProgressColumn(
                        "% Finalización", min_value=0, max_value=100, format="%.1f%%"
                    ),
                },
            )

    # ==================================================
    # DIRECTOR DE CARRERA
    # ==================================================
    elif vista_modulo == "Director de Carrera":
        if vista == "Director de carrera" and carrera:
            area_sel = carrera
            st.info(f"Vista filtrada para: **{area_sel}**")
            df_area = df_permitido.copy()
        else:
            areas = sorted(df_permitido["area_de_adscripcion"].dropna().unique())
            area_sel = st.selectbox("Selecciona área / carrera", areas, key="sel_area_dc")
            df_area = df_permitido[df_permitido["area_de_adscripcion"] == area_sel]

        st.subheader(f"Seguimiento de capacitación — {area_sel}")

        tabla_cursos_area = resumen_por_curso(df_area)
        mostrar_kpis_generales(calcular_kpis(df_area))
        mostrar_tarjetas_finalizacion_por_curso(tabla_cursos_area)
        st.divider()

        st.markdown("### Resumen por curso de la carrera")
        st.dataframe(
            tabla_cursos_area,
            use_container_width=True,
            hide_index=True,
            key="tabla_cursos_area_dc",
            column_config={
                "% En proceso": st.column_config.ProgressColumn(
                    "% En proceso", min_value=0, max_value=100, format="%.1f%%"
                ),
                "% Finalizados": st.column_config.ProgressColumn(
                    "% Finalizados", min_value=0, max_value=100, format="%.1f%%"
                ),
            },
        )

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("**% finalizados por curso en la carrera**")
            st.plotly_chart(
                grafica_finalizacion_curso(df_area),
                use_container_width=True,
                key="grafica_finalizacion_curso_dc",
            )

        with col2:
            st.markdown("**En proceso vs finalizados por curso**")
            st.plotly_chart(
                grafica_proceso_vs_finalizado(df_area),
                use_container_width=True,
                key="grafica_proceso_vs_finalizado_dc",
            )

        st.markdown("### Docentes de la carrera")
        st.dataframe(
            tabla_resumen_docentes(df_area).drop(columns=["Área"], errors="ignore"),
            use_container_width=True,
            hide_index=True,
            key="tabla_docentes_area",
            column_config={
                "% Finalización": st.column_config.ProgressColumn(
                    "% Finalización", min_value=0, max_value=100, format="%.1f%%"
                ),
            },
        )

    # ==================================================
    # POR CURSO
    # ==================================================
    elif vista_modulo == "Por curso":
        st.subheader("Análisis individual por curso")

        cursos = sorted([c for c in df_permitido["curso"].dropna().unique().tolist() if c != "SIN CURSO"])

        if not cursos:
            st.warning("No hay cursos detectados.")
            return

        curso_sel = st.selectbox("Selecciona curso", cursos, key="sel_curso_capacitacion")
        df_curso = df_permitido[df_permitido["curso"] == curso_sel]

        st.markdown(f"### {curso_sel}")

        resumen_curso = resumen_por_curso(df_curso)
        mostrar_kpis_generales(calcular_kpis(df_curso))
        mostrar_tarjetas_finalizacion_por_curso(resumen_curso)
        st.divider()

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("**% en proceso vs % finalizados**")
            st.plotly_chart(
                grafica_proceso_vs_finalizado(df_curso),
                use_container_width=True,
                key="grafica_proceso_vs_finalizado_curso",
            )

        with col2:
            st.markdown("**Docentes inscritos en el curso**")

            cols = [
                "nombre_normalizado",
                "area_de_adscripcion",
                "correo_docente",
                "avance_pct",
                "estatus_final",
            ]

            tabla = df_curso[cols].copy()

            tabla = tabla.rename(columns={
                "nombre_normalizado": "Docente",
                "area_de_adscripcion": "Área",
                "correo_docente": "Correo",
                "avance_pct": "Avance %",
                "estatus_final": "Estatus",
            }).sort_values("Avance %", ascending=False).reset_index(drop=True)

            st.dataframe(
                tabla,
                use_container_width=True,
                hide_index=True,
                key="tabla_docentes_curso",
                column_config={
                    "Avance %": st.column_config.ProgressColumn(
                        "Avance %", min_value=0, max_value=100, format="%.1f%%"
                    ),
                },
            )

    # ==================================================
    # RANKINGS
    # ==================================================
    elif vista_modulo == "Rankings":
        st.subheader("Rankings destacados")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Top 5 áreas/carreras con más docentes inscritos**")
            r1 = (
                df_permitido.groupby("area_de_adscripcion")["nombre_normalizado"]
                .nunique()
                .reset_index(name="Docentes inscritos")
                .sort_values("Docentes inscritos", ascending=False)
                .head(5)
            )
            st.dataframe(r1, use_container_width=True, hide_index=True, key="rank_top5_areas")

        with col2:
            st.markdown("**Cursos con mayor demanda**")
            r2 = resumen_por_curso(df_permitido)[[
                "Curso", "Docentes inscritos", "Inscripciones", "% Finalizados"
            ]]
            st.dataframe(r2, use_container_width=True, hide_index=True, key="rank_cursos_demanda")

        st.markdown("**Docentes con mayor número de cursos finalizados**")
        r3 = docentes_top_finalizados(df_permitido)

        st.dataframe(
            r3,
            use_container_width=True,
            hide_index=True,
            key="rank_docentes_finalizados_area",
            column_config={
                "% Finalización": st.column_config.ProgressColumn(
                    "% Finalización", min_value=0, max_value=100, format="%.1f%%"
                ),
            },
        )

    # ==================================================
    # DETALLE GENERAL
    # ==================================================
    elif vista_modulo == "Detalle general":
        st.subheader("Tabla completa con filtros")

        f1, f2, f3 = st.columns(3)

        areas_todas = ["Todas"] + sorted(df_permitido["area_de_adscripcion"].dropna().unique().tolist())
        cursos_todos = ["Todos"] + sorted([c for c in df_permitido["curso"].dropna().unique().tolist() if c != "SIN CURSO"])
        estatus_todos = ["Todos"] + sorted(df_permitido["estatus_final"].dropna().unique().tolist())

        area_f = f1.selectbox("Área", areas_todas, key="f_area_detalle")
        curso_f = f2.selectbox("Curso", cursos_todos, key="f_curso_detalle")
        estatus_f = f3.selectbox("Estatus", estatus_todos, key="f_estatus_detalle")

        df_det = df_permitido.copy()

        if area_f != "Todas":
            df_det = df_det[df_det["area_de_adscripcion"] == area_f]
        if curso_f != "Todos":
            df_det = df_det[df_det["curso"] == curso_f]
        if estatus_f != "Todos":
            df_det = df_det[df_det["estatus_final"] == estatus_f]

        cols_mostrar = [
            c for c in [
                "nombre_normalizado",
                "matricula",
                "correo_docente",
                "area_de_adscripcion",
                "curso",
                "avance_pct",
                "tareas_entregadas",
                "tareas_totales",
                "estatus_final",
                "fecha_ultimo_corte",
                "observaciones",
            ]
            if c in df_det.columns
        ]

        tabla_det = df_det[cols_mostrar].rename(columns={
            "nombre_normalizado": "Docente",
            "matricula": "Matrícula",
            "correo_docente": "Correo",
            "area_de_adscripcion": "Área",
            "curso": "Curso",
            "avance_pct": "Avance %",
            "tareas_entregadas": "T. Entregadas",
            "tareas_totales": "T. Totales",
            "estatus_final": "Estatus",
            "fecha_ultimo_corte": "Último corte",
            "observaciones": "Observaciones",
        }).reset_index(drop=True)

        st.caption(f"{len(tabla_det)} registros encontrados")

        st.dataframe(
            tabla_det,
            use_container_width=True,
            hide_index=True,
            key="tabla_detalle_capacitacion",
            column_config={
                "Avance %": st.column_config.ProgressColumn(
                    "Avance %", min_value=0, max_value=100, format="%.1f%%"
                ),
            },
        )
