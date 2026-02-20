# app.py
import streamlit as st
import pandas as pd  # ✅ IMPORT ARRIBA (evita NameError en type hints)

import encuesta_calidad
import observacion_clases
import aulas_virtuales
import indice_reprobacion  # ✅ NUEVO
import evaluacion_docente  # ✅ NUEVO (módulo Evaluación docente)
from examenes_departamentales import render_examenes_departamentales

# ✅ NUEVO: import defensivo del módulo Encuesta de calidad_F (DF-only)
try:
    import encuesta_calidad_f  # archivo nuevo: encuesta_calidad_f.py
    HAS_EC_F_MOD = True
except Exception:
    encuesta_calidad_f = None
    HAS_EC_F_MOD = False

# ✅ NUEVO: import defensivo del módulo de Bajas/Retención
try:
    import bajas_retencion  # este archivo lo crearás después
    HAS_BAJAS_MOD = True
except Exception:
    bajas_retencion = None
    HAS_BAJAS_MOD = False

# ✅ NUEVO: import defensivo del módulo Seguimiento de Inscripciones
try:
    import seguimiento_inscripciones  # este archivo lo crearás ahora / después
    HAS_SEGUIMIENTO_INS_MOD = True
except Exception:
    seguimiento_inscripciones = None
    HAS_SEGUIMIENTO_INS_MOD = False

import gspread
import json
import re
from google.oauth2.service_account import Credentials
from catalogos import cargar_cat_carreras_desde_gsheets

# ============================================================
# Configuración básica (antes de cualquier st.*)
# ============================================================
st.set_page_config(
    page_title="Dirección Académica",
    layout="wide",
    initial_sidebar_state="expanded",  # ✅ sidebar abierto por defecto
)

DEBUG = False

# ============================================================
# ACCESOS: Sheet donde está la pestaña ACCESOS (tu URL)
# ============================================================
ACCESOS_SHEET_URL = "https://docs.google.com/spreadsheets/d/1CK7nphUH9YS2JqSWRhrgamYoQdgJCsn5tERA-WnwXes/edit?gid=770892546#gid=770892546"
ACCESOS_GID = 770892546
ACCESOS_TAB_NAME = "ACCESOS"

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]

# ============================================================
# Módulos: claves internas (columna MODULOS) y nombres visibles
# ============================================================
MOD_KEY_BY_SECCION = {
    "Encuesta de calidad": "encuesta_calidad",
    # ✅ NUEVO
    "Encuesta de calidad_F": "encuesta_calidad_f",

    "Observación de clases": "observacion_clases",
    "Evaluación docente": "evaluacion_docente",
    "Capacitaciones": "capacitaciones",
    "Índice de reprobación": "indice_reprobacion",
    "Titulación": "titulacion",
    "Ceneval": "ceneval",
    "Exámenes departamentales": "examenes_departamentales",
    "Aulas virtuales": "aulas_virtuales",
    # ✅ NUEVO
    "Bajas / Retención": "bajas_retencion",
    # ✅ NUEVO
    "Seguimiento de Inscripciones": "seguimiento_inscripciones",
}

# ============================================================
# Helpers
# ============================================================
def _extract_sheet_id(url: str) -> str:
    m = re.search(r"/d/([a-zA-Z0-9-_]+)", url or "")
    if not m:
        raise ValueError("No pude extraer el ID del Google Sheet desde la URL.")
    return m.group(1)


def _first_nonempty_row_index(values: list[list[str]]) -> int:
    for i, row in enumerate(values):
        if any(str(c).strip() for c in row):
            return i
    return 0


def _load_creds_dict() -> dict:
    raw = st.secrets["gcp_service_account_json"]
    if isinstance(raw, str):
        return json.loads(raw)
    return dict(raw)


def _norm_email(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\u00A0", "")  # NBSP
    s = s.replace("\u200B", "")  # zero-width space
    s = s.replace(" ", "")
    return s.strip().lower()


def _parse_servicios_cell(cell: str) -> list[str]:
    """
    Permite múltiples servicios/carreras en SERVICIO_ASIGNADO:
      - Separadores: coma (,) o pipe (|)
    """
    if cell is None:
        return []
    txt = str(cell).strip()
    if not txt:
        return []
    parts = re.split(r"[,\|]", txt)
    return [p.strip() for p in parts if p.strip()]


def _goto_seccion(nombre_seccion: str):
    st.session_state["seccion_forzada"] = nombre_seccion
    st.rerun()


def _parse_modulos_cell(modulos_cell: str) -> set[str]:
    """
    Reglas:
      - ALL => acceso total
      - "a,b,c" => set {"a","b","c"}
      - vacío => set vacío (sin módulos)
    """
    if modulos_cell is None:
        return set()
    txt = str(modulos_cell).strip()
    if not txt:
        return set()
    if txt.upper() == "ALL":
        return {"ALL"}
    parts = [p.strip() for p in txt.split(",") if p.strip()]
    return set(parts)


def _is_modulo_visible(mod_key: str) -> bool:
    permitted = st.session_state.get("user_modulos", set())
    allow_all = st.session_state.get("user_allow_all", False)
    if allow_all:
        return True
    return mod_key in permitted


def _placeholder_en_construccion(titulo: str):
    st.subheader(titulo)
    st.warning("📝 En construcción")
    st.caption("Este módulo se habilitará próximamente.")

    st.markdown("**Módulos disponibles:**")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        if _is_modulo_visible(MOD_KEY_BY_SECCION["Observación de clases"]):
            if st.button("🔎 Observación de clases", use_container_width=True, key=f"btn_oc_{titulo}"):
                _goto_seccion("Observación de clases")
        else:
            st.button("🔎 Observación de clases", use_container_width=True, disabled=True, key=f"btn_oc_dis_{titulo}")

    with c2:
        if _is_modulo_visible(MOD_KEY_BY_SECCION["Encuesta de calidad"]):
            if st.button("📋 Encuesta de calidad", use_container_width=True, key=f"btn_ec_{titulo}"):
                _goto_seccion("Encuesta de calidad")
        else:
            st.button("📋 Encuesta de calidad", use_container_width=True, disabled=True, key=f"btn_ec_dis_{titulo}")

    with c3:
        if _is_modulo_visible(MOD_KEY_BY_SECCION["Exámenes departamentales"]):
            if st.button("🧾 Exámenes departamentales", use_container_width=True, key=f"btn_ed_{titulo}"):
                _goto_seccion("Exámenes departamentales")
        else:
            st.button("🧾 Exámenes departamentales", use_container_width=True, disabled=True, key=f"btn_ed_dis_{titulo}")

    with c4:
        if _is_modulo_visible(MOD_KEY_BY_SECCION["Aulas virtuales"]):
            if st.button("🧑‍🏫 Aulas virtuales", use_container_width=True, key=f"btn_av_{titulo}"):
                _goto_seccion("Aulas virtuales")
        else:
            st.button("🧑‍🏫 Aulas virtuales", use_container_width=True, disabled=True, key=f"btn_av_dis_{titulo}")


def _get_logged_in_email() -> str:
    """
    Obtiene el email desde st.user (OIDC).
    Se mantiene defensivo por si cambian atributos disponibles.
    """
    try:
        email = getattr(st.user, "email", None)
        if email:
            return _norm_email(email)
        d = st.user.to_dict() if hasattr(st.user, "to_dict") else {}
        return _norm_email(d.get("email") or d.get("mail") or d.get("preferred_username") or "")
    except Exception:
        return ""


def _show_traceback_expander(title: str = "Ver detalle técnico (diagnóstico)"):
    """Muestra el traceback completo en un expander (sin depender de DEBUG)."""
    import traceback
    with st.expander(title):
        st.code(traceback.format_exc())


# ============================================================
# Normalización de UNIDADES compactadas (DC / ACCESOS)
# ============================================================
def _slug(s: str) -> str:
    s = str(s or "").strip().upper()
    s = re.sub(r"\s+", "", s)
    s = s.replace("_", "")
    return s


# IDs finales acordados
UNIDAD_ID_ALIASES = {
    "EDN": "EDN",
    "ECDG": "ECDG",
    "EDG": "ECDG",  # por si alguien lo escribe así
    "EJEC": "EJEC",
    "EJECUTIVAS": "EJEC",
    "LICENCIATURASEJECUTIVAS": "EJEC",
    "LICENCIATURAEJECUTIVA": "EJEC",
}

UNIDAD_ID_LABEL = {
    "EDN": "EDN — Mercadotecnia / Finanzas / Contaduría / Administración de empresas",
    "ECDG": "ECDG — Diseño Gráfico / Comunicación Multimedia / Cine y TV Digital",
    "EJEC": "EJEC — Licenciaturas Ejecutivas",
}


def _normalize_servicio_asignado(x: str) -> str:
    raw = str(x or "").strip()
    if not raw:
        return ""
    k = _slug(raw)
    if k in UNIDAD_ID_ALIASES:
        return UNIDAD_ID_ALIASES[k]
    return raw


def _display_servicio(x: str) -> str:
    v = str(x or "").strip()
    if not v:
        return ""
    if v in UNIDAD_ID_LABEL:
        return UNIDAD_ID_LABEL[v]
    return v


# ============================================================
# Cliente gspread global (reutilizable)
# ============================================================
@st.cache_resource(show_spinner=False)
def get_gspread_client():
    creds_dict = _load_creds_dict()
    creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    return gspread.authorize(creds)


# ============================================================
# Catálogo maestro: carreras (cacheado)
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def get_cat_carreras_df():
    try:
        gc = get_gspread_client()
        return cargar_cat_carreras_desde_gsheets(gc)
    except Exception:
        return pd.DataFrame(columns=["carrera_id", "nombre_oficial", "variantes", "variante_norm"])


# ============================================================
# Lectura de ACCESOS
# ============================================================
@st.cache_data(ttl=120, show_spinner=False)
def cargar_accesos_df() -> tuple[pd.DataFrame, str]:
    creds_dict = _load_creds_dict()
    sa_email = creds_dict.get("client_email", "")

    client = get_gspread_client()

    sheet_id = _extract_sheet_id(ACCESOS_SHEET_URL)
    sh = client.open_by_key(sheet_id)

    ws = None
    try:
        ws = sh.worksheet(ACCESOS_TAB_NAME)
    except Exception:
        ws = None

    if ws is None:
        try:
            ws = sh.get_worksheet_by_id(ACCESOS_GID)
        except Exception:
            ws = None

    if ws is None:
        ws = sh.sheet1

    values = ws.get_all_values()
    if not values or len(values) < 1:
        return pd.DataFrame(columns=["EMAIL", "ROL", "SERVICIO_ASIGNADO", "ACTIVO", "MODULOS"]), sa_email

    header_idx = _first_nonempty_row_index(values)
    header = [str(c).strip() for c in values[header_idx]]
    data = values[header_idx + 1 :]

    max_cols = max(len(header), max((len(r) for r in data), default=len(header)))
    header = header + [""] * (max_cols - len(header))
    header = [h if h else f"COL_{i+1}" for i, h in enumerate(header)]

    norm_data = []
    for r in data:
        r = [str(c) for c in r]
        r = r + [""] * (max_cols - len(r))
        norm_data.append(r[:max_cols])

    df = pd.DataFrame(norm_data, columns=header)
    df.columns = [str(c).strip().upper() for c in df.columns]

    for col in ["EMAIL", "ROL", "SERVICIO_ASIGNADO", "ACTIVO", "MODULOS"]:
        if col not in df.columns:
            df[col] = ""

    df["EMAIL"] = df["EMAIL"].apply(_norm_email)
    df["ROL"] = df["ROL"].astype(str).str.strip().str.upper()
    df["SERVICIO_ASIGNADO"] = df["SERVICIO_ASIGNADO"].astype(str).str.strip()
    df["MODULOS"] = df["MODULOS"].astype(str).str.strip()

    activo_raw = df["ACTIVO"].astype(str).str.strip().str.upper()
    df["ACTIVO"] = activo_raw.isin(["TRUE", "1", "SI", "SÍ", "YES", "ACTIVO"])

    df = df[df["EMAIL"] != ""]
    df = df[df["ACTIVO"]]

    return df, sa_email


def resolver_permiso_por_email(email: str, df_accesos: pd.DataFrame) -> dict:
    email_norm = _norm_email(email)
    if not email_norm:
        return {"ok": False, "rol": None, "servicios": [], "modulos": set(), "mensaje": "No fue posible obtener el correo del usuario autenticado."}

    fila = df_accesos[df_accesos["EMAIL"] == email_norm]
    if fila.empty:
        return {
            "ok": False,
            "rol": None,
            "servicios": [],
            "modulos": set(),
            "mensaje": "Tu correo autenticado no está habilitado en ACCESOS (o está inactivo).",
        }

    rol = str(fila.iloc[0]["ROL"]).strip().upper()

    servicios_raw = _parse_servicios_cell(fila.iloc[0].get("SERVICIO_ASIGNADO", ""))

    servicios = []
    for s in servicios_raw:
        s2 = _normalize_servicio_asignado(s)
        if s2:
            servicios.append(s2)

    seen = set()
    servicios = [x for x in servicios if not (x in seen or seen.add(x))]

    modulos = _parse_modulos_cell(fila.iloc[0].get("MODULOS", ""))

    if rol not in ["DG", "DC", "DF"]:
        return {"ok": False, "rol": None, "servicios": [], "modulos": set(), "mensaje": "ROL inválido en ACCESOS. Usa DG, DC o DF."}

    if rol == "DC" and not servicios:
        return {"ok": False, "rol": None, "servicios": [], "modulos": set(), "mensaje": "Falta SERVICIO_ASIGNADO (ROL=DC)."}

    if not modulos:
        return {
            "ok": False,
            "rol": None,
            "servicios": [],
            "modulos": set(),
            "mensaje": "Tu usuario no tiene MODULOS asignados en ACCESOS. Coloca ALL o una lista (ej. observacion_clases,aulas_virtuales).",
        }

    return {
        "ok": True,
        "rol": rol,
        "servicios": (servicios if rol == "DC" else []),  # ✅ DF no necesita servicios
        "modulos": modulos,
        "mensaje": "OK",
    }


# ============================================================
# Header (logo + título)
# ============================================================
logo_url = "udl_logo.png"
try:
    col1, col2 = st.columns([1, 5], vertical_alignment="center")
    with col1:
        st.image(logo_url, width=140)
    with col2:
        st.markdown("# Dirección Académica")
        st.caption("Seguimiento del Plan Anual.")
except Exception as e:
    st.warning("No se pudo cargar el logo (esto no detiene la app).")
    if DEBUG:
        st.exception(e)

st.divider()

# ============================================================
# LOGIN / ACCESO (OIDC Google con st.login)
# ============================================================
try:
    is_logged_in = bool(getattr(st.user, "is_logged_in", False))
except Exception:
    is_logged_in = False

if not is_logged_in:
    st.subheader("Acceso")
    st.info("Inicia sesión con Google para acceder a la plataforma.")
    if st.button("Iniciar sesión con Google", use_container_width=True):
        st.login("google")
    st.stop()

if "user_rol" not in st.session_state:
    user_email = _get_logged_in_email()

    try:
        cargar_accesos_df.clear()
        df_accesos, _ = cargar_accesos_df()
        res = resolver_permiso_por_email(user_email, df_accesos)

        if not res["ok"]:
            st.subheader("Acceso")
            st.error(res["mensaje"])
            st.caption(f"Correo autenticado: {user_email or '(no disponible)'}")

            if st.button("Cerrar sesión", use_container_width=True):
                try:
                    st.logout()
                except Exception:
                    pass
                for k in [
                    "user_email",
                    "user_rol",
                    "user_servicios",
                    "user_modulos",
                    "user_allow_all",
                    "carrera_seleccionada_dc",
                ]:
                    st.session_state.pop(k, None)
                st.rerun()
            st.stop()

        st.session_state["user_email"] = user_email
        st.session_state["user_rol"] = res["rol"]
        st.session_state["user_servicios"] = res["servicios"]
        st.session_state["user_modulos"] = res["modulos"]
        st.session_state["user_allow_all"] = ("ALL" in res["modulos"])
        st.session_state.pop("carrera_seleccionada_dc", None)

    except Exception as e:
        st.error("No fue posible validar el acceso en ACCESOS. Revisa permisos del Google Sheet.")
        try:
            sa_email = _load_creds_dict().get("client_email", "")
        except Exception:
            sa_email = ""
        if sa_email:
            st.info(f"Comparte el Sheet de ACCESOS con este correo (Viewer): {sa_email}")

        if DEBUG:
            st.exception(e)
        else:
            _show_traceback_expander()
        st.stop()

# ============================================================
# Catálogo maestro en memoria (disponible para módulos)
# ============================================================
df_cat_carreras = get_cat_carreras_df()
st.session_state["df_cat_carreras"] = df_cat_carreras

# ============================================================
# Contexto de usuario (DG vs DC vs DF)
# ============================================================
ROL = st.session_state["user_rol"]

if ROL == "DG":
    vista = "Dirección General"
    carrera = None
elif ROL == "DF":
    vista = "Dirección Finanzas"
    carrera = None
else:
    vista = "Director de carrera"
    carrera = None  # se define en sidebar

# ============================================================
# Menú lateral (izquierdo)
# ============================================================
with st.sidebar:
    st.markdown("### Navegación")

    st.success(f"Sesión activa:\n{st.session_state.get('user_email','')}")
    if st.button("Salir", use_container_width=True):
        try:
            st.logout()
        except Exception:
            pass
        for k in [
            "user_email",
            "user_rol",
            "user_servicios",
            "user_modulos",
            "user_allow_all",
            "carrera_seleccionada_dc",
        ]:
            st.session_state.pop(k, None)
        st.rerun()

    st.caption(f"Rol: **{ROL}**")
    st.caption(f"Vista: **{vista}**")
    st.divider()

    if ROL == "DC":
        SERVICIOS_DC = st.session_state.get("user_servicios") or []
        if isinstance(SERVICIOS_DC, str):
            SERVICIOS_DC = [SERVICIOS_DC] if SERVICIOS_DC.strip() else []

        SERVICIOS_DC = [_normalize_servicio_asignado(s) for s in SERVICIOS_DC]
        SERVICIOS_DC = [s for s in SERVICIOS_DC if s]

        seen = set()
        SERVICIOS_DC = [x for x in SERVICIOS_DC if not (x in seen or seen.add(x))]

        if len(SERVICIOS_DC) == 1:
            carrera = SERVICIOS_DC[0]
            st.info(f"Acceso a:\n**{_display_servicio(carrera)}**")
        else:
            default_idx = 0
            prev = st.session_state.get("carrera_seleccionada_dc")
            if prev:
                prev = _normalize_servicio_asignado(prev)
            if prev and prev in SERVICIOS_DC:
                default_idx = SERVICIOS_DC.index(prev)

            carrera = st.selectbox(
                "Servicio/Carrera",
                SERVICIOS_DC,
                index=default_idx,
                format_func=_display_servicio,
            )
            st.session_state["carrera_seleccionada_dc"] = carrera
            st.caption("Acceso limitado a tus servicios asignados.")

        st.divider()

    # ============================================================
    # Menú de apartados (Plan anual)
    # ============================================================
    SECCIONES_TODAS = [
        "Encuesta de calidad",
        # ✅ NUEVO
        "Encuesta de calidad_F",

        "Observación de clases",
        "Evaluación docente",
        "Capacitaciones",
        "Índice de reprobación",
        "Titulación",
        "Ceneval",
        "Exámenes departamentales",
        "Aulas virtuales",
        "Bajas / Retención",
        "Seguimiento de Inscripciones",
    ]

    try:
        if st.session_state.get("user_allow_all", False):
            SECCIONES = SECCIONES_TODAS[:]
        else:
            permitted = st.session_state.get("user_modulos", set())
            SECCIONES = [s for s in SECCIONES_TODAS if MOD_KEY_BY_SECCION.get(s, "") in permitted]

        if not SECCIONES:
            st.error("Sin módulos habilitados.")
            st.stop()
    except Exception:
        st.error("Error al filtrar módulos por permisos.")
        _show_traceback_expander()
        st.stop()

    if "seccion_forzada" in st.session_state:
        forced = st.session_state.get("seccion_forzada")
        st.session_state.pop("seccion_forzada", None)
        idx_forzada = SECCIONES.index(forced) if forced in SECCIONES else 0
    else:
        idx_forzada = 0

    seccion = st.selectbox(
        "Apartado del plan anual",
        SECCIONES,
        index=idx_forzada,
    )

if isinstance(carrera, str):
    carrera = carrera.strip()

if df_cat_carreras.empty:
    st.caption(
        "Nota: CAT_CARRERAS aún no está disponible o no se pudo cargar "
        "(esto no bloquea la app)."
    )

st.divider()

# ============================================================
# Bloqueo duro: por si alguien intenta forzar estado previo
# ============================================================
try:
    key = MOD_KEY_BY_SECCION.get(seccion, "")
    if not key:
        st.error("Sección inválida.")
        st.stop()

    if not st.session_state.get("user_allow_all", False):
        if key not in st.session_state.get("user_modulos", set()):
            st.error("Sin acceso a este módulo.")
            st.stop()
except Exception:
    st.error("Error validando permisos del módulo.")
    _show_traceback_expander()
    st.stop()

# ============================================================
# Router
# ============================================================
try:
    if seccion == "Encuesta de calidad":
        encuesta_calidad.render_encuesta_calidad(vista=vista, carrera=carrera)

    # ✅ NUEVO: Encuesta de calidad_F (DF-only)
    elif seccion == "Encuesta de calidad_F":
        if not HAS_EC_F_MOD or encuesta_calidad_f is None:
            st.subheader("Encuesta de calidad_F")
            st.warning("🧪 Módulo habilitado, pero aún no está cargado en el repositorio.")
            st.caption("Crea `encuesta_calidad_f.py` con `render_encuesta_calidad_f()`.")
        else:
            if not hasattr(encuesta_calidad_f, "render_encuesta_calidad_f"):
                st.subheader("Encuesta de calidad_F")
                st.error("El módulo `encuesta_calidad_f.py` no tiene la función `render_encuesta_calidad_f`.")
                st.caption("Define: `def render_encuesta_calidad_f(): ...`")
            else:
                encuesta_calidad_f.render_encuesta_calidad_f()

    elif seccion == "Observación de clases":
        observacion_clases.render_observacion_clases(vista=vista, carrera=carrera)

    elif seccion == "Evaluación docente":
        evaluacion_docente.render_evaluacion_docente(
            vista=vista,
            carrera=carrera,
            ed_url="https://docs.google.com/spreadsheets/d/1bCQmPZ1MIZNpLKBAMYEwBrivfq_YjcxO3zN734Rgt7o/edit?gid=0#gid=0",
        )

    elif seccion == "Capacitaciones":
        _placeholder_en_construccion("Capacitaciones")

    elif seccion == "Índice de reprobación":
        indice_reprobacion.render_indice_reprobacion(vista=vista, carrera=carrera)

    elif seccion == "Titulación":
        _placeholder_en_construccion("Titulación")

    elif seccion == "Ceneval":
        _placeholder_en_construccion("Ceneval")

    elif seccion == "Exámenes departamentales":
        render_examenes_departamentales(
            "https://docs.google.com/spreadsheets/d/1GqlE9SOkSNCdA9mi65hk45uuLAao8GHHoresiyhRfQU/edit",
            vista=vista,
            carrera=carrera,
        )

    elif seccion == "Aulas virtuales":
        try:
            if vista != "Dirección General":
                carrera_forzada = "EDN"  # ajusta si lo necesitas
                aulas_virtuales.mostrar(vista=vista, carrera=carrera_forzada)
            else:
                aulas_virtuales.mostrar(vista=vista, carrera=carrera)
        except Exception:
            st.error("Error al cargar Aulas virtuales.")
            _show_traceback_expander("Detalle técnico Aulas virtuales (diagnóstico)")
            st.stop()

    elif seccion == "Bajas / Retención":
        if not HAS_BAJAS_MOD or bajas_retencion is None:
            st.subheader("Bajas / Retención")
            st.warning("🧪 Módulo habilitado, pero aún no está cargado en el repositorio.")
            st.caption("Siguiente paso: crear `bajas_retencion.py` con `render_bajas_retencion(vista, carrera)`.")
        else:
            if not hasattr(bajas_retencion, "render_bajas_retencion"):
                st.subheader("Bajas / Retención")
                st.error("El módulo `bajas_retencion.py` no tiene la función `render_bajas_retencion`.")
                st.caption("Define: `def render_bajas_retencion(vista: str, carrera: str | None): ...`")
            else:
                bajas_retencion.render_bajas_retencion(vista=vista, carrera=carrera)

    elif seccion == "Seguimiento de Inscripciones":
        if not HAS_SEGUIMIENTO_INS_MOD or seguimiento_inscripciones is None:
            st.subheader("Seguimiento de Inscripciones")
            st.warning("🧪 Módulo habilitado, pero aún no está cargado en el repositorio.")
            st.caption("Siguiente paso: crear `seguimiento_inscripciones.py` con `render_seguimiento_inscripciones(vista, carrera)`.")
        else:
            if not hasattr(seguimiento_inscripciones, "render_seguimiento_inscripciones"):
                st.subheader("Seguimiento de Inscripciones")
                st.error("El módulo `seguimiento_inscripciones.py` no tiene la función `render_seguimiento_inscripciones`.")
                st.caption("Define: `def render_seguimiento_inscripciones(vista: str, carrera: str | None): ...`")
            else:
                seguimiento_inscripciones.render_seguimiento_inscripciones(vista=vista, carrera=carrera)

    else:
        st.subheader("Panel inicial")
        st.write(f"Rol: **{ROL}**")
        st.write(f"Vista actual: **{vista}**")
        st.write(f"Apartado seleccionado: **{seccion}**")

except Exception:
    st.error("Ocurrió un error al cargar el apartado seleccionado.")
    if DEBUG:
        st.exception(Exception)
    else:
        _show_traceback_expander()
    st.stop()
