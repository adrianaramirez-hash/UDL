# app.py
import streamlit as st
import pandas as pd
import gspread
import json
import re
from google.oauth2.service_account import Credentials

import encuesta_calidad
import observacion_clases
import aulas_virtuales
import indice_reprobacion
import evaluacion_docente
from examenes_departamentales import render_examenes_departamentales


# ============================================================
# IMPORTS DEFENSIVOS DE MÓDULOS
# ============================================================
try:
    import capacitacion_docente
    HAS_CAPACITACION_MOD = True
    ERROR_CAP = None
except Exception as e:
    capacitacion_docente = None
    HAS_CAPACITACION_MOD = False
    ERROR_CAP = e

try:
    import encuesta_calidad_f
    HAS_EC_F_MOD = True
except Exception:
    encuesta_calidad_f = None
    HAS_EC_F_MOD = False

try:
    import bajas_retencion
    HAS_BAJAS_MOD = True
except Exception:
    bajas_retencion = None
    HAS_BAJAS_MOD = False

try:
    import seguimiento_inscripciones
    HAS_SEGUIMIENTO_INS_MOD = True
except Exception:
    seguimiento_inscripciones = None
    HAS_SEGUIMIENTO_INS_MOD = False


# ============================================================
# CONFIG STREAMLIT
# ============================================================
st.set_page_config(
    page_title="Dirección Académica",
    layout="wide",
    initial_sidebar_state="expanded"
)

DEBUG = False


# ============================================================
# CONFIG ACCESOS
# ============================================================
ACCESOS_SHEET_URL = "https://docs.google.com/spreadsheets/d/1CK7nphUH9YS2JqSWRhrgamYoQdgJCsn5tERA-WnwXes/edit?gid=770892546#gid=770892546"
ACCESOS_GID = 770892546
ACCESOS_TAB_NAME = "ACCESOS"

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]


# ============================================================
# MÓDULOS Y CLAVES DE PERMISO
# ============================================================
MOD_KEY_BY_SECCION = {
    "Encuesta de calidad": "encuesta_calidad",
    "Observación de clases": "observacion_clases",
    "Evaluación docente": "evaluacion_docente",
    "Capacitación Docente": "capacitacion_docente",
    "Índice de reprobación": "indice_reprobacion",
    "Exámenes departamentales": "examenes_departamentales",
    "Aulas virtuales": "aulas_virtuales",
    "Bajas / Retención": "bajas_retencion",
    "Seguimiento de Inscripciones": "seguimiento_inscripciones",
}


SECCIONES_TODAS = [
    "Encuesta de calidad",
    "Observación de clases",
    "Evaluación docente",
    "Capacitación Docente",
    "Índice de reprobación",
    "Exámenes departamentales",
    "Aulas virtuales",
    "Bajas / Retención",
    "Seguimiento de Inscripciones",
]


# ============================================================
# HELPERS GENERALES
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
    s = s.replace("\u00A0", "")
    s = s.replace("\u200B", "")
    s = s.replace(" ", "")
    return s.strip().lower()


def _parse_servicios_cell(cell: str) -> list[str]:
    if cell is None:
        return []
    txt = str(cell).strip()
    if not txt:
        return []
    parts = re.split(r"[,\|]", txt)
    return [p.strip() for p in parts if p.strip()]


def _parse_modulos_cell(modulos_cell: str) -> set[str]:
    if modulos_cell is None:
        return set()

    txt = str(modulos_cell).strip()

    if not txt:
        return set()

    if txt.upper() == "ALL":
        return {"ALL"}

    parts = [p.strip() for p in txt.split(",") if p.strip()]
    return set(parts)


def _slug(s: str) -> str:
    s = str(s or "").strip().upper()
    s = re.sub(r"\s+", "", s)
    s = s.replace("_", "")
    return s


UNIDAD_ID_ALIASES = {
    "EDN": "EDN",
    "ECDG": "ECDG",
    "EDG": "ECDG",
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


def _get_logged_in_email() -> str:
    try:
        email = getattr(st.user, "email", None)

        if email:
            return _norm_email(email)

        d = st.user.to_dict() if hasattr(st.user, "to_dict") else {}

        return _norm_email(
            d.get("email")
            or d.get("mail")
            or d.get("preferred_username")
            or ""
        )

    except Exception:
        return ""


def _show_traceback_expander(title: str = "Ver detalle técnico"):
    import traceback
    with st.expander(title):
        st.code(traceback.format_exc())


def _logout_and_clear():
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
        "seccion_forzada",
    ]:
        st.session_state.pop(k, None)

    st.rerun()


def _is_modulo_visible(mod_key: str) -> bool:
    permitted = st.session_state.get("user_modulos", set())
    allow_all = st.session_state.get("user_allow_all", False)

    if allow_all:
        return True

    return mod_key in permitted


# ============================================================
# CLIENTE GSPREAD
# ============================================================
@st.cache_resource(show_spinner=False)
def get_gspread_client():
    creds_dict = _load_creds_dict()
    creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    return gspread.authorize(creds)


# ============================================================
# CARGA DE ACCESOS
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

    if not values:
        return pd.DataFrame(
            columns=["EMAIL", "ROL", "SERVICIO_ASIGNADO", "ACTIVO", "MODULOS"]
        ), sa_email

    header_idx = _first_nonempty_row_index(values)
    header = [str(c).strip() for c in values[header_idx]]
    data = values[header_idx + 1:]

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
        return {
            "ok": False,
            "rol": None,
            "servicios": [],
            "modulos": set(),
            "mensaje": "No fue posible obtener el correo del usuario autenticado.",
        }

    fila = df_accesos[df_accesos["EMAIL"] == email_norm]

    if fila.empty:
        return {
            "ok": False,
            "rol": None,
            "servicios": [],
            "modulos": set(),
            "mensaje": "Tu correo autenticado no está habilitado en ACCESOS o está inactivo.",
        }

    rol = str(fila.iloc[0]["ROL"]).strip().upper()

    servicios_raw = _parse_servicios_cell(
        fila.iloc[0].get("SERVICIO_ASIGNADO", "")
    )

    servicios = []

    for s in servicios_raw:
        s2 = _normalize_servicio_asignado(s)
        if s2:
            servicios.append(s2)

    seen = set()
    servicios = [x for x in servicios if not (x in seen or seen.add(x))]

    modulos = _parse_modulos_cell(fila.iloc[0].get("MODULOS", ""))

    if rol not in ["DG", "DC", "DF"]:
        return {
            "ok": False,
            "rol": None,
            "servicios": [],
            "modulos": set(),
            "mensaje": "ROL inválido en ACCESOS. Usa DG, DC o DF.",
        }

    if rol == "DC" and not servicios:
        return {
            "ok": False,
            "rol": None,
            "servicios": [],
            "modulos": set(),
            "mensaje": "Falta SERVICIO_ASIGNADO para este usuario con ROL=DC.",
        }

    if not modulos:
        return {
            "ok": False,
            "rol": None,
            "servicios": [],
            "modulos": set(),
            "mensaje": "Tu usuario no tiene MODULOS asignados en ACCESOS. Coloca ALL o una lista de módulos.",
        }

    return {
        "ok": True,
        "rol": rol,
        "servicios": servicios if rol == "DC" else [],
        "modulos": modulos,
        "mensaje": "OK",
    }


# ============================================================
# HEADER
# ============================================================
st.title("🎓 Dirección Académica")
st.caption("Ecosistema académico")
st.divider()


# ============================================================
# LOGIN CON GOOGLE
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


# ============================================================
# VALIDACIÓN CONTRA HOJA ACCESOS
# ============================================================
if "user_rol" not in st.session_state:
    user_email = _get_logged_in_email()

    try:
        cargar_accesos_df.clear()
        df_accesos, _ = cargar_accesos_df()
        res = resolver_permiso_por_email(user_email, df_accesos)

        if not res["ok"]:
            st.subheader("Acceso no autorizado")
            st.error(res["mensaje"])
            st.caption(f"Correo autenticado: {user_email or '(no disponible)'}")

            if st.button("Cerrar sesión", use_container_width=True):
                _logout_and_clear()

            st.stop()

        st.session_state["user_email"] = user_email
        st.session_state["user_rol"] = res["rol"]
        st.session_state["user_servicios"] = res["servicios"]
        st.session_state["user_modulos"] = res["modulos"]
        st.session_state["user_allow_all"] = "ALL" in res["modulos"]
        st.session_state.pop("carrera_seleccionada_dc", None)

    except Exception:
        st.error("No fue posible validar el acceso en la hoja ACCESOS.")

        try:
            sa_email = _load_creds_dict().get("client_email", "")
        except Exception:
            sa_email = ""

        if sa_email:
            st.info(f"Comparte el Sheet de ACCESOS con este correo como lector: {sa_email}")

        _show_traceback_expander("Ver error técnico de validación")
        st.stop()


# ============================================================
# CONTEXTO DE USUARIO
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
    carrera = None


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### Navegación")

    st.success(f"Sesión activa:\n{st.session_state.get('user_email', '')}")

    if st.button("Salir", use_container_width=True):
        _logout_and_clear()

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

    if st.session_state.get("user_allow_all", False):
        SECCIONES = SECCIONES_TODAS[:]
    else:
        permitted = st.session_state.get("user_modulos", set())
        SECCIONES = [
            s for s in SECCIONES_TODAS
            if MOD_KEY_BY_SECCION.get(s, "") in permitted
        ]

    if not SECCIONES:
        st.error("Sin módulos habilitados.")
        st.stop()

    seccion = st.selectbox("Selecciona módulo", SECCIONES)


# ============================================================
# BLOQUEO DURO POR MÓDULO
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
# ROUTER
# ============================================================
try:
    if seccion == "Encuesta de calidad":

        if vista == "Dirección Finanzas":
            if HAS_EC_F_MOD and hasattr(encuesta_calidad_f, "render_encuesta_calidad_f"):
                encuesta_calidad_f.render_encuesta_calidad_f(
                    vista=vista,
                    carrera=carrera
                )
            else:
                st.error("El módulo de Encuesta de calidad para Finanzas no está disponible.")
        else:
            encuesta_calidad.render_encuesta_calidad(
                vista=vista,
                carrera=carrera
            )

    elif seccion == "Observación de clases":
        observacion_clases.render_observacion_clases(
            vista=vista,
            carrera=carrera
        )

    elif seccion == "Evaluación docente":
        evaluacion_docente.render_evaluacion_docente(
            vista=vista,
            carrera=carrera
        )

    elif seccion == "Capacitación Docente":

        if not HAS_CAPACITACION_MOD:
            st.error("❌ No se pudo importar el módulo de Capacitación Docente.")

            with st.expander("Ver error real"):
                st.exception(ERROR_CAP)

        else:
            if hasattr(capacitacion_docente, "render_capacitacion_docente"):
                capacitacion_docente.render_capacitacion_docente(
                    vista=vista,
                    carrera=carrera
                )

            elif hasattr(capacitacion_docente, "mostrar_modulo_capacitacion_docente"):
                capacitacion_docente.mostrar_modulo_capacitacion_docente()

            else:
                st.error("❌ El archivo capacitacion_docente.py no tiene una función válida.")
                st.caption("Debe tener `render_capacitacion_docente(vista, carrera)` o `mostrar_modulo_capacitacion_docente()`.")

    elif seccion == "Índice de reprobación":
        indice_reprobacion.render_indice_reprobacion(
            vista=vista,
            carrera=carrera
        )

    elif seccion == "Exámenes departamentales":
        render_examenes_departamentales(
            "https://docs.google.com/spreadsheets/d/1GqlE9SOkSNCdA9mi65hk45uuLAao8GHHoresiyhRfQU/edit",
            vista=vista,
            carrera=carrera
        )

    elif seccion == "Aulas virtuales":
        aulas_virtuales.mostrar(
            vista=vista,
            carrera=carrera
        )

    elif seccion == "Bajas / Retención":
        if HAS_BAJAS_MOD and hasattr(bajas_retencion, "render_bajas_retencion"):
            bajas_retencion.render_bajas_retencion(
                vista=vista,
                carrera=carrera
            )
        else:
            st.warning("Módulo Bajas / Retención no disponible.")

    elif seccion == "Seguimiento de Inscripciones":
        if HAS_SEGUIMIENTO_INS_MOD and hasattr(seguimiento_inscripciones, "render_seguimiento_inscripciones"):
            seguimiento_inscripciones.render_seguimiento_inscripciones(
                vista=vista,
                carrera=carrera
            )
        else:
            st.warning("Módulo Seguimiento de Inscripciones no disponible.")

    else:
        st.subheader("Panel inicial")
        st.write(f"Rol: **{ROL}**")
        st.write(f"Vista actual: **{vista}**")
        st.write(f"Módulo seleccionado: **{seccion}**")

except Exception:
    st.error("Ocurrió un error al cargar el módulo seleccionado.")
    _show_traceback_expander("Ver detalle técnico del módulo")
    st.stop()
