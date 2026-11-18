from dataclasses import dataclass
import re

QUALITY_MODULE = "encuesta_calidad"
OBSERVATION_MODULE = "observacion_clases"
QUALITY_VIEW = "encuesta_calidad.ver"
QUALITY_GLOBAL = "encuesta_calidad.ver_global"
QUALITY_CAREER = "encuesta_calidad.ver_carrera"
QUALITY_PDF = "encuesta_calidad.descargar_pdf"
QUALITY_COMMENTS = "encuesta_calidad.ver_comentarios"
QUALITY_EXPORT = "encuesta_calidad.exportar_respuestas"
QUALITY_SIA = "encuesta_calidad.usar_sia_cualitativa"
OBSERVATION_VIEW = "observacion_clases.ver"

ROLE_NAMES = {"DG": "Dirección General", "DC": "Dirección de carrera", "DF": "Dirección Finanzas"}
ACTIVE_VALUES = {"TRUE", "1", "SI", "SÍ", "YES", "ACTIVO"}
ACCESS_COLUMNS = {"EMAIL", "ROL", "SERVICIO_ASIGNADO", "ACTIVO", "MODULOS"}


@dataclass(frozen=True)
class Identity:
    issuer: str
    subject: str
    email: str
    name: str


@dataclass(frozen=True)
class Grants:
    role: str
    modules: frozenset[str]
    permissions: frozenset[str]
    services: tuple[str, ...]


def email_key(value: str) -> str:
    return str(value or "").replace("\u200b", "").strip().casefold()


def grants_for(row: dict) -> Grants | None:
    if str(row.get("ACTIVO", "")).strip().upper() not in ACTIVE_VALUES:
        return None
    role = str(row.get("ROL", "")).strip().upper()
    modules = {part.strip().casefold() for part in str(row.get("MODULOS", "")).split(",")}
    services = tuple(dict.fromkeys(
        part.strip() for part in re.split(r"[,|]", str(row.get("SERVICIO_ASIGNADO", ""))) if part.strip()
    )) if role == "DC" else ()
    enabled = set()
    permissions = set()
    if ("all" in modules or QUALITY_MODULE in modules) and (role == "DG" or (role == "DC" and services)):
        enabled.add(QUALITY_MODULE)
        permissions.update({QUALITY_VIEW, QUALITY_CAREER, QUALITY_PDF, QUALITY_COMMENTS})
        if role == "DG":
            permissions.update({QUALITY_GLOBAL, QUALITY_EXPORT})
    if ("all" in modules or OBSERVATION_MODULE in modules) and (role == "DG" or (role == "DC" and services)):
        enabled.add(OBSERVATION_MODULE)
        permissions.add(OBSERVATION_VIEW)
    return Grants(role, frozenset(enabled), frozenset(permissions), services if enabled else ())
