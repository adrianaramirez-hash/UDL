from contextvars import ContextVar
from dataclasses import dataclass
from unicodedata import combining, normalize


def scope_key(value) -> str:
    text = normalize("NFKD", str(value or ""))
    return " ".join("".join(c for c in text if not combining(c)).casefold().split())


def service_key(value) -> str:
    key = scope_key(value)
    prefix = "lic. ejecutiva:"
    if key.startswith(prefix):
        key = "licenciatura ejecutiva:" + key[len(prefix):]
    if key.startswith("licenciatura ejecutiva:"):
        key = "licenciatura ejecutiva: " + key.split(":", 1)[1].strip()
    if key == "maestria:educacion especial":
        return "maestria en educacion especial"
    return key


def population(row: dict) -> tuple[str, str]:
    return (
        scope_key(row.get("CARRERA_BASE") or row.get("SERVICIO_PROGRAMA")),
        scope_key(row.get("MODALIDAD_ACADEMICA") or row.get("MODALIDAD_DETALLE")),
    )


@dataclass(frozen=True)
class QualityScope:
    services: frozenset[str]
    service: str
    career: str
    modality: str

    def matches(self, row: dict) -> bool:
        service = service_key(row.get("SERVICIO_PROGRAMA"))
        if service not in self.services or (self.service and service != self.service):
            return False
        career, modality = population(row)
        return (not self.career or career == self.career) and (not self.modality or modality == self.modality)

    def select(self, rows: list[dict]) -> list[dict]:
        return [row for row in rows if self.matches(row)]

    @property
    def cache_key(self) -> tuple:
        return ("asignado", tuple(sorted(self.services)), self.service, self.career, self.modality)


quality_scope: ContextVar[QualityScope | None] = ContextVar("quality_scope", default=None)
quality_permissions: ContextVar[frozenset[str] | None] = ContextVar("quality_permissions", default=None)


def quality_has(permission: str) -> bool:
    permissions = quality_permissions.get()
    return permissions is None or permission in permissions
