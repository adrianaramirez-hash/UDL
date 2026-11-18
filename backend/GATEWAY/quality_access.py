from fastapi import Depends, HTTPException, Request
from google.auth.exceptions import GoogleAuthError
from gspread.exceptions import APIError
from requests.exceptions import RequestException

from backend.CORE.auth.policy import QUALITY_CAREER, QUALITY_GLOBAL, QUALITY_PDF, QUALITY_VIEW
from backend.CORE.auth.quality_scope import QualityScope, quality_permissions, quality_scope, scope_key, service_key
from backend.GATEWAY.auth_api import DENIED, Principal, current_principal


def resolve_quality_scope(request: Request, principal: Principal = Depends(current_principal)) -> QualityScope | None:
    permissions = principal.grants.permissions
    if not {QUALITY_VIEW, QUALITY_CAREER}.issubset(permissions):
        raise HTTPException(403, DENIED)
    is_pdf = request.url.path.endswith("/informe-resultados.pdf")
    if is_pdf and QUALITY_PDF not in permissions:
        raise HTTPException(403, DENIED)
    if QUALITY_GLOBAL in permissions:
        return None
    params = request.query_params
    selectors = ("servicio", "carrera", "modalidad_academica", "modalidad", "periodo", "alcance", "global")
    if any(len(params.getlist(key)) > 1 for key in selectors):
        raise HTTPException(403, DENIED)
    if (params.get("alcance", "asignado") != "asignado" or "global" in params
            or ("modalidad" in params and not params["modalidad"].strip())):
        raise HTTPException(403, DENIED)

    services = frozenset(filter(None, map(service_key, principal.grants.services)))
    selection = {
        "servicio": service_key(params.get("servicio")),
        "carrera": scope_key(params.get("carrera")),
        "modalidad_academica": scope_key(params.get("modalidad_academica")),
    }
    if (not services or any(key in params and not value for key, value in selection.items())
            or ("servicio" in params and selection["servicio"] not in services)):
        raise HTTPException(403, DENIED)
    scope = QualityScope(services, selection["servicio"], selection["carrera"], selection["modalidad_academica"])

    # A population can be absent in one year and still have a valid comparison.
    from backend.GATEWAY.calidad_api import _CONFIG_MODALIDADES_2025, _CONFIG_MODALIDADES_2026, _obtener_base_preview

    configs = {"2025": _CONFIG_MODALIDADES_2025, "2026": _CONFIG_MODALIDADES_2026}
    period = params.get("periodo", "2026").strip() or "2026"
    source = params.get("modalidad", "Escolarizadas").strip() or "Escolarizadas"
    if is_pdf and (period != "2026" or source != "Escolarizadas"):
        raise HTTPException(400, "El informe corresponde a Escolarizadas, instrumento 2026")
    if period not in configs or source not in configs[period]:
        raise HTTPException(403, DENIED)
    other = "2025" if period == "2026" else "2026"
    try:
        for year in (period, other):
            if source in configs[year] and any(
                scope.matches(row) for row in _obtener_base_preview(source, year)["registros"]
            ):
                return scope
    except (APIError, GoogleAuthError, RequestException):
        raise HTTPException(503, "La fuente de datos no está disponible temporalmente",
                            headers={"Retry-After": "60"}) from None
    raise HTTPException(403, DENIED)


async def require_quality(scope: QualityScope | None = Depends(resolve_quality_scope),
                          principal: Principal = Depends(current_principal)):
    # Propagate the authorized input scope to synchronous handlers and services.
    token = quality_scope.set(scope)
    permissions = quality_permissions.set(principal.grants.permissions)
    try:
        yield
    finally:
        quality_permissions.reset(permissions)
        quality_scope.reset(token)
