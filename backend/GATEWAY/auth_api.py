from dataclasses import dataclass
import logging
import secrets

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from google.auth.exceptions import GoogleAuthError
from requests.exceptions import RequestException

from backend.CONFIG.auth import auth_settings
from backend.CORE.auth.google_oidc import exchange_code, login_url
from backend.CORE.auth.policy import Grants, Identity, OBSERVATION_VIEW, QUALITY_GLOBAL, ROLE_NAMES
from backend.REPOSITORIES.auth.access import access_repository
from backend.REPOSITORIES.auth.sessions import session_store

router = APIRouter(prefix="/api/auth", tags=["Acceso"])
logger = logging.getLogger(__name__)
LOGIN_SECONDS = 300
DENIED = "Acceso no autorizado"


def cookie_name(name: str) -> str:
    return f"__Host-udl_{name}" if auth_settings.secure_cookie else f"udl_{name}"


def set_cookie(response: Response, name: str, value: str, lifetime: int):
    response.set_cookie(cookie_name(name), value, max_age=lifetime, path="/", httponly=True,
                        secure=auth_settings.secure_cookie, samesite="lax")


def remove_cookie(response: Response, name: str):
    response.delete_cookie(cookie_name(name), path="/", httponly=True,
                           secure=auth_settings.secure_cookie, samesite="lax")


def authorization(email: str) -> Grants:
    try:
        grants = access_repository.lookup(email)
    except Exception as exc:
        logger.warning("No se pudo consultar ACCESOS (%s)", type(exc).__name__)
        raise HTTPException(503, "No se pudo verificar el acceso") from None
    if grants is None:
        raise HTTPException(403, DENIED)
    return grants


@dataclass(frozen=True)
class Principal:
    identity: Identity
    grants: Grants


def current_principal(request: Request) -> Principal:
    token = request.cookies.get(cookie_name("session"), "")
    identity = session_store.identity(token) if token else None
    if identity is None:
        raise HTTPException(401, "Sesión requerida")
    return Principal(identity, authorization(identity.email))


def require_observation(principal: Principal = Depends(current_principal)) -> Principal:
    if OBSERVATION_VIEW not in principal.grants.permissions:
        raise HTTPException(403, DENIED)
    return principal


def _require_config():
    if not auth_settings.configured:
        raise HTTPException(503, "Inicio de sesión no configurado")


@router.get("/login")
def login():
    _require_config()
    state, nonce, verifier = session_store.begin_login("/", LOGIN_SECONDS)
    response = RedirectResponse(login_url(auth_settings, state, nonce, verifier), status_code=303)
    set_cookie(response, "login", state, LOGIN_SECONDS)
    return response


@router.get("/callback")
def callback(request: Request, state: str = "", code: str = "", error: str = ""):
    _require_config()
    browser_state = request.cookies.get(cookie_name("login"), "")
    state_matches = bool(state and browser_state and secrets.compare_digest(state.encode(), browser_state.encode()))
    state_consumed = identity_verified = access_granted = session_created = cookie_emitted = False
    status = 500
    try:
        if not state_matches:
            raise HTTPException(400, "Inicio de sesión inválido o vencido")
        flow = session_store.consume_login(state)
        state_consumed = flow is not None
        if flow is None or not code or error:
            raise HTTPException(400, "Inicio de sesión inválido o vencido")
        _, nonce, verifier = flow
        try:
            identity = exchange_code(auth_settings, code, nonce, verifier)
        except (ValueError, GoogleAuthError, RequestException) as exc:
            logger.warning("No se pudo validar el inicio con Google (%s)", type(exc).__name__)
            raise HTTPException(401, "No se pudo validar la identidad de Google") from None
        identity_verified = True
        authorization(identity.email)
        access_granted = True
        session = session_store.create(identity, auth_settings.session_seconds, request.cookies.get(cookie_name("session"), ""))
        session_created = True
        response = RedirectResponse(f"{auth_settings.public_origin}/", status_code=303)
        remove_cookie(response, "login")
        set_cookie(response, "session", session, auth_settings.session_seconds)
        cookie_emitted = True
        status = response.status_code
        return response
    except HTTPException as exc:
        status = exc.status_code
        raise
    finally:
        logger.info(
            "OIDC callback status=%s code_present=%s state_present=%s state_matches=%s "
            "state_consumed=%s provider_error_present=%s identity_verified=%s access_granted=%s "
            "session_created=%s cookie_emitted=%s",
            status, bool(code), bool(state), state_matches, state_consumed, bool(error),
            identity_verified, access_granted, session_created, cookie_emitted,
        )


@router.get("/me")
def me(principal: Principal = Depends(current_principal)):
    identity, grants = principal.identity, principal.grants
    return {
        "nombre": identity.name, "email": identity.email, "rol": grants.role,
        "nombre_rol": ROLE_NAMES.get(grants.role, grants.role),
        "modulos": sorted(grants.modules), "permisos": sorted(grants.permissions),
        "alcance": {
            "tipo": "global" if QUALITY_GLOBAL in grants.permissions else "asignado" if grants.services else "sin_acceso",
            "servicios": grants.services,
        },
    }


@router.post("/logout", status_code=204)
def logout(request: Request):
    # Origin checking also covers requests sent by same-site sibling domains.
    if not auth_settings.public_origin or request.headers.get("origin") != auth_settings.public_origin:
        raise HTTPException(403, DENIED)
    if token := request.cookies.get(cookie_name("session")):
        session_store.revoke(token)
    if state := request.cookies.get(cookie_name("login")):
        session_store.consume_login(state)
    response = Response(status_code=204)
    remove_cookie(response, "session")
    remove_cookie(response, "login")
    return response
