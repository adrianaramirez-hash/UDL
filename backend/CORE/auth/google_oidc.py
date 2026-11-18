from base64 import urlsafe_b64encode
from hashlib import sha256
import logging
import secrets
from urllib.parse import urlencode

import requests
from google.auth.transport.requests import Request
from google.oauth2 import id_token

from backend.CONFIG.auth import AuthSettings
from backend.CORE.auth.policy import Identity, email_key

ISSUER = "https://accounts.google.com"
AUTH_URL = f"{ISSUER}/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
logger = logging.getLogger(__name__)


def login_url(settings: AuthSettings, state: str, nonce: str, verifier: str) -> str:
    challenge = urlsafe_b64encode(sha256(verifier.encode()).digest()).decode().rstrip("=")
    return f"{AUTH_URL}?{urlencode({
        'client_id': settings.client_id, 'redirect_uri': settings.redirect_uri,
        'response_type': 'code', 'scope': 'openid email profile', 'state': state, 'nonce': nonce,
        'code_challenge': challenge, 'code_challenge_method': 'S256', 'prompt': 'select_account',
    })}"


def verified_identity(claims: dict, settings: AuthSettings, nonce: str) -> Identity:
    subject = claims.get("sub")
    email = email_key(claims.get("email", ""))
    domain = str(claims.get("hd", "")).lower()
    received_nonce = claims.get("nonce")
    if (claims.get("iss") not in {ISSUER, "accounts.google.com"}
            or claims.get("aud") != settings.client_id
            or claims.get("azp", settings.client_id) != settings.client_id
            or claims.get("email_verified") is not True
            or not isinstance(subject, str) or not subject.strip() or not email
            or not isinstance(received_nonce, str) or not secrets.compare_digest(received_nonce.encode(), nonce.encode())
            or (settings.hosted_domain and domain != settings.hosted_domain)
            or (not domain and not email.endswith("@gmail.com"))):
        raise ValueError("La identidad Google no cumple los requisitos")
    return Identity(ISSUER, subject, email, str(claims.get("name") or email))


def exchange_code(settings: AuthSettings, code: str, nonce: str, verifier: str) -> Identity:
    response = requests.post(
        TOKEN_URL,
        data={
            "code": code, "client_id": settings.client_id, "client_secret": settings.client_secret,
            "redirect_uri": settings.redirect_uri, "grant_type": "authorization_code", "code_verifier": verifier,
        },
        timeout=10,
        allow_redirects=False,
    )
    response.raise_for_status()
    logger.info("OIDC token_exchange_accepted=True")
    token = response.json().get("id_token")
    if not isinstance(token, str) or not token:
        raise ValueError("Google no devolvió un ID token")
    claims = id_token.verify_oauth2_token(token, Request(), settings.client_id)
    logger.info("OIDC signed_claims=True email_verified=%s", claims.get("email_verified") is True)
    return verified_identity(claims, settings, nonce)
