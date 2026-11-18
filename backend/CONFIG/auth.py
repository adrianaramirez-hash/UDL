import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from backend.CONFIG import settings as _settings  # Load the project's .env first.


def _seconds(name: str, default: int, maximum: int) -> int:
    value = int(os.getenv(name, "").strip() or default)
    if not 0 < value <= maximum:
        raise ValueError(f"{name} debe estar entre 1 y {maximum} segundos")
    return value


@dataclass(frozen=True)
class AuthSettings:
    public_origin: str = field(default_factory=lambda: os.getenv("AUTH_PUBLIC_ORIGIN", "").strip().rstrip("/"))
    allow_local_http: bool = field(default_factory=lambda: os.getenv("AUTH_ALLOW_HTTP_LOCALHOST") == "1")
    client_id: str = field(default_factory=lambda: os.getenv("GOOGLE_OIDC_CLIENT_ID", "").strip())
    client_secret: str = field(default_factory=lambda: os.getenv("GOOGLE_OIDC_CLIENT_SECRET", "").strip(), repr=False)
    hosted_domain: str = field(default_factory=lambda: os.getenv("GOOGLE_OIDC_HOSTED_DOMAIN", "").strip().lower())
    access_sheet_id: str = field(default_factory=lambda: os.getenv("ACCESOS_SHEET_ID", "").strip())
    session_db: Path = field(default_factory=lambda: Path(os.getenv("AUTH_SESSION_DB", "").strip() or ".runtime/sessions.sqlite3"))
    session_seconds: int = field(default_factory=lambda: _seconds("AUTH_SESSION_SECONDS", 8 * 3600, 24 * 3600))
    access_ttl_seconds: int = field(default_factory=lambda: _seconds("ACCESOS_CACHE_SECONDS", 60, 300))

    def __post_init__(self):
        if not self.public_origin:
            return
        url = urlsplit(self.public_origin)
        if (not url.hostname or url.username is not None or url.password is not None
                or url.path or url.query or url.fragment or self.public_origin != f"{url.scheme}://{url.netloc}"
                or not (url.scheme == "https" or (url.scheme == "http" and self.allow_local_http
                                                   and url.hostname in {"localhost", "127.0.0.1", "::1"}))):
            raise ValueError("AUTH_PUBLIC_ORIGIN debe ser un origen HTTPS o un localhost HTTP habilitado")

    @property
    def secure_cookie(self) -> bool:
        return not self.public_origin.startswith("http://")

    @property
    def redirect_uri(self) -> str:
        return f"{self.public_origin}/api/auth/callback"

    @property
    def configured(self) -> bool:
        return bool(self.public_origin and self.client_id and self.client_secret and self.access_sheet_id)


auth_settings = AuthSettings()
