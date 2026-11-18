from pathlib import Path
from threading import Lock
from time import monotonic

import gspread
from google.oauth2.service_account import Credentials

from backend.CONFIG.auth import auth_settings
from backend.CONFIG.settings import settings
from backend.CORE.auth.policy import ACCESS_COLUMNS, Grants, email_key, grants_for


def read_access_rows() -> list[dict]:
    if not auth_settings.access_sheet_id or not settings.google_application_credentials:
        raise ValueError("Falta configurar la hoja ACCESOS o su cuenta de servicio")
    path = Path(settings.google_application_credentials)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[3] / path
    creds = Credentials.from_service_account_file(
        path, scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]
    )
    client = gspread.authorize(creds)
    client.set_timeout(10)
    values = client.open_by_key(auth_settings.access_sheet_id).worksheet("ACCESOS").get_all_values()
    populated = iter(row for row in values if any(str(cell).strip() for cell in row))
    headers = [str(cell).strip().upper() for cell in next(populated, [])]
    if not ACCESS_COLUMNS.issubset(headers) or any(headers.count(col) != 1 for col in ACCESS_COLUMNS):
        raise ValueError("La pestaña ACCESOS no contiene encabezados válidos")
    return [dict(zip(headers, row)) for row in populated]


class AccessRepository:
    def __init__(self, load_rows, ttl: int, clock=monotonic):
        self.load_rows = load_rows
        self.ttl = ttl
        self.clock = clock
        self.lock = Lock()
        self.expires = 0.0
        self.grants: dict[str, Grants | None] = {}

    def lookup(self, email: str) -> Grants | None:
        with self.lock:
            if self.clock() >= self.expires:
                updated: dict[str, Grants | None] = {}
                for row in self.load_rows():
                    key = email_key(row.get("EMAIL", ""))
                    if key:
                        # A duplicate is ambiguous, even when one row is inactive.
                        updated[key] = None if key in updated else grants_for(row)
                self.grants = updated
                self.expires = self.clock() + self.ttl
            return self.grants.get(email_key(email))


access_repository = AccessRepository(read_access_rows, auth_settings.access_ttl_seconds)
