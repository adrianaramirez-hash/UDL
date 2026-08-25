from pathlib import Path

import gspread
from google.oauth2.service_account import Credentials

from backend.CONFIG.settings import settings


SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]


def crear_cliente_google_sheets():
    credentials_path = Path(
        settings.google_application_credentials
    )

    if not credentials_path.is_absolute():
        project_root = Path(__file__).resolve().parents[3]
        credentials_path = project_root / credentials_path

    if not credentials_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de credenciales: {credentials_path}"
        )

    creds = Credentials.from_service_account_file(
        credentials_path,
        scopes=SCOPES,
    )

    return gspread.authorize(creds)


def leer_encabezados(
    spreadsheet_id: str,
    hoja: str,
) -> list[str]:
    client = crear_cliente_google_sheets()
    spreadsheet = client.open_by_key(spreadsheet_id)
    worksheet = spreadsheet.worksheet(hoja)

    return worksheet.row_values(1)


def obtener_ultima_fila_con_datos(
    spreadsheet_id: str,
    hoja: str,
) -> int:
    """
    Devuelve la última fila que contiene un valor
    en la columna A.
    """
    client = crear_cliente_google_sheets()
    spreadsheet = client.open_by_key(spreadsheet_id)
    worksheet = spreadsheet.worksheet(hoja)

    valores = worksheet.col_values(1)

    for indice in range(
        len(valores) - 1,
        -1,
        -1,
    ):
        if str(valores[indice]).strip():
            return indice + 1

    return 0
