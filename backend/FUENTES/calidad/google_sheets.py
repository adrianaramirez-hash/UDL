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


def _columna_a_numero(columna: str) -> int:
    """
    Convierte una letra de columna de Google Sheets
    a su número: A=1, Z=26, AA=27...
    """
    resultado = 0

    for caracter in columna.strip().upper():
        resultado = (
            resultado * 26
            + ord(caracter)
            - ord("A")
            + 1
        )

    return resultado


def leer_filas(
    spreadsheet_id: str,
    hoja: str,
    fila_inicio: int,
    fila_fin: int,
    columna_fin: str,
) -> list[list]:
    """
    Lee un bloque exacto de filas desde la columna A
    hasta columna_fin.

    No modifica ningún dato.
    """
    if fila_fin < fila_inicio:
        return []

    client = crear_cliente_google_sheets()
    spreadsheet = client.open_by_key(spreadsheet_id)
    worksheet = spreadsheet.worksheet(hoja)

    rango = (
        f"A{fila_inicio}:"
        f"{columna_fin.strip().upper()}{fila_fin}"
    )

    filas = worksheet.get(rango)

    ancho = _columna_a_numero(
        columna_fin
    )

    return [
        list(fila)
        + [""] * max(
            0,
            ancho - len(fila),
        )
        for fila in filas
    ]
