from pathlib import Path

import gspread
import pandas as pd
from google.oauth2.service_account import Credentials

from backend.CONFIG.settings import settings

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]


def cargar_observacion_clases():
    """
    Lee Google Sheets y devuelve:

    - df_respuestas
    - df_cortes
    """

    if not settings.google_application_credentials:
        raise ValueError(
            "Falta configurar GOOGLE_APPLICATION_CREDENTIALS en el archivo .env"
        )

    if not settings.oc_sheet_url:
        raise ValueError(
            "Falta configurar OC_SHEET_URL en el archivo .env"
        )

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

    client = gspread.authorize(creds)

    spreadsheet = client.open_by_url(
        settings.oc_sheet_url
    )

    ws_respuestas = spreadsheet.worksheet(
        "Respuestas de formulario 1"
    )

    ws_cortes = spreadsheet.worksheet(
        "Cortes"
    )

    df_respuestas = pd.DataFrame(
        ws_respuestas.get_all_records()
    )

    df_cortes = pd.DataFrame(
        ws_cortes.get_all_records()
    )

    return df_respuestas, df_cortes