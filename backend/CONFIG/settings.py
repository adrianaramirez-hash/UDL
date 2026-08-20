import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    google_application_credentials: str
    oc_sheet_url: str


def cargar_settings() -> Settings:
    return Settings(
        google_application_credentials=os.getenv(
            "GOOGLE_APPLICATION_CREDENTIALS",
            "",
        ).strip(),
        oc_sheet_url=os.getenv(
            "OC_SHEET_URL",
            "",
        ).strip(),
    )


settings = cargar_settings()