import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    google_application_credentials: str
    oc_sheet_url: str

    calidad_master_sheet_id: str
    calidad_esc_main_sheet_id: str
    calidad_vir_main_sheet_id: str
    calidad_pre_main_sheet_id: str
    calidad_esc_archive_sheet_id: str


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
        calidad_master_sheet_id=os.getenv(
            "CALIDAD_MASTER_SHEET_ID",
            "1Y-UIrRC0FykccgK-620Zta5piXsB2vIXL1_lAhYjoXI",
        ).strip(),
        calidad_esc_main_sheet_id=os.getenv(
            "CALIDAD_ESC_MAIN_SHEET_ID",
            "16utee3iOfSIGaY1tW1FUWbPCToIWR_2if2u9bw4PNDA",
        ).strip(),
        calidad_vir_main_sheet_id=os.getenv(
            "CALIDAD_VIR_MAIN_SHEET_ID",
            "1j-u1kWdt7rqe92OpAIIq2PvBVdfkKcspa_myoGhBAn4",
        ).strip(),
        calidad_pre_main_sheet_id=os.getenv(
            "CALIDAD_PRE_MAIN_SHEET_ID",
            "1i8jivO6uRNicWRhz2OqHOeHgOpf4MeBm34AGnQbydL0",
        ).strip(),
        calidad_esc_archive_sheet_id=os.getenv(
            "CALIDAD_ESC_ARCHIVE_SHEET_ID",
            "1bNjMD6GJpW8b1mM166IN99fdM0mLqKSW3aA_hs1h-2s",
        ).strip(),
    )


settings = cargar_settings()
