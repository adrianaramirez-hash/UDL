from backend.CONFIG.settings import settings
from backend.FUENTES.calidad.google_sheets import (
    abrir_spreadsheet,
    crear_cliente_google_sheets,
)


class CalidadRepository:
    """
    Acceso a los datos de control de Encuesta de Calidad.
    """

    def __init__(self):
        self.client = crear_cliente_google_sheets()
        self.master = abrir_spreadsheet(
            self.client,
            settings.calidad_master_sheet_id,
        )

    def obtener_estado_etl(self) -> list[dict]:
        """
        Devuelve el estado de todas las fuentes
        registrado en 12_ESTADO_ETL.
        """
        worksheet = self.master.worksheet(
            "12_ESTADO_ETL"
        )

        return worksheet.get_all_records()

    def obtener_fuentes_incrementales(self) -> list[dict]:
        """
        Devuelve únicamente las fuentes configuradas
        para carga incremental.
        """
        estados = self.obtener_estado_etl()

        return [
            fila
            for fila in estados
            if str(
                fila.get("MODO_CARGA", "")
            ).strip().upper()
            == "INCREMENTAL"
        ]

    def obtener_estado_fuente(
        self,
        fuente_id: str,
    ) -> dict | None:
        """
        Devuelve el estado ETL de una fuente concreta.
        """
        fuente_buscada = fuente_id.strip().upper()

        for fila in self.obtener_estado_etl():
            actual = str(
                fila.get("FUENTE_ID", "")
            ).strip().upper()

            if actual == fuente_buscada:
                return fila

        return None


    def obtener_resumen_pendientes(self) -> list[dict]:
        """
        Calcula cuántas filas nuevas están pendientes
        en cada fuente incremental.
        """
        from backend.FUENTES.calidad.google_sheets import (
            obtener_ultima_fila_con_datos,
        )

        resumen = []

        for estado in self.obtener_fuentes_incrementales():
            siguiente_fila = int(
                estado["SIGUIENTE_FILA_A_LEER"]
            )

            ultima_fila = obtener_ultima_fila_con_datos(
                str(estado["SPREADSHEET_ID"]).strip(),
                str(estado["HOJA_RESPUESTAS"]).strip(),
            )

            pendientes = max(
                0,
                ultima_fila - siguiente_fila + 1,
            )

            resumen.append(
                {
                    "fuente_id": estado["FUENTE_ID"],
                    "siguiente_fila": siguiente_fila,
                    "ultima_fila": ultima_fila,
                    "pendientes": pendientes,
                }
            )

        return resumen
