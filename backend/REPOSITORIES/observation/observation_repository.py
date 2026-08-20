import pandas as pd

from backend.FUENTES.observation.google_sheets import (
    cargar_observacion_clases,
)


class ObservationRepository:
    """
    Repositorio encargado de obtener los datos
    de Observación de Clases.
    """

    def obtener_datos(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Devuelve:

        df_respuestas
        df_cortes
        """

        return cargar_observacion_clases()