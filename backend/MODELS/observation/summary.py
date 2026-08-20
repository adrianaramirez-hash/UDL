from pydantic import BaseModel


class ObservationSummary(BaseModel):
    observaciones: int
    consolidado: int
    en_proceso: int
    no_consolidado: int
    promedio: float
    casos_criticos: int
    