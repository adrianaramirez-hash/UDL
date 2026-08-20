from typing import Any

from pydantic import BaseModel, Field


class ObservationChatRequest(BaseModel):
    pregunta: str = Field(
        ...,
        min_length=1,
        description="Pregunta en lenguaje natural sobre Observación de Clases.",
    )

    contexto: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Contexto conversacional conservado entre preguntas, "
            "por ejemplo el último docente consultado."
        ),
    )


class ObservationChatResponse(BaseModel):
    respuesta: str
    tipo_respuesta: str

    filtros: dict[str, Any] = Field(
        default_factory=dict,
    )

    datos: dict[str, Any] = Field(
        default_factory=dict,
    )

    contexto: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Contexto actualizado que debe conservar el frontend "
            "para la siguiente pregunta."
        ),
    )