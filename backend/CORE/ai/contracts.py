from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class AIResponse:
    provider: str
    model: str
    text: str
    data: dict[str, Any] | None = None


class AIClient(Protocol):
    @property
    def provider(self) -> str:
        ...

    @property
    def model(self) -> str:
        ...

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> AIResponse:
        ...

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any] | None = None,
    ) -> AIResponse:
        ...
