from __future__ import annotations

from dataclasses import dataclass

from backend.CONFIG.settings import settings


@dataclass(frozen=True)
class SIAAIConfig:
    provider: str
    api_key: str
    model: str

    @property
    def configured(self) -> bool:
        return bool(
            self.provider
            and self.api_key
            and self.model
        )


def get_sia_ai_config() -> SIAAIConfig:
    return SIAAIConfig(
        provider=settings.sia_ai_provider.strip(),
        api_key=settings.sia_ai_api_key.strip(),
        model=settings.sia_ai_model.strip(),
    )
