"""Frontend configuration loaded from environment variables."""

from functools import lru_cache

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class FrontendSettings(BaseSettings):
    """Validated settings used by the Streamlit application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="GROWTHCREW_",
        extra="ignore",
    )

    backend_url: AnyHttpUrl = "http://127.0.0.1:8000"
    http_timeout_seconds: float = Field(default=5.0, gt=0, le=30)

    @property
    def normalized_backend_url(self) -> str:
        """Return the backend URL without a trailing slash."""

        return str(self.backend_url).rstrip("/")


@lru_cache
def get_frontend_settings() -> FrontendSettings:
    """Return one cached frontend settings instance per process."""

    return FrontendSettings()
