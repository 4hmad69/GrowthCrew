"""Application configuration loaded from environment variables."""

from functools import lru_cache
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated backend settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="GROWTHCREW_",
        extra="ignore",
    )

    app_name: str = "GrowthCrew API"
    app_version: str = "0.3.0"
    environment: Literal[
        "development",
        "test",
        "production",
    ] = "development"

    api_v1_prefix: str = Field(
        default="/api/v1",
        pattern=r"^/",
    )
    log_level: Literal[
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ] = "INFO"

    database_url: SecretStr | None = None
    database_echo: bool = False
    database_pool_size: int = Field(
        default=5,
        ge=1,
        le=50,
    )
    database_max_overflow: int = Field(
        default=5,
        ge=0,
        le=100,
    )
    database_pool_timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=120,
    )
    database_pool_recycle_seconds: int = Field(
        default=1800,
        ge=60,
        le=86400,
    )
    database_connect_timeout_seconds: int = Field(
        default=5,
        ge=1,
        le=60,
    )

    llm_provider: Literal["ollama", "local"] = "ollama"
    llm_model: str = "gpt-oss:120b-cloud"
    llm_base_url: str = "http://localhost:11434"
    llm_request_timeout_seconds: int = Field(
        default=300,
        ge=1,
        le=1800,
    )
    llm_num_predict: int = Field(
        default=1024,
        ge=1,
        le=32768,
    )
    llm_retry_attempts: int = Field(
        default=4,
        ge=0,
        le=10,
    )
    llm_retry_initial_delay_seconds: float = Field(
        default=2.0,
        ge=0.1,
        le=60.0,
    )
    llm_cost_per_1k_input_tokens: float = Field(
        default=0.0,
        ge=0.0,
    )
    llm_cost_per_1k_output_tokens: float = Field(
        default=0.0,
        ge=0.0,
    )
    embeddings_provider: Literal["ollama", "local"] = "ollama"
    embeddings_model: str = "nomic-embed-text"
    embeddings_dimension: int = Field(
        default=768,
        ge=1,
        le=4096,
    )
    embeddings_request_timeout_seconds: int = Field(
        default=60,
        ge=1,
        le=600,
    )

    @model_validator(mode="after")
    def validate_production_database_configuration(
        self,
    ) -> Self:
        """Require non-placeholder database configuration in production."""

        if self.environment != "production":
            return self

        if self.database_url is None:
            raise ValueError("GROWTHCREW_DATABASE_URL is required in production.")

        url_value = self.database_url.get_secret_value()

        if "local-development-only" in url_value:
            raise ValueError("A local development database URL cannot be used in production.")

        return self

    @property
    def database_url_value(self) -> str | None:
        """Return the database URL only to trusted infrastructure code."""

        if self.database_url is None:
            return None

        return self.database_url.get_secret_value()


@lru_cache
def get_settings() -> Settings:
    """Return one cached settings instance for the process."""

    return Settings()
