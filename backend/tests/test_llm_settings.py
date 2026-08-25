"""Tests for LLM gateway configuration defaults and overrides."""

from backend.app.config import Settings


def test_llm_settings_have_ollama_cloud_defaults() -> None:
    """Defaults should match the proven agentic-rag configuration."""

    settings = Settings(environment="test")

    assert settings.llm_provider == "ollama"
    assert settings.llm_model == "gpt-oss:120b-cloud"
    assert settings.llm_base_url == "http://localhost:11434"
    assert settings.llm_request_timeout_seconds == 300
    assert settings.llm_num_predict == 1024
    assert settings.llm_retry_attempts == 4
    assert settings.llm_retry_initial_delay_seconds == 2.0


def test_llm_settings_respect_env_overrides(monkeypatch) -> None:
    """GROWTHCREW_-prefixed env vars should override every LLM default."""

    monkeypatch.setenv("GROWTHCREW_LLM_PROVIDER", "local")
    monkeypatch.setenv("GROWTHCREW_LLM_MODEL", "test-model")
    monkeypatch.setenv("GROWTHCREW_LLM_REQUEST_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("GROWTHCREW_LLM_RETRY_ATTEMPTS", "2")

    settings = Settings(environment="test")

    assert settings.llm_provider == "local"
    assert settings.llm_model == "test-model"
    assert settings.llm_request_timeout_seconds == 30
    assert settings.llm_retry_attempts == 2
