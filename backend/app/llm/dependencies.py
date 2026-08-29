"""FastAPI dependencies for LLM-gateway-backed routes."""

from typing import cast

from fastapi import Request

from backend.app.llm.gateway import LLMGateway


def get_llm_gateway(request: Request) -> LLMGateway:
    """Return the application's shared LLM gateway."""

    return cast(LLMGateway, request.app.state.llm_gateway)
