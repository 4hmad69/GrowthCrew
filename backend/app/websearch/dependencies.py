"""FastAPI dependencies for web-search-gateway-backed routes."""

from typing import cast

from fastapi import Request

from backend.app.websearch.gateway import WebSearchGateway


def get_web_search_gateway(request: Request) -> WebSearchGateway:
    """Return the application's shared web-search gateway."""

    return cast(WebSearchGateway, request.app.state.web_search_gateway)
