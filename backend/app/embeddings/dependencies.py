"""FastAPI dependencies for embeddings-gateway-backed routes."""

from typing import cast

from fastapi import Request

from backend.app.embeddings.gateway import EmbeddingsGateway


def get_embeddings_gateway(request: Request) -> EmbeddingsGateway:
    """Return the application's shared embeddings gateway."""

    return cast(EmbeddingsGateway, request.app.state.embeddings_gateway)
