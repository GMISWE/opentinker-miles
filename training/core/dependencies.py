"""
Shared FastAPI Dependencies

Provides dependency injection functions for routers.
"""
from typing import Optional
from fastapi import Depends, Header, Request
from ..utils.auth import APIKeyAuth


def get_auth(request: Request) -> APIKeyAuth:
    """Get the auth manager instance from app state"""
    auth = getattr(request.app.state, "auth", None)
    if auth is None:
        raise RuntimeError("APIKeyAuth not initialized on app state")
    return auth


def get_checkpoint_store(request: Request):
    """The CheckpointStore built at startup (training/checkpoints/store.py)."""
    store = getattr(request.app.state, "checkpoint_store", None)
    if store is None:
        raise RuntimeError("CheckpointStore not initialized on app state")
    return store


async def verify_api_key_dep(
    x_api_key: Optional[str] = Header(None),
    auth: APIKeyAuth = Depends(get_auth)
):
    """Dependency to verify API key"""
    auth.verify(x_api_key)
