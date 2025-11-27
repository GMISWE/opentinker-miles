#!/usr/bin/env python3
"""
Training API Server for kgateway AI Extension.

Runs FastAPI server implementing Tinker API for RL training.
"""
import os
import logging
import asyncio
import uvicorn

from .api import app
from .config import get_config

logger = logging.getLogger(__name__).getChild("training-server")


async def serve():
    """Run the training API server."""
    # Use centralized configuration
    cfg = get_config()

    logger.info(f"Starting Training API server on {cfg.server.host}:{cfg.server.port}")
    logger.info(f"Access logging: {'enabled' if cfg.server.access_log else 'disabled'} (set KGATEWAY_ACCESS_LOG=true to enable)")

    config = uvicorn.Config(
        app=app,
        host=cfg.server.host,
        port=cfg.server.port,
        log_level=cfg.server.log_level.lower(),
        access_log=cfg.server.access_log,
    )

    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    cfg = get_config()
    logging.basicConfig(
        level=cfg.server.log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    asyncio.run(serve())
