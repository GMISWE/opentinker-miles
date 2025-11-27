"""
Health Router - System Health and Capabilities

Simple router for health checks and server capabilities.
No service layer needed - just queries system state.
"""
import logging
import os
import json
import ray
from datetime import datetime
from fastapi import APIRouter, Depends, Request

from ..models.responses import HealthResponse, ServerCapabilities, ModelInfo
from ..core.dependencies import verify_api_key_dep
from ..config import TrainingConfig

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    tags=["health"]
)

def _get_runtime(request: Request):
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise RuntimeError("Training runtime state not initialized")
    return runtime


def _get_config(request: Request) -> TrainingConfig:
    config = getattr(request.app.state, "config", None)
    if config is None:
        raise RuntimeError("Training config not initialized")
    return config


@router.get("/health", response_model=HealthResponse)
async def health_simple(request: Request):
    """Health check for k8s probes (backward compatibility)"""
    runtime = _get_runtime(request)
    training_clients = runtime.training_clients
    futures_store = runtime.futures_store

    return HealthResponse(
        status="healthy",
        version="3.1.0",
        timestamp=datetime.now().isoformat(),
        ray_initialized=ray.is_initialized(),
        active_training_clients=len(training_clients),
        model_ids=list(training_clients.keys()),
        futures_count=len(futures_store)
    )


@router.get("/api/v1/health", response_model=HealthResponse)
async def health(request: Request):
    """Health check endpoint - refactored with typed response"""
    runtime = _get_runtime(request)
    training_clients = runtime.training_clients
    futures_store = runtime.futures_store

    return HealthResponse(
        status="healthy",
        version="3.1.0",
        timestamp=datetime.now().isoformat(),
        ray_initialized=ray.is_initialized(),
        active_training_clients=len(training_clients),
        model_ids=list(training_clients.keys()),
        futures_count=len(futures_store)
    )


@router.get("/api/v1/get_server_capabilities", response_model=ServerCapabilities)
async def get_server_capabilities(
    request: Request,
    _: None = Depends(verify_api_key_dep)
):
    """Get server capabilities - refactored with config and typed response"""
    config = _get_config(request)

    if config.supported_models:
        supported_models = [
            ModelInfo(
                model_name=model.model_name,
                max_context_length=model.max_context_length,
                supports_lora=model.supports_lora
            )
            for model in config.supported_models
        ]
    else:
        # Fallback to environment or defaults
        default_models = [
            {
                "model_name": "/data/models/Qwen2.5-0.5B-Instruct_torch_dist",
                "max_context_length": 512,
                "supports_lora": True
            }
        ]
        env_models = os.getenv("SUPPORTED_MODELS")
        if env_models:
            try:
                models_config = json.loads(env_models)
            except Exception as e:
                logger.warning(f"Failed to parse SUPPORTED_MODELS env var: {e}")
                models_config = default_models
        else:
            models_config = default_models

        supported_models = [
            ModelInfo(
                model_name=model_config.get("model_name", "unknown"),
                max_context_length=model_config.get("max_context_length", 512),
                supports_lora=model_config.get("supports_lora", True),
            )
            for model_config in models_config
        ]

    return ServerCapabilities(
        supported_models=supported_models,
        features=["gradient_accumulation", "lora", "checkpointing"],
        version="3.1.0"
    )


@router.post("/api/v1/telemetry")
async def send_telemetry(request: Request):
    """
    Telemetry endpoint - stub that accepts and discards data.

    No API key verification needed - telemetry is optional and non-critical.
    """
    return {"status": "accepted"}
