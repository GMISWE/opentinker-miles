"""
Checkpoints Router - HTTP Layer for Checkpoint Management

Endpoints:
- POST /api/v1/save_weights - Save model weights to disk
- POST /api/v1/save_weights_for_sampler - Save weights for SGLang sampler
- POST /api/v1/load_weights - Deprecated endpoint (returns error message)
"""
import logging
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Request

from ..services.checkpoint_service import CheckpointService
from ..core.task_manager import TaskManager
from ..core.dependencies import verify_api_key_dep
from ..storage import MetadataStorage, FuturesStorage
from ..models.requests import (
    SaveWeightsRequest,
    SaveWeightsForSamplerRequest,
)
from ..models.responses import (
    AsyncOperationResponse,
    SaveWeightsForSamplerResult,
    DeprecatedEndpointError,
)
from ..utils import generate_request_id

logger = logging.getLogger(__name__)

router = APIRouter()

def _get_runtime(request: Request):
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise RuntimeError("Training runtime state not initialized")
    return runtime


def get_checkpoint_service(request: Request) -> CheckpointService:
    """Dependency injection for CheckpointService."""
    service = getattr(request.app.state, "checkpoint_service", None)
    if service is None:
        raise RuntimeError("CheckpointService not initialized on app state")
    return service


def get_metadata_storage(request: Request) -> MetadataStorage:
    """Dependency injection for MetadataStorage."""
    storage = getattr(request.app.state, "metadata_storage", None)
    if storage is None:
        raise RuntimeError("MetadataStorage not initialized on app state")
    return storage


def get_futures_storage(request: Request) -> FuturesStorage:
    """Dependency injection for FuturesStorage."""
    storage = getattr(request.app.state, "futures_storage", None)
    if storage is None:
        raise RuntimeError("FuturesStorage not initialized on app state")
    return storage


def get_training_clients(request: Request) -> Dict[str, Dict[str, Any]]:
    """Dependency injection for training_clients."""
    runtime = _get_runtime(request)
    return runtime.training_clients


def get_task_manager(
    futures_storage: FuturesStorage = Depends(get_futures_storage)
) -> TaskManager:
    """Create TaskManager with FuturesStorage dependency."""
    return TaskManager(futures_storage)


# ============================================================================
# Checkpoint Management Endpoints
# ============================================================================

@router.post("/api/v1/save_weights", response_model=AsyncOperationResponse)
async def save_weights(
    request: SaveWeightsRequest,
    _: None = Depends(verify_api_key_dep),
    service: CheckpointService = Depends(get_checkpoint_service),
    task_manager: TaskManager = Depends(get_task_manager),
    metadata_storage: MetadataStorage = Depends(get_metadata_storage),
    training_clients: Dict = Depends(get_training_clients)
):
    """
    Save model weights to disk.
    This operation is asynchronous - use retrieve_future to check status.
    """
    request_id = generate_request_id()

    # Check if model exists
    if request.model_id not in training_clients:
        raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")

    async def execute():
        return await service.save_weights(
            model_id=request.model_id,
            request_id=request_id,
            path=request.path,
            training_clients=training_clients,
            metadata_storage=metadata_storage
        )

    # Create async task
    task_manager.create_task(
        request_id=request_id,
        operation="save_weights",
        model_id=request.model_id,
        payload=request.dict(),
        task_func=execute
    )

    return AsyncOperationResponse(
        request_id=request_id,
        model_id=request.model_id
    )


@router.post("/api/v1/save_weights_for_sampler", response_model=AsyncOperationResponse)
async def save_weights_for_sampler(
    request: SaveWeightsForSamplerRequest,
    _: None = Depends(verify_api_key_dep),
    service: CheckpointService = Depends(get_checkpoint_service),
    task_manager: TaskManager = Depends(get_task_manager),
    metadata_storage: MetadataStorage = Depends(get_metadata_storage),
    training_clients: Dict = Depends(get_training_clients)
):
    """
    Save weights for SGLang sampler.
    This operation is asynchronous - use retrieve_future to check status.
    """
    request_id = generate_request_id()

    # Check if model exists
    if request.model_id not in training_clients:
        raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")

    async def execute():
        result = await service.save_weights_for_sampler(
            model_id=request.model_id,
            request_id=request_id,
            name=request.name,
            training_clients=training_clients,
            metadata_storage=metadata_storage
        )
        return SaveWeightsForSamplerResult(**result)

    # Create async task
    task_manager.create_task(
        request_id=request_id,
        operation="save_weights_for_sampler",
        model_id=request.model_id,
        payload=request.dict(),
        task_func=execute
    )

    return AsyncOperationResponse(
        request_id=request_id,
        model_id=request.model_id
    )


@router.post("/api/v1/load_weights")
async def load_weights(
    _: None = Depends(verify_api_key_dep)
):
    """Deprecated endpoint - use checkpoint_path in create_model instead."""
    return DeprecatedEndpointError(
        error="Endpoint deprecated",
        reason="load_weights is no longer supported as a separate operation",
        solution={
            "description": "Use checkpoint_path parameter in create_model request",
            "example": {
                "base_model": "meta-llama/Llama-3.1-8B",
                "checkpoint_path": "tinker://run_abc123/weights/checkpoint_001"
            }
        }
    )
