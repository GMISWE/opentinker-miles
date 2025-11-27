"""
Models Router - HTTP Layer for Model Management

Endpoints:
- POST /api/v1/create_model - Create training model with Ray actors
- POST /api/v1/delete_model - Delete model and free GPU resources
- POST /api/v1/get_info - Get model metadata for tokenizer
- GET /api/v1/get_tokenizer - Get tokenizer information
- GET /api/v1/training_runs/{model_id} - Get training run metadata
"""
import logging
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Request

from ..services.model_service import ModelService
from ..core.task_manager import TaskManager
from ..core.dependencies import verify_api_key_dep
from ..core import SlimeArgumentBuilder
from ..storage import MetadataStorage, FuturesStorage
from ..models.requests import (
    CreateModelRequest,
    DeleteModelRequest,
    GetInfoRequest,
)
from ..models.responses import (
    AsyncOperationResponse,
    DeleteModelResponse,
    GetInfoResponse,
    ModelData,
    TrainingRunResponse,
    TokenizerInfo,
    SpecialTokens,
)
from ..utils import generate_request_id, generate_model_id

logger = logging.getLogger(__name__)

router = APIRouter()

def _get_runtime(request: Request):
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise RuntimeError("Training runtime state not initialized")
    return runtime


def get_model_service(request: Request) -> ModelService:
    """Dependency injection for ModelService."""
    service = getattr(request.app.state, "model_service", None)
    if service is None:
        raise RuntimeError("ModelService not initialized on app state")
    return service


def get_slime_builder(request: Request) -> SlimeArgumentBuilder:
    """Dependency injection for SlimeArgumentBuilder."""
    builder = getattr(request.app.state, "slime_builder", None)
    if builder is None:
        raise RuntimeError("SlimeArgumentBuilder not initialized on app state")
    return builder


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


def get_training_runs_metadata(request: Request) -> Dict[str, Dict[str, Any]]:
    """Dependency injection for training_runs_metadata."""
    runtime = _get_runtime(request)
    return runtime.training_runs_metadata


def get_task_manager(
    futures_storage: FuturesStorage = Depends(get_futures_storage)
) -> TaskManager:
    """Create TaskManager with FuturesStorage dependency."""
    return TaskManager(futures_storage)


# ============================================================================
# Model Management Endpoints
# ============================================================================

@router.post("/api/v1/create_model", response_model=AsyncOperationResponse)
async def create_model(
    request: CreateModelRequest,
    _: None = Depends(verify_api_key_dep),
    service: ModelService = Depends(get_model_service),
    task_manager: TaskManager = Depends(get_task_manager),
    slime_builder: SlimeArgumentBuilder = Depends(get_slime_builder),
    metadata_storage: MetadataStorage = Depends(get_metadata_storage),
    training_clients: Dict = Depends(get_training_clients),
    training_runs_metadata: Dict = Depends(get_training_runs_metadata)
):
    """
    Create a new training model with Ray actors and GPU resources.
    This operation is asynchronous - use retrieve_future to check status.
    """
    request_id = generate_request_id()
    model_id = generate_model_id()

    async def execute():
        return await service.create_model(
            model_id=model_id,
            request_id=request_id,
            base_model=request.base_model,
            lora_config=request.lora_config.dict() if request.lora_config else None,
            debug_train_only=request.debug_train_only,
            checkpoint_path=request.checkpoint_path,
            parallelism_config=request.parallelism_config.dict() if request.parallelism_config else None,
            slime_builder=slime_builder,
            metadata_storage=metadata_storage,
            training_clients=training_clients,
            training_runs_metadata=training_runs_metadata
        )

    # Create async task
    task_manager.create_task(
        request_id=request_id,
        operation="create_model",
        model_id=model_id,
        payload=request.dict(),
        task_func=execute
    )

    return AsyncOperationResponse(
        request_id=request_id,
        model_id=model_id
    )


@router.post("/api/v1/delete_model", response_model=DeleteModelResponse)
async def delete_model(
    request: DeleteModelRequest,
    _: None = Depends(verify_api_key_dep),
    service: ModelService = Depends(get_model_service),
    metadata_storage: MetadataStorage = Depends(get_metadata_storage),
    training_clients: Dict = Depends(get_training_clients)
):
    """Delete training client and release GPU resources."""
    try:
        result = await service.delete_model(
            model_id=request.model_id,
            training_clients=training_clients,
            metadata_storage=metadata_storage
        )
        return DeleteModelResponse(**result)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/api/v1/get_info", response_model=GetInfoResponse)
async def get_info(
    request: GetInfoRequest,
    _: None = Depends(verify_api_key_dep),
    service: ModelService = Depends(get_model_service),
    training_clients: Dict = Depends(get_training_clients)
):
    """Get model info for tokenizer initialization."""
    try:
        result = service.get_model_info(
            model_id=request.model_id,
            training_clients=training_clients
        )
        return GetInfoResponse(
            model_id=result["model_id"],
            model_data=ModelData(**result["model_data"]),
            is_lora=result["is_lora"],
            lora_rank=result["lora_rank"],
            model_name=result["model_name"]
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/api/v1/get_tokenizer")
async def get_tokenizer(
    model_id: str,
    _: None = Depends(verify_api_key_dep),
    service: ModelService = Depends(get_model_service),
    training_clients: Dict = Depends(get_training_clients)
):
    """Get tokenizer information."""
    try:
        result = service.get_tokenizer_info(
            model_id=model_id,
            training_clients=training_clients
        )
        return TokenizerInfo(
            vocab_size=result["vocab_size"],
            model_max_length=result["model_max_length"],
            pad_token_id=result["pad_token_id"],
            eos_token_id=result["eos_token_id"],
            bos_token_id=result["bos_token_id"],
            special_tokens=SpecialTokens(**result["special_tokens"]),
            hf_checkpoint=result["hf_checkpoint"]
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load tokenizer: {str(e)}")


@router.get("/api/v1/training_runs/{model_id}", response_model=TrainingRunResponse)
async def get_training_run(
    model_id: str,
    _: None = Depends(verify_api_key_dep),
    service: ModelService = Depends(get_model_service),
    metadata_storage: MetadataStorage = Depends(get_metadata_storage)
):
    """Get persistent training run metadata."""
    try:
        metadata = service.get_training_run_metadata(
            model_id=model_id,
            metadata_storage=metadata_storage
        )
        return TrainingRunResponse(**metadata)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
